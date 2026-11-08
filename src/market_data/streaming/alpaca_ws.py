"""
Async WebSocket feed for real-time market data from Alpaca.

Lifecycle: connect -> authenticate -> subscribe -> route messages to
registered async handlers. Every disconnect (error or clean close by the
server) is retried with exponential backoff. A connection only resets the
failure counter once it has stayed up for `ws_stable_after_seconds`, so a
server that accepts and immediately drops the connection still exhausts
`ws_max_reconnect_attempts`. `stop()` shuts the feed down cleanly. URL,
credentials and reconnect behavior come from DataSettings.

Alpaca protocol: every frame is a JSON array of messages tagged by 'T'
('t' trade, 'q' quote, 'b' bar, 'success'/'error'/'subscription' control).
"""

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import DataSettings, get_settings
from src.market_data.providers.alpaca_client import alpaca_credentials

logger = logging.getLogger(__name__)

Handler = Callable[[Dict[str, Any]], Awaitable[None]]

_STREAM_TYPES = {'trades', 'quotes', 'bars'}
_CONTROL_TYPES = {'success', 'error', 'subscription'}


def parse_message(msg: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Convert one Alpaca stream message into (event, payload).

    Returns None for message types the feed does not route. Raises
    KeyError/TypeError/ValueError for malformed messages.
    """
    kind = msg.get('T')
    if kind == 't':
        return 'trade', {
            'symbol': msg.get('S'),
            'price': float(msg.get('p', 0)),
            'size': float(msg.get('s', 0)),
            'timestamp': pd.Timestamp(msg['t']).to_pydatetime(),
            'condition': msg.get('c', []),
        }
    if kind == 'q':
        return 'quote', {
            'symbol': msg.get('S'),
            'bid_price': float(msg.get('bp', 0)),
            'ask_price': float(msg.get('ap', 0)),
            'bid_size': float(msg.get('bs', 0)),
            'ask_size': float(msg.get('as', 0)),
        }
    if kind == 'b':
        return 'bar', {
            'symbol': msg.get('S'),
            'timestamp': pd.Timestamp(msg['t']).to_pydatetime(),
            'open': float(msg.get('o', 0)),
            'high': float(msg.get('h', 0)),
            'low': float(msg.get('l', 0)),
            'close': float(msg.get('c', 0)),
            'volume': float(msg.get('v', 0)),
        }
    if kind in _CONTROL_TYPES:
        return 'status', msg
    return None


class AlpacaWebSocketFeed:
    """Real-time trades/quotes/bars from Alpaca with automatic reconnects."""

    def __init__(self, settings: Optional[DataSettings] = None):
        self.settings = settings if settings is not None else get_settings().data
        self.api_key, self.api_secret = alpaca_credentials(self.settings)

        self._handlers: Dict[str, List[Handler]] = {
            'trade': [], 'quote': [], 'bar': [], 'status': [],
        }
        self._connected = False
        self._reconnect_count = 0
        self._subscriptions: List[str] = []
        self._running = False
        self._ws = None

    @property
    def connected(self) -> bool:
        return self._connected

    def on_trade(self, handler: Handler) -> None:
        """Register an async handler for trade events."""
        self._handlers['trade'].append(handler)

    def on_quote(self, handler: Handler) -> None:
        """Register an async handler for quote (NBBO) events."""
        self._handlers['quote'].append(handler)

    def on_bar(self, handler: Handler) -> None:
        """Register an async handler for minute bar events."""
        self._handlers['bar'].append(handler)

    def on_status(self, handler: Handler) -> None:
        """Register an async handler for connection/control messages."""
        self._handlers['status'].append(handler)

    def backoff_delay(self, attempt: int) -> float:
        """Delay before reconnect `attempt` (1-based): base, 2x base, 4x base, ... capped."""
        s = self.settings
        return min(s.ws_base_reconnect_delay * 2 ** (attempt - 1), s.ws_max_reconnect_delay)

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        # A failing handler must not take down the feed or starve other handlers.
        for handler in self._handlers[event]:
            try:
                await handler(payload)
            except Exception:
                logger.exception("%s handler %r raised; continuing", event, handler)

    async def _dispatch(self, messages: List[Dict[str, Any]]) -> None:
        """Route each message in a frame to the handlers for its type."""
        for msg in messages:
            try:
                parsed = parse_message(msg)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed message %r: %s", msg, exc)
                continue
            if parsed is None:
                continue

            event, payload = parsed
            if event == 'status':
                if payload.get('T') == 'error':
                    logger.error("Stream error: %s", payload.get('msg'))
                else:
                    logger.info("Stream status: %s %s", payload.get('T'), payload.get('msg', ''))
            await self._emit(event, payload)

    async def _handle_frame(self, raw_message) -> None:
        try:
            messages = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse message: %s", exc)
            return
        await self._dispatch(messages if isinstance(messages, list) else [messages])

    async def _authenticate(self, ws) -> bool:
        timeout = self.settings.ws_auth_timeout_seconds
        await asyncio.wait_for(ws.recv(), timeout)  # Greeting: [{"T": "success", "msg": "connected"}]
        await ws.send(json.dumps({
            'action': 'auth', 'key': self.api_key, 'secret': self.api_secret,
        }))
        reply = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        return any(
            m.get('T') == 'success' and m.get('msg') == 'authenticated' for m in reply
        )

    async def connect_and_stream(
        self, symbols: List[str], streams: Optional[List[str]] = None
    ) -> None:
        """
        Connect and stream until stopped or reconnect attempts are exhausted.

        Args:
            symbols: Tickers to subscribe to (e.g., ['AAPL', 'MSFT']).
            streams: Any of 'trades', 'quotes', 'bars'. Defaults to
                trades and quotes.
        """
        streams = streams or ['trades', 'quotes']
        unknown = set(streams) - _STREAM_TYPES
        if unknown:
            raise ValueError(f"Unknown streams {sorted(unknown)}; use {sorted(_STREAM_TYPES)}")

        import websockets

        s = self.settings
        self._running = True
        self._reconnect_count = 0

        while self._running:
            subscribed_at: Optional[float] = None
            try:
                logger.info("Connecting to %s...", s.alpaca_ws_url)
                async with websockets.connect(
                    s.alpaca_ws_url, ping_interval=s.ws_ping_interval_seconds
                ) as ws:
                    self._ws = ws
                    self._connected = True

                    if not await self._authenticate(ws):
                        logger.error("Authentication failed; not reconnecting.")
                        break
                    logger.info("Authenticated successfully")

                    await ws.send(json.dumps(
                        {'action': 'subscribe', **{st: list(symbols) for st in streams}}
                    ))
                    self._subscriptions = list(symbols)
                    subscribed_at = time.monotonic()
                    logger.info("Subscribed to %s for %s", streams, symbols)

                    async for raw_message in ws:
                        await self._handle_frame(raw_message)

                reason = "server closed the connection"
            except (websockets.exceptions.WebSocketException, OSError, asyncio.TimeoutError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                self._connected = False
                self._ws = None

            if not self._running:
                break

            if subscribed_at is not None and time.monotonic() - subscribed_at >= s.ws_stable_after_seconds:
                self._reconnect_count = 0  # it was a healthy session, not a failure loop
            self._reconnect_count += 1
            if self._reconnect_count >= s.ws_max_reconnect_attempts:
                logger.error(
                    "Giving up after %d consecutive failed connections (last: %s)",
                    self._reconnect_count, reason,
                )
                break

            delay = self.backoff_delay(self._reconnect_count)
            logger.warning(
                "Connection lost (%s); reconnect attempt %d in %.1fs",
                reason, self._reconnect_count, delay,
            )
            await asyncio.sleep(delay)

        self._running = False
        logger.info("WebSocket feed stopped.")

    async def stop(self) -> None:
        """Gracefully stop the feed and close the connection."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
