"""Tests for the Alpaca WebSocket feed, using a fake socket (no network)."""

import asyncio
import json

import pytest
import websockets

from src.market_data.streaming.alpaca_ws import AlpacaWebSocketFeed, parse_message
from tests.factories import data_settings

TRADE = {'T': 't', 'S': 'AAPL', 'p': 190.5, 's': 10, 't': '2024-01-02T15:00:00Z'}


def feed(**settings):
    values = dict(alpaca_api_key='key', alpaca_api_secret='secret',
                  ws_base_reconnect_delay=0, ws_max_reconnect_delay=0,
                  ws_max_reconnect_attempts=3)
    values.update(settings)
    return AlpacaWebSocketFeed(data_settings(**values))


class FakeSocket:
    def __init__(self, frames=(), authenticated=True):
        reply = 'authenticated' if authenticated else 'auth failed'
        self._replies = [
            json.dumps([{'T': 'success', 'msg': 'connected'}]),
            json.dumps([{'T': 'success' if authenticated else 'error', 'msg': reply}]),
        ]
        self.frames = list(frames)
        self.sent = []

    async def recv(self):
        return self._replies.pop(0)

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def close(self):
        self.frames.clear()

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        while self.frames:
            yield self.frames.pop(0)


class FakeConnect:
    """Replaces websockets.connect; each call yields a fresh FakeSocket."""

    def __init__(self, make_socket):
        self.make_socket = make_socket
        self.calls = 0

    def __call__(self, url, **kwargs):
        self.calls += 1
        socket = self.make_socket()

        class _Context:
            async def __aenter__(self):
                return socket

            async def __aexit__(self, *exc):
                return False

        return _Context()


def test_parse_message_routes_by_type():
    event, payload = parse_message(TRADE)
    assert event == 'trade' and payload['price'] == 190.5 and payload['symbol'] == 'AAPL'
    assert parse_message({'T': 'q', 'S': 'AAPL', 'bp': 1, 'ap': 2})[0] == 'quote'
    assert parse_message({'T': 'b', 'S': 'AAPL', 't': '2024-01-02T15:00:00Z'})[0] == 'bar'
    assert parse_message({'T': 'subscription'})[0] == 'status'
    assert parse_message({'T': 'unknown'}) is None


def test_failing_handler_does_not_block_others_and_bad_messages_are_skipped():
    f = feed()
    received = []

    async def broken(payload):
        raise RuntimeError('handler bug')

    async def recorder(payload):
        received.append(payload['symbol'])

    f.on_trade(broken)
    f.on_trade(recorder)
    asyncio.run(f._dispatch([{'T': 't', 'S': 'BAD'}, TRADE]))  # first lacks 't'

    assert received == ['AAPL']


def test_server_closing_repeatedly_exhausts_reconnect_attempts(monkeypatch):
    connect = FakeConnect(lambda: FakeSocket(frames=[json.dumps([TRADE])]))
    monkeypatch.setattr(websockets, 'connect', connect)
    f = feed(ws_stable_after_seconds=3600)
    trades = []

    async def on_trade(payload):
        trades.append(payload)

    f.on_trade(on_trade)
    asyncio.run(f.connect_and_stream(['AAPL']))

    assert connect.calls == 3
    assert len(trades) == 3


def test_authentication_failure_does_not_reconnect(monkeypatch):
    connect = FakeConnect(lambda: FakeSocket(authenticated=False))
    monkeypatch.setattr(websockets, 'connect', connect)
    asyncio.run(feed().connect_and_stream(['AAPL']))
    assert connect.calls == 1


def test_stop_ends_the_feed_without_reconnecting(monkeypatch):
    connect = FakeConnect(lambda: FakeSocket(frames=[json.dumps([TRADE])] * 5))
    monkeypatch.setattr(websockets, 'connect', connect)
    f = feed()

    async def stop_on_first_trade(payload):
        await f.stop()

    f.on_trade(stop_on_first_trade)
    asyncio.run(f.connect_and_stream(['AAPL'], streams=['trades']))

    assert connect.calls == 1


def test_subscribe_message_lists_requested_streams(monkeypatch):
    socket = FakeSocket()
    monkeypatch.setattr(websockets, 'connect', FakeConnect(lambda: socket))
    asyncio.run(feed(ws_max_reconnect_attempts=1).connect_and_stream(['AAPL'], ['trades', 'bars']))
    assert socket.sent[-1] == {'action': 'subscribe', 'trades': ['AAPL'], 'bars': ['AAPL']}


def test_unknown_stream_is_rejected():
    with pytest.raises(ValueError, match='Unknown streams'):
        asyncio.run(feed().connect_and_stream(['AAPL'], ['orders']))


def test_backoff_doubles_and_is_capped():
    f = feed(ws_base_reconnect_delay=2, ws_max_reconnect_delay=60)
    assert [f.backoff_delay(n) for n in range(1, 8)] == [2, 4, 8, 16, 32, 60, 60]
