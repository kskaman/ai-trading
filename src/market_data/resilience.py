"""Synchronous, thread-safe retry executor with a three-state circuit breaker.

The breaker counts *failed operations*, not individual retry attempts.
One executor should be shared for each external service/provider.

Only exceptions listed in ``retryable_exceptions`` are retried and counted
toward the breaker. The default (built-in ConnectionError/TimeoutError) does
NOT cover third-party clients: ccxt raises ccxt.NetworkError and alpaca-py
raises requests exceptions, neither of which subclass the built-ins. Each
provider passes its own retryable exceptions.
"""

from __future__ import annotations

import inspect
import logging
import random
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from src.config.settings import get_settings

if TYPE_CHECKING:
    from src.config.settings import DataSettings

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitBreakerOpen(RuntimeError):
    """The operation was rejected locally without contacting the provider."""


class CircuitState(Enum):
    """Breaker state: CLOSED admits calls, OPEN rejects them, HALF_OPEN runs one probe."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ResilientExecutor:
    """Retry transient errors and stop admitting operations after repeated failures.

    - A normal operation gets one initial attempt plus ``max_retries`` retries.
    - An operation counts as one circuit failure only after all retries fail.
    - At ``failure_threshold`` consecutive failed operations, the breaker opens.
    - After the cooldown, exactly one HALF_OPEN probe (no retries) is admitted.
    - The state is thread-safe within one Python process.
    - Provide a request timeout in the actual API client; this class cannot
      enforce an execution timeout on an arbitrary synchronous callable.
    """

    def __init__(
        self,
        name: str = "provider",
        non_retryable: tuple[type[Exception], ...] = (ValueError,),
        *,
        retryable_exceptions: tuple[type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
        ),
        settings: Optional[DataSettings] = None,
    ) -> None:
        """
        Create an executor in the CLOSED state.

        Retry counts, delays, the failure threshold and the cooldown come
        from DataSettings (DATA_* env vars / .env), which validates them.

        Args:
            name: Label used in log messages and errors, e.g. 'alpaca'.
            non_retryable: Errors raised at once and never counted by the
                breaker, even if they also appear in `retryable_exceptions`.
            retryable_exceptions: Transient errors to retry and count. Any
                other exception is raised at once without counting. Pass the
                types your client library really raises.
            settings: Data settings; defaults to the global settings.

        Raises:
            TypeError: If either exception argument is not a tuple of
                Exception classes.
        """
        cfg = settings if settings is not None else get_settings().data

        for key, value in (
            ("non_retryable", non_retryable),
            ("retryable_exceptions", retryable_exceptions),
        ):
            if not isinstance(value, tuple) or not all(
                isinstance(t, type) and issubclass(t, Exception) for t in value
            ):
                raise TypeError(f"{key} must be a tuple of Exception classes")

        self.name = name
        self.max_retries = cfg.max_retry_attempts
        self.base_delay = float(cfg.retry_base_delay)
        self.max_delay = float(cfg.retry_max_delay)
        self.failure_threshold = cfg.circuit_breaker_threshold
        self.half_open_timeout = float(cfg.circuit_breaker_timeout_seconds)
        self.non_retryable = non_retryable
        self.retryable_exceptions = retryable_exceptions
        self.jitter = cfg.retry_jitter

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._generation = 0  # Ignore completions from an older circuit era.

    @property
    def state(self) -> CircuitState:
        """Current breaker state (read under the lock)."""
        with self._lock:
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Failed operations in a row since the last success; resets to 0 on success."""
        with self._lock:
            return self._consecutive_failures

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run `func(*args, **kwargs)` with retries and circuit breaking.

        Args:
            func: A synchronous callable, e.g. a provider's API method.
            *args: Positional arguments passed to `func`.
            **kwargs: Keyword arguments passed to `func`.

        Returns:
            Whatever `func` returns.

        Raises:
            CircuitBreakerOpen: If the breaker is OPEN (still cooling down),
                a probe is already running, or the breaker opened while this
                call was waiting to retry.
            TypeError: If `func` returns a coroutine or other awaitable.
            Exception: Whatever `func` last raised, once retries are used up
                or immediately for a non-retryable error.
        """
        is_probe, generation = self._admit()
        if is_probe:
            try:
                result = self._invoke(func, *args, **kwargs)
            except BaseException as exc:
                # Only an explicitly retryable error establishes provider
                # unavailability. Invalid inputs and programming errors do not.
                if isinstance(exc, self.retryable_exceptions) and not isinstance(
                    exc, self.non_retryable
                ):
                    self._probe_failed(generation)
                else:
                    self._probe_aborted(generation)
                raise
            else:
                self._probe_succeeded(generation)
                return result

        for attempt in range(self.max_retries + 1):
            if attempt:
                self._check_still_closed(generation)
            try:
                result = self._invoke(func, *args, **kwargs)
            except Exception as exc:
                if isinstance(exc, self.non_retryable) or not isinstance(
                    exc, self.retryable_exceptions
                ):
                    raise
                if attempt == self.max_retries:
                    self._operation_failed(generation)
                    logger.error(
                        "[%s] operation failed after %d attempts: %s",
                        self.name,
                        attempt + 1,
                        exc,
                    )
                    raise
                delay = min(self.max_delay, self.base_delay * (2.0 ** min(attempt, 1023)))
                if self.jitter:
                    delay = random.uniform(0.0, delay)
                logger.warning(
                    "[%s] retryable error on attempt %d/%d (%s); retry in %.3fs",
                    self.name,
                    attempt + 1,
                    self.max_retries + 1,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)  # Never hold the state lock during backoff.
            else:
                self._operation_succeeded(generation)
                return result

        raise AssertionError("unreachable")

    @staticmethod
    def _invoke(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call `func` once and refuse async functions.

        Args:
            func: The callable to run.
            *args: Positional arguments for `func`.
            **kwargs: Keyword arguments for `func`.

        Returns:
            What `func` returned.

        Raises:
            TypeError: If `func` returned an awaitable. A coroutine is closed
                first so Python does not warn that it was never awaited.
        """
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()  # Avoid unawaited-coroutine warnings.
            raise TypeError("ResilientExecutor supports only synchronous callables")
        return result

    def _admit(self) -> tuple[bool, int]:
        """
        Decide whether a new call may run, moving OPEN to HALF_OPEN after the cooldown.

        Returns:
            (is_probe, generation). `is_probe` is True when this call is the
            single HALF_OPEN probe. `generation` identifies the circuit era,
            so a slow call cannot change state that has moved on since.

        Raises:
            CircuitBreakerOpen: If the cooldown has not passed or a probe is
                already running.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return False, self._generation
            if self._state is CircuitState.HALF_OPEN:
                raise CircuitBreakerOpen(f"[{self.name}] HALF_OPEN: probe in progress")

            if self._opened_at is None:
                raise RuntimeError("Circuit state is OPEN but opened_at is missing")
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self.half_open_timeout:
                raise CircuitBreakerOpen(
                    f"[{self.name}] OPEN: retry after "
                    f"{self.half_open_timeout - elapsed:.1f}s"
                )
            self._state = CircuitState.HALF_OPEN
            logger.warning("[%s] circuit HALF_OPEN: running one probe", self.name)
            return True, self._generation

    def _check_still_closed(self, generation: int) -> None:
        """
        Stop retrying if another call opened the breaker during our backoff.

        Args:
            generation: The era returned by `_admit` for this call.

        Raises:
            CircuitBreakerOpen: If the state or era has changed.
        """
        with self._lock:
            if self._state is not CircuitState.CLOSED or self._generation != generation:
                raise CircuitBreakerOpen(f"[{self.name}] circuit opened during retry backoff")

    def _operation_succeeded(self, generation: int) -> None:
        """
        Reset the failure count after a successful normal call.

        Args:
            generation: The call's era; ignored if the circuit has moved on.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED and self._generation == generation:
                self._consecutive_failures = 0

    def _operation_failed(self, generation: int) -> None:
        """
        Count a call whose retries all failed; open the breaker at the threshold.

        Args:
            generation: The call's era; ignored if the circuit has moved on.
        """
        with self._lock:
            if self._state is not CircuitState.CLOSED or self._generation != generation:
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._generation += 1
                logger.error(
                    "[%s] circuit OPEN after %d failed operations",
                    self.name,
                    self._consecutive_failures,
                )

    def _probe_succeeded(self, generation: int) -> None:
        """
        Close the breaker after the HALF_OPEN probe worked.

        Args:
            generation: The probe's era; ignored if the circuit has moved on.
        """
        with self._lock:
            if self._state is CircuitState.HALF_OPEN and self._generation == generation:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._opened_at = None
                self._generation += 1
                logger.info("[%s] circuit CLOSED: probe succeeded", self.name)

    def _probe_failed(self, generation: int) -> None:
        """
        Reopen the breaker with a fresh cooldown after a retryable probe error.

        Args:
            generation: The probe's era; ignored if the circuit has moved on.
        """
        with self._lock:
            if self._state is CircuitState.HALF_OPEN and self._generation == generation:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()  # Fresh cooldown.
                self._generation += 1
                logger.error("[%s] circuit OPEN: probe failed", self.name)

    def _probe_aborted(self, generation: int) -> None:
        """
        Return to OPEN without a new cooldown after a non-retryable probe error.

        Bad input says nothing about the provider's health, so the expired
        cooldown is kept and the next call may probe straight away.

        Args:
            generation: The probe's era; ignored if the circuit has moved on.
        """
        with self._lock:
            if self._state is CircuitState.HALF_OPEN and self._generation == generation:
                # The exception is not evidence of provider health; restore
                # OPEN while retaining its expired cooldown. Another caller
                # may probe immediately (not before).
                self._state = CircuitState.OPEN
                self._generation += 1
                logger.info("[%s] probe aborted by a non-retryable error", self.name)
