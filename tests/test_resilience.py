"""Tests for the retry executor and circuit breaker (settings come from DataSettings)."""
import inspect

import pytest
from pydantic import ValidationError

import src.market_data.resilience as mod
from src.market_data.resilience import CircuitBreakerOpen, CircuitState, ResilientExecutor
from src.config.settings import reset_settings
from tests.factories import data_settings


def test_defaults_loaded_from_central_settings():
    e = ResilientExecutor(name="quotes")
    assert (e.max_retries, e.base_delay, e.max_delay) == (3, 1.0, 30.0)
    assert (e.failure_threshold, e.half_open_timeout, e.jitter) == (5, 30.0, True)
    assert e.name == "quotes"


def test_all_env_variables_loaded(monkeypatch):
    values = {
        "DATA_MAX_RETRY_ATTEMPTS": "1",
        "DATA_RETRY_BASE_DELAY": "0.25",
        "DATA_RETRY_MAX_DELAY": "1.5",
        "DATA_RETRY_JITTER": "false",
        "DATA_CIRCUIT_BREAKER_THRESHOLD": "2",
        "DATA_CIRCUIT_BREAKER_TIMEOUT_SECONDS": "9",
    }
    for name, val in values.items():
        monkeypatch.setenv(name, val)
    e = ResilientExecutor(name="api")
    assert (e.max_retries, e.base_delay, e.max_delay) == (1, 0.25, 1.5)
    assert (e.jitter, e.failure_threshold, e.half_open_timeout) == (False, 2, 9.0)


def test_env_file_is_loaded(tmp_path):
    (tmp_path / ".env").write_text(
        "DATA_MAX_RETRY_ATTEMPTS=0\n"
        "DATA_RETRY_MAX_DELAY=7.5\n"
        "DATA_RETRY_JITTER=false\n"
    )
    e = ResilientExecutor()
    assert (e.max_retries, e.max_delay, e.jitter) == (0, 7.5, False)


def test_injected_settings_override_global_settings():
    e = ResilientExecutor(settings=data_settings(max_retry_attempts=7, circuit_breaker_threshold=2))
    assert (e.max_retries, e.failure_threshold) == (7, 2)
    assert ResilientExecutor().max_retries == 3  # global settings untouched


def test_no_numeric_configuration_in_constructor():
    params = inspect.signature(ResilientExecutor).parameters
    for key in ("max_retries", "base_delay", "max_delay", "failure_threshold", "half_open_timeout", "jitter"):
        assert key not in params


def test_retry_count_and_delay_from_settings(monkeypatch):
    for name, value in (
        ("DATA_MAX_RETRY_ATTEMPTS", "2"),
        ("DATA_RETRY_BASE_DELAY", "0.5"),
        ("DATA_RETRY_MAX_DELAY", "0.7"),
        ("DATA_RETRY_JITTER", "false"),
    ):
        monkeypatch.setenv(name, value)
    waits = []
    monkeypatch.setattr(mod.time, "sleep", lambda delay: waits.append(delay))
    count = []
    def fail():
        count.append(1)
        raise ConnectionError("down")
    e = ResilientExecutor()
    with pytest.raises(ConnectionError):
        e.call(fail)
    assert len(count) == 3
    assert waits == [0.5, 0.7]
    assert e.consecutive_failures == 1


def test_non_retryable_and_unknown_errors_are_not_retried():
    count = []
    def bad_input():
        count.append(1)
        raise ValueError("bad symbol")
    e = ResilientExecutor(settings=data_settings())
    with pytest.raises(ValueError):
        e.call(bad_input)
    assert len(count) == 1
    assert e.consecutive_failures == 0


def test_threshold_counts_failed_operations_and_opens(monkeypatch):
    monkeypatch.setenv("DATA_MAX_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("DATA_CIRCUIT_BREAKER_THRESHOLD", "2")
    e = ResilientExecutor()
    def fail():
        raise TimeoutError("down")
    with pytest.raises(TimeoutError):
        e.call(fail)
    assert e.state is CircuitState.CLOSED
    with pytest.raises(TimeoutError):
        e.call(fail)
    assert e.state is CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        e.call(lambda: True)


def test_half_open_probe_closes_circuit_on_success(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    e = ResilientExecutor(settings=data_settings(
        max_retry_attempts=0, circuit_breaker_threshold=1, circuit_breaker_timeout_seconds=10,
    ))
    def down():
        raise ConnectionError("down")
    with pytest.raises(ConnectionError):
        e.call(down)
    assert e.state is CircuitState.OPEN

    clock[0] += 11
    assert e.call(lambda: "ok") == "ok"
    assert e.state is CircuitState.CLOSED


def test_reset_settings_affects_new_executors_not_existing(monkeypatch):
    monkeypatch.setenv("DATA_MAX_RETRY_ATTEMPTS", "1")
    e1 = ResilientExecutor()
    monkeypatch.setenv("DATA_MAX_RETRY_ATTEMPTS", "6")
    e2 = ResilientExecutor()  # cached settings remain at 1
    assert (e1.max_retries, e2.max_retries) == (1, 1)
    reset_settings()
    e3 = ResilientExecutor()
    assert (e1.max_retries, e3.max_retries) == (1, 6)


@pytest.mark.parametrize("name,value", [
    ("DATA_MAX_RETRY_ATTEMPTS", "-2"),
    ("DATA_RETRY_BASE_DELAY", "-1"),
    ("DATA_RETRY_MAX_DELAY", "-1"),
    ("DATA_RETRY_MAX_DELAY", "nan"),
    ("DATA_CIRCUIT_BREAKER_THRESHOLD", "0"),
    ("DATA_CIRCUIT_BREAKER_TIMEOUT_SECONDS", "-1"),
])
def test_invalid_env_variables_fail_early(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        ResilientExecutor()


def test_threshold_can_be_less_than_number_of_retries(monkeypatch):
    monkeypatch.setenv("DATA_MAX_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("DATA_CIRCUIT_BREAKER_THRESHOLD", "1")
    e = ResilientExecutor()
    assert (e.max_retries, e.failure_threshold) == (5, 1)


def test_explicit_policy_still_settable_without_numeric_configuration():
    e = ResilientExecutor(
        "quotes", retryable_exceptions=(TimeoutError,), non_retryable=(ValueError,)
    )
    assert e.retryable_exceptions == (TimeoutError,)
    assert e.non_retryable == (ValueError,)


def test_exception_policy_must_be_exception_classes():
    with pytest.raises(TypeError):
        ResilientExecutor(retryable_exceptions=[ConnectionError])
