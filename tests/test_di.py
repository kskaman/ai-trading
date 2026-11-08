"""Tests for the dependency injection container."""

import pytest

from src.config.di import DependencyContainer


def test_singleton_and_transient_scopes():
    c = DependencyContainer()
    c.register('single', lambda: object())
    c.register('fresh', lambda: object(), scope='transient')

    assert c.resolve('single') is c.resolve('single')
    assert c.resolve('fresh') is not c.resolve('fresh')


def test_dependencies_are_injected_as_keyword_arguments():
    c = DependencyContainer()
    c.register('config', lambda: {'x': 1})
    c.register('service', lambda config: ('service', config), depends_on=['config'])
    assert c.resolve('service') == ('service', {'x': 1})


def test_circular_dependency_raises_value_error():
    c = DependencyContainer()
    c.register('a', lambda b: None, depends_on=['b'])
    c.register('b', lambda a: None, depends_on=['a'])
    with pytest.raises(ValueError, match='a -> b -> a'):
        c.resolve('a')


def test_unknown_component_and_scope():
    c = DependencyContainer()
    with pytest.raises(KeyError):
        c.resolve('missing')
    with pytest.raises(ValueError):
        c.register('x', lambda: None, scope='request')


def test_build_trading_system_wires_existing_components():
    components = DependencyContainer().build_trading_system()
    try:
        assert set(components) == {'settings', 'data_store', 'ingestion_service'}
        assert components['ingestion_service'].store is components['data_store']
    finally:
        components['data_store'].close()
