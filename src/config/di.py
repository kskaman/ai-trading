"""

Dependency injection container for the trading framework.

This module provides a lightweight DI container that manages component
lifecycle, resolves dependencies automatically, and supports both
singleton and transient scopes.

Design decisions:

1. We use a simple registry-based container rather than a full DI
framework (like dependency-injector or injector) because the trading
system has a relatively small number of components (<20). Adding a
heavy DI framework would introduce unnecessary complexity.

2. The container supports two scopes: singleton (one instance shared
across the application) and transient (new instance each time).
Most components are singletons (data store, model registry, broker
client); strategy instances may be transient if you want to create
fresh instances per backtest run.

3. Dependencies are resolved lazily: a component is not created until
it is first requested. This enables circular dependency detection
and faster startup (unused components are never instantiated).

Common pitfalls:

- Creating dependencies inside component constructors (the anti-pattern
this container is designed to prevent). Always declare dependencies
as constructor parameters and let the container inject them.

- Forgetting to register a dependency before requesting a component
that depends on it. The container raises a clear error in this case,
but it can be confusing during initial setup. Register all components
at application startup, not ad hoc.
"""

from typing import Any, Callable, Dict, List, Optional

_SCOPES = ('singleton', 'transient')


class DependencyContainer:
    """
    Lightweight dependency injection container.

    Usage:
        container = DependencyContainer()

        # Register factories; dependencies arrive as keyword arguments
        container.register('data_store', lambda: FinancialDataStore())
        container.register(
            'ingestion_service',
            lambda data_store: DataIngestionService.from_settings(store=data_store),
            depends_on=['data_store'],
        )

        # Resolve components
        service = container.resolve('ingestion_service')
    """

    def __init__(self):
        """Create an empty container with no registered components."""
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._scopes: Dict[str, str] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._resolving: List[str] = []

    def register(self, name: str, factory: Callable,
        scope: str = 'singleton',
        depends_on: Optional[List[str]] = None) -> None:
        """
        Register a component with the container.

        Args:
            name: Unique identifier for the component (lowercase,
                underscore-separated, matching the src/ module names).
            factory: Callable that creates the component instance. It
                receives resolved dependencies as keyword arguments.
            scope: 'singleton' (one shared instance) or 'transient'
                (new instance each time).
            depends_on: Names of components to inject into the factory as
                keyword arguments.

        Raises:
            ValueError: If `scope` is not 'singleton' or 'transient'.
        """
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}, got '{scope}'")

        self._factories[name] = factory
        self._scopes[name] = scope
        self._dependencies[name] = depends_on or []

    def resolve(self, name: str) -> Any:
        """
        Resolve a component, creating it if necessary.

        For singletons, returns the cached instance after first creation.
        For transients, creates a new instance each time.

        Args:
            name: Component identifier.

        Returns:
            Component instance with all dependencies injected.

        Raises:
            KeyError: If the component is not registered.
            ValueError: If there is a circular dependency.
        """
        
        if name not in self._factories:
            raise KeyError(f"Component '{name}' not registered in container")

        # Return cached singleton
        if self._scopes[name] == 'singleton' and name in self._instances:
            return self._instances[name]

        if name in self._resolving:
            cycle = ' -> '.join(self._resolving[self._resolving.index(name):] + [name])
            raise ValueError(f"Circular dependency: {cycle}")

        # Resolve dependencies first
        self._resolving.append(name)
        try:
            deps = {dep: self.resolve(dep) for dep in self._dependencies.get(name, [])}
        finally:
            self._resolving.pop()

        # Create instance with injected dependencies
        instance = self._factories[name](**deps)

        # Cache singleton
        if self._scopes[name] == 'singleton':
            self._instances[name] = instance

        return instance

    def build_trading_system(self) -> Dict[str, Any]:
        """
        Register and resolve the components that exist today: settings, the
        data store and the ingestion service.

        Model registry, broker and audit logging get registered here once
        those modules exist. The feature engineer is not registered because
        it is configured per interval/market (FeatureEngineer(interval, market)).

        Returns:
            Dictionary of resolved components keyed by name.
        """
        from src.config.settings import get_settings
        from src.market_data.ingest import DataIngestionService
        from src.market_data.store import FinancialDataStore

        settings = get_settings()

        self.register('settings', factory=lambda: settings)
        self.register(
            'data_store',
            factory=lambda: FinancialDataStore(
                db_path=settings.data.duckdb_path,
                parquet_dir=settings.data.parquet_base_dir,
            ),
        )
        self.register(
            'ingestion_service',
            factory=lambda data_store: DataIngestionService.from_settings(
                settings.data, store=data_store
            ),
            depends_on=['data_store'],
        )

        return {name: self.resolve(name) for name in self._factories}