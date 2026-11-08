from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Every model must implement:
        - fit(): Train the model on features and targets
        - predict(): Generate predictions from features
        - get_feature_importance(): Return feature importance rankings

    Models are trained using walk-forward validation and registered
    in MLflow for versioning and deployment tracking.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier (e.g., 'xgboost_regression', 'lstm_temporal')."""
        pass

    @abstractmethod
    def fit(
        self, X: np.ndarray, 
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> 'BaseModel':
        """
        Train the model on features and targets.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            sample_weights: Optional sample weights for weighted fitting.

        Returns:
            Self (for method chaining).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from features.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Prediction array (n_samples,) for regression or
            (n_samples, n_classes) for classification.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance rankings as a DataFrame."""
        pass

    def save(self, path: str) -> None:
        """Save model to disk for later loading."""
        raise NotImplementedError("Subclasses must implement save()")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load a model from disk."""
        raise NotImplementedError("Subclasses must implement load()")