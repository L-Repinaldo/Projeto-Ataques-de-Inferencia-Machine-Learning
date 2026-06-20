from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    y_train_true: Any
    y_train_pred: Any
    y_test_true: Any
    y_test_pred: Any
    model: Any
    preprocessor: Any
    X_train: Any = None
    X_test: Any = None
    train_indices: Any = None
    test_indices: Any = None
    target_col: str = "salario"
    seed: Any = None
    test_size: Any = None
