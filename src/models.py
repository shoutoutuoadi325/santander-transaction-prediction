"""Model factories for the Santander classification pipeline.

Hyperparameters mirror 方案_v2 §8 / 分工 T6. Random state = 42.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def make_logistic_regression(**overrides: Any) -> LogisticRegression:
    params: Dict[str, Any] = dict(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return LogisticRegression(**params)


def make_gaussian_nb(**overrides: Any) -> GaussianNB:
    return GaussianNB(**overrides)


def make_decision_tree(**overrides: Any) -> DecisionTreeClassifier:
    params: Dict[str, Any] = dict(
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return DecisionTreeClassifier(**params)


def make_random_forest(**overrides: Any) -> RandomForestClassifier:
    params: Dict[str, Any] = dict(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(overrides)
    return RandomForestClassifier(**params)


def make_lightgbm(**overrides: Any):
    """LightGBM is heavy; import lazily so the rest of ``src`` works without it."""
    from lightgbm import LGBMClassifier  # noqa: WPS433 - lazy import on purpose

    params: Dict[str, Any] = dict(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    early_stopping_rounds: int = 100,
    log_period: int = 200,
    **overrides: Any,
):
    """Train a LightGBM with early stopping on the supplied validation set."""
    import lightgbm as lgb  # noqa: WPS433

    model = make_lightgbm(**overrides)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(log_period),
        ],
    )
    return model


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    """Return ``P(target=1)`` for any sklearn-compatible classifier."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]
