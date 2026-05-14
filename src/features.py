"""Feature engineering helpers for the Santander dataset."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(
    X_train: pd.DataFrame,
    X_valid: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """Fit StandardScaler on train, apply to valid/test.

    Returns numpy arrays so downstream sklearn estimators don't trip on the
    column-name mismatch between transformed arrays and the original frame.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid) if X_valid is not None else None
    X_test_s = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_valid_s, X_test_s, scaler


def add_frequency_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    suffix: str = "_freq",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Append count-encoded features to train and test.

    Frequencies are computed on the union of train+test values, which is the
    canonical Santander recipe (the test set is roughly the same size as the
    train set so both halves contribute meaningful counts).

    Returns the augmented frames plus the list of new column names.
    """
    feature_cols = list(feature_cols)
    full = pd.concat([train[feature_cols], test[feature_cols]], axis=0)

    new_cols: List[str] = [f"{col}{suffix}" for col in feature_cols]

    train_extra = {}
    test_extra = {}
    for col, new_col in zip(feature_cols, new_cols):
        freq = full[col].value_counts()
        train_extra[new_col] = train[col].map(freq).astype("int32").values
        test_extra[new_col] = test[col].map(freq).astype("int32").values

    train = pd.concat(
        [train, pd.DataFrame(train_extra, index=train.index)], axis=1
    )
    test = pd.concat(
        [test, pd.DataFrame(test_extra, index=test.index)], axis=1
    )
    return train, test, new_cols


def attach_extra_feature(
    X: pd.DataFrame,
    values: np.ndarray,
    name: str,
) -> pd.DataFrame:
    """Append a single 1-D numpy column to ``X`` as a named feature.

    Used to bolt the cross-line ``anomaly_score`` and ``cluster_id`` features
    onto the base feature matrix without mutating the input.
    """
    if len(values) != len(X):
        raise ValueError(
            f"Length mismatch: feature '{name}' has {len(values)} rows but X has {len(X)}"
        )
    out = X.copy()
    out[name] = values
    return out
