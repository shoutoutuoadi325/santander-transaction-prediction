"""Data loading and splitting utilities for the Santander project.

Convention: random_state = 42 everywhere. Stratified splits because the
positive class is ~10%.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42

DEFAULT_TRAIN_PATH = os.path.join("..", "data", "train.csv")
DEFAULT_TEST_PATH = os.path.join("..", "data", "test.csv")


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the 200 ``var_*`` feature column names in original order."""
    return [c for c in df.columns if c.startswith("var_")]


def load_data(
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs from disk.

    Parameters
    ----------
    train_path, test_path
        Paths to the Kaggle CSVs. Default values assume the notebook is run
        from ``notebooks/`` and the CSVs live in ``data/``.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def split_xy(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop ``ID_code`` / ``target`` and return ``(X, y)``."""
    X = train.drop(columns=["ID_code", "target"])
    y = train["target"]
    return X, y


def split_data(
    train: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Single 80/20 stratified split for quick analyses.

    The project default for the modelling track is 5-fold StratifiedKFold,
    but a single split is convenient for EDA-adjacent notebooks.
    """
    X, y = split_xy(train)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_valid, y_train, y_valid


def stratified_kfold_split(
    train: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, valid_idx)`` pairs from a 5-fold stratified split.

    Examples
    --------
    >>> for fold, (tr, va) in enumerate(stratified_kfold_split(train)):
    ...     X_tr, X_va = X.iloc[tr], X.iloc[va]
    ...     y_tr, y_va = y.iloc[tr], y.iloc[va]
    """
    X, y = split_xy(train)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(X, y):
        yield tr_idx, va_idx


def report_missing(*frames: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing-value counts for arbitrary frames.

    Returns one row per frame with ``total_missing`` and the number of
    columns containing any NaNs. The Santander dataset is expected to have
    zero missing values; if anything appears, the report makes that obvious.
    """
    rows = []
    for i, df in enumerate(frames):
        total = int(df.isnull().sum().sum())
        cols_with_na = int((df.isnull().sum() > 0).sum())
        rows.append(
            {
                "frame_index": i,
                "rows": len(df),
                "cols": df.shape[1],
                "total_missing": total,
                "cols_with_missing": cols_with_na,
            }
        )
    return pd.DataFrame(rows)
