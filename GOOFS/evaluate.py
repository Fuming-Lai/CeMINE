#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold-out evaluation for GOOFS."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from GOOFS.stacking import fit_goofs, predict_goofs


def evaluate_goofs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    groups_train: Optional[pd.Series],
    labels: Sequence,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    random_state: int = 42,
    n_splits: int = 4,
) -> dict:
    bundle = fit_goofs(
        X_train,
        y_train,
        groups_train,
        labels,
        categorical_features,
        numeric_features,
        random_state=random_state,
        n_splits=n_splits,
    )
    y_pred, y_proba = predict_goofs(bundle, X_test, return_proba=True)
    y_te = np.asarray(y_test)
    try:
        roc = float(
            roc_auc_score(
                y_te, y_proba, multi_class="ovr", average="macro", labels=list(labels)
            )
        )
    except Exception:
        roc = None
    return {
        "model": "GOOFS",
        "accuracy_test": float(accuracy_score(y_te, y_pred)),
        "precision_macro": float(
            precision_score(y_te, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_te, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro_test": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": roc,
        "bundle": bundle,
    }
