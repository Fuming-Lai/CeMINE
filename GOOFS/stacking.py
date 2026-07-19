#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group out-of-fold stacking: fit and predict."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

from GOOFS.base_models import build_base_estimators
from GOOFS.preprocess import make_preprocessor


def _splitter(X_train, y_train, groups_train, n_splits, random_state):
    groups_arr = None
    if groups_train is not None:
        g = groups_train.astype(str).str.strip()
        g = g.where(g.ne("") & ~g.str.lower().isin({"nan", "none"}), other=np.nan)
        if g.notna().sum() >= n_splits and g.nunique() >= n_splits:
            groups_arr = g.to_numpy()
    if groups_arr is not None:
        return GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups_arr)
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    ).split(X_train, y_train)


def fit_goofs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
    labels: Sequence,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    random_state: int = 42,
    n_splits: int = 4,
) -> dict:
    y_tr = np.asarray(y_train)
    classes = list(labels)
    n_classes = len(classes)
    bases = build_base_estimators(random_state, n_classes)
    n_base = len(bases)
    oof = np.zeros((len(X_train), n_base * n_classes), dtype=float)

    for tr_idx, val_idx in _splitter(
        X_train, y_tr, groups_train, n_splits, random_state
    ):
        pre = make_preprocessor(list(categorical_features), list(numeric_features))
        Xt_tr = pre.fit_transform(X_train.iloc[tr_idx])
        Xt_val = pre.transform(X_train.iloc[val_idx])
        for j, (_, est) in enumerate(bases):
            m = clone(est)
            m.fit(Xt_tr, y_tr[tr_idx])
            oof[val_idx, j * n_classes : (j + 1) * n_classes] = m.predict_proba(Xt_val)

    meta = LogisticRegression(
        solver="saga",
        max_iter=15000,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
    )
    meta.fit(oof, y_tr)

    pre_full = make_preprocessor(list(categorical_features), list(numeric_features))
    Xt_full = pre_full.fit_transform(X_train)
    fitted_bases = []
    for name, est in bases:
        m = clone(est)
        m.fit(Xt_full, y_tr)
        fitted_bases.append((name, m))

    return {
        "classes_": classes,
        "preprocessor": pre_full,
        "base_models": fitted_bases,
        "meta_learner": meta,
        "n_splits": n_splits,
    }


def predict_goofs(
    bundle: dict, X: pd.DataFrame, *, return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    Xt = bundle["preprocessor"].transform(X)
    X_meta = np.hstack([m.predict_proba(Xt) for _, m in bundle["base_models"]])
    y_pred = bundle["meta_learner"].predict(X_meta)
    if not return_proba:
        return y_pred
    return y_pred, bundle["meta_learner"].predict_proba(X_meta)
