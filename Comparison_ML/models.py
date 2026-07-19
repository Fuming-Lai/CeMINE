#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline models: MLR, KNN, MLP, XGBoost."""

from __future__ import annotations

import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from GOOFS import make_preprocessor

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


class XGBLabelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes: int, random_state: int):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            num_class=self.n_classes,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_mlr(cat_cols, num_cols, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(cat_cols, num_cols)),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=20000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_knn(cat_cols, num_cols) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(cat_cols, num_cols)),
            (
                "model",
                KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1),
            ),
        ]
    )


def build_mlp(cat_cols, num_cols, random_state: int):
    pipe = Pipeline(
        [
            ("preprocessor", make_preprocessor(cat_cols, num_cols)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=32,
                    learning_rate_init=1e-3,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipe


def mlp_fit_kwargs(y_train) -> dict:
    if "sample_weight" in inspect.signature(MLPClassifier.fit).parameters:
        return {"model__sample_weight": compute_sample_weight("balanced", y_train)}
    return {}


def build_xgboost(cat_cols, num_cols, n_classes: int, random_state: int):
    if not HAS_XGB:
        return None
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(cat_cols, num_cols)),
            ("model", XGBLabelWrapper(n_classes, random_state)),
        ]
    )
