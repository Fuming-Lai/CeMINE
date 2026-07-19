#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-set metrics helpers."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_dict(name, y_true, y_pred, y_proba, labels) -> dict:
    try:
        roc = float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=list(labels)
            )
        )
    except Exception:
        roc = None
    return {
        "model": name,
        "accuracy_test": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro_test": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": roc,
    }


def run_pipeline(name, pipe, X_train, X_test, y_train, y_test, labels, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    pipe.fit(X_train, y_train, **fit_kwargs)
    y_pred = pipe.predict(X_test)
    y_proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None
    return metrics_dict(name, y_test, y_pred, y_proba, labels)
