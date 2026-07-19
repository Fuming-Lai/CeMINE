#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base estimators used inside GOOFS stacking."""

from typing import Any, List, Tuple

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def build_base_estimators(random_state: int, n_classes: int) -> List[Tuple[str, Any]]:
    bases: List[Tuple[str, Any]] = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gbt",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                random_state=random_state,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    if HAS_LGBM:
        bases.append(
            (
                "lgbm",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=n_classes,
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=random_state,
                    verbosity=-1,
                ),
            )
        )
    return bases
