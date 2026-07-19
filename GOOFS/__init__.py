#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOOFS: Group Out-of-Fold Stacking for CeO2 morphology classification."""

from GOOFS.evaluate import evaluate_goofs
from GOOFS.preprocess import make_preprocessor
from GOOFS.stacking import fit_goofs, predict_goofs

__all__ = [
    "make_preprocessor",
    "fit_goofs",
    "predict_goofs",
    "evaluate_goofs",
]
