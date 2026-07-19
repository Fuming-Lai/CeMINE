#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare MLR, KNN, MLP, XGBoost, and GOOFS for CeO2 morphology multiclass prediction.

Input:  CeO2_training_set.csv
Output: Comparison_ML_output/model_comparison_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from GOOFS import evaluate_goofs
from Comparison_ML.data import (
    ensure_extended,
    feature_columns,
    read_csv_flexible,
    valid_morphology_mask,
)
from Comparison_ML.metrics import run_pipeline
from Comparison_ML.models import (
    HAS_XGB,
    build_knn,
    build_mlp,
    build_mlr,
    build_xgboost,
    mlp_fit_kwargs,
)


def grouped_train_test_indices(
    df: pd.DataFrame,
    y: pd.Series,
    *,
    group_col: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint holdout split with approximate class balance."""
    if group_col not in df.columns:
        raise ValueError(
            f"Required grouping column {group_col!r} is missing. "
            "A row-wise fallback is intentionally disabled because samples from "
            "the same paper could leak across the train/test boundary."
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    groups = df[group_col].astype("string").str.strip()
    missing_group = groups.isna() | groups.eq("") | groups.str.lower().isin(
        {"nan", "none", "unknown"}
    )
    if missing_group.any():
        raise ValueError(
            f"Grouping column {group_col!r} contains "
            f"{int(missing_group.sum())} missing/unknown value(s). "
            "Every sample needs a paper identifier to guarantee a leakage-free split."
        )

    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError(
            f"Grouping column {group_col!r} has only {n_groups} unique group(s); "
            "at least two papers are required for a train/test split."
        )

    # GroupShuffleSplit keeps papers intact but does not stratify. Generate several
    # deterministic candidates and retain the one with the best class coverage and
    # closest class distribution, while always keeping every class in training.
    labels = pd.Index(y.unique())
    overall_distribution = y.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    splitter = GroupShuffleSplit(
        n_splits=min(max(32, n_groups * 4), 512),
        test_size=test_size,
        random_state=random_state,
    )
    best: tuple[tuple[float, ...], np.ndarray, np.ndarray] | None = None
    for idx_tr, idx_te in splitter.split(df, y, groups):
        y_train_candidate = y.iloc[idx_tr]
        if y_train_candidate.nunique() != len(labels):
            continue
        test_distribution = (
            y.iloc[idx_te]
            .value_counts(normalize=True)
            .reindex(labels, fill_value=0.0)
        )
        missing_test_classes = float((test_distribution == 0).sum())
        distribution_error = float(
            (test_distribution - overall_distribution).abs().sum()
        )
        size_error = abs(len(idx_te) / len(df) - test_size)
        score = (missing_test_classes, distribution_error, size_error)
        if best is None or score < best[0]:
            best = (score, idx_tr, idx_te)

    if best is None:
        raise ValueError(
            "Could not construct a grouped split that retains every morphology "
            "class in the training set. Add papers for rare classes or adjust test_size."
        )

    idx_tr, idx_te = best[1], best[2]
    train_groups = set(groups.iloc[idx_tr])
    test_groups = set(groups.iloc[idx_te])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"Internal error: {len(overlap)} group(s) occur in both train and test sets"
        )
    return idx_tr, idx_te


def run_comparison(
    input_csv: str | Path,
    output_dir: str | Path = "Comparison_ML_output",
    *,
    encoding: str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    group_col: str = "paper_id",
    oof_splits: int = 4,
    feature_set: str = "extended",
) -> pd.DataFrame:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv_flexible(Path(input_csv), encoding)
    cat_cols, num_cols = feature_columns(feature_set)
    if feature_set == "extended":
        df = ensure_extended(df)

    df = df.loc[valid_morphology_mask(df["target_morphology"])].copy()
    y = df["target_morphology"]
    X = df[cat_cols + num_cols].copy()
    labels = sorted(y.unique().tolist(), key=str)

    idx_tr, idx_te = grouped_train_test_indices(
        df,
        y,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test = X.iloc[idx_tr], X.iloc[idx_te]
    y_train, y_test = y.iloc[idx_tr], y.iloc[idx_te]
    groups_train = df.iloc[idx_tr][group_col]

    results = []
    results.append(
        run_pipeline(
            "MLR",
            build_mlr(cat_cols, num_cols, random_state),
            X_train,
            X_test,
            y_train,
            y_test,
            labels,
        )
    )
    results.append(
        run_pipeline(
            "KNN",
            build_knn(cat_cols, num_cols),
            X_train,
            X_test,
            y_train,
            y_test,
            labels,
        )
    )
    results.append(
        run_pipeline(
            "MLP",
            build_mlp(cat_cols, num_cols, random_state),
            X_train,
            X_test,
            y_train,
            y_test,
            labels,
            mlp_fit_kwargs(y_train),
        )
    )

    xgb = build_xgboost(cat_cols, num_cols, len(labels), random_state)
    if xgb is not None and HAS_XGB:
        results.append(
            run_pipeline("XGBoost", xgb, X_train, X_test, y_train, y_test, labels)
        )
    else:
        results.append(
            {
                "model": "XGBoost",
                "accuracy_test": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro_test": None,
                "roc_auc_macro_ovr": None,
            }
        )

    goofs = evaluate_goofs(
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        labels,
        cat_cols,
        num_cols,
        random_state=random_state,
        n_splits=oof_splits,
    )
    goofs.pop("bundle", None)
    results.append(goofs)

    cols = [
        "model",
        "accuracy_test",
        "precision_macro",
        "recall_macro",
        "f1_macro_test",
        "roc_auc_macro_ovr",
    ]
    out = pd.DataFrame(results)[cols]
    out_path = outdir / "model_comparison_summary.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"Saved: {out_path.resolve()}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compare MLR, KNN, MLP, XGBoost, and GOOFS."
    )
    ap.add_argument("--input_csv", default="CeO2_training_set.csv")
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--output_dir", default="Comparison_ML_output")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--group_col", default="paper_id")
    ap.add_argument("--oof_splits", type=int, default=4)
    ap.add_argument("--feature_set", choices=("basic", "extended"), default="extended")
    args = ap.parse_args()
    run_comparison(
        args.input_csv,
        args.output_dir,
        encoding=args.encoding,
        test_size=args.test_size,
        random_state=args.random_state,
        group_col=args.group_col,
        oof_splits=args.oof_splits,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
