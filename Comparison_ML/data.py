#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load and prepare the CeO2 modeling table."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from ceo2_base_ce_molar_ratio import add_base_ce_molar_ratio_columns
    from ceo2_data_cleaning import normalize_addition_mode_series
except Exception:
    add_base_ce_molar_ratio_columns = None
    normalize_addition_mode_series = None

CAT_BASIC = ["precursor", "base_or_mineralizer", "facet_primary"]
NUM_BASIC = ["temperature_num", "time_num", "calc_temp_num", "calc_time_num"]
CAT_EXT = CAT_BASIC + ["addition_mode"]
NUM_EXT = NUM_BASIC + ["base_ce_molar_ratio"]


def read_csv_flexible(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"cannot decode {path}")


def valid_morphology_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    sl = s.str.lower()
    return series.notna() & s.ne("") & ~sl.isin({"nan", "unknown", "none"})


def ensure_extended(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "addition_mode" not in out.columns and "feeding_method" in out.columns:
        out["addition_mode"] = out["feeding_method"]
    elif "addition_mode" not in out.columns:
        out["addition_mode"] = ""
    if normalize_addition_mode_series is not None:
        out["addition_mode"] = normalize_addition_mode_series(out["addition_mode"])
    if (
        "base_ce_molar_ratio" not in out.columns
        or out["base_ce_molar_ratio"].isna().all()
    ) and add_base_ce_molar_ratio_columns is not None:
        out = add_base_ce_molar_ratio_columns(out)
    if "base_ce_molar_ratio" in out.columns:
        out["base_ce_molar_ratio"] = pd.to_numeric(
            out["base_ce_molar_ratio"], errors="coerce"
        ).fillna(-1.0)
    return out


def feature_columns(feature_set: str):
    if feature_set == "extended":
        return list(CAT_EXT), list(NUM_EXT)
    return list(CAT_BASIC), list(NUM_BASIC)
