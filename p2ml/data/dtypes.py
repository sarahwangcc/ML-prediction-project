"""Normalise column types coming from WRDS / parquet.

WRDS (via SQLAlchemy) and parquet round-trips often return pandas *nullable*
dtypes (Int64, Float64, string) and mixed datetime units. Merges such as
merge_asof require identical key types, and numpy code cannot handle pd.NA,
so every loader passes its output through `standardize`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_INT_COLS = ("permno", "lpermno", "permco")
ID_STR_COLS = ("gvkey",)


def standardize(df: pd.DataFrame, date_cols=()) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    for c in ID_INT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df[c].notna()]
            df[c] = df[c].astype(np.float64).astype(np.int64)
    for c in ID_STR_COLS:
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    for c in date_cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    for c in df.columns:
        if c in ID_INT_COLS or c in date_cols:
            continue
        dt = df[c].dtype
        if isinstance(dt, (pd.Int64Dtype, pd.Int32Dtype, pd.Int16Dtype, pd.Int8Dtype,
                           pd.Float64Dtype, pd.Float32Dtype)) or str(dt) in ("Int64", "Float64"):
            df[c] = df[c].astype("float64")  # pd.NA -> np.nan
        elif dt == object and c not in ID_STR_COLS:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= df[c].notna().sum() * 0.99 and df[c].notna().any():
                df[c] = conv.astype("float64")  # numeric columns that arrived as objects (e.g. Decimal)
    return df.reset_index(drop=True)
