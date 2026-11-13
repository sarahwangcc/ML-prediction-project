"""GKX feature transforms: cross-sectional rank to [-1, 1], median fill,
macro interactions and industry dummies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_normalize(df: pd.DataFrame, cols: list[str], date_col: str = "date") -> pd.DataFrame:
    """Each month, map each characteristic's cross-sectional rank linearly to [-1, 1].
    Missing values are filled with the cross-sectional median, i.e. 0 after mapping."""
    out = df.copy()
    grp = out.groupby(date_col)[cols]
    r = grp.rank(method="average")
    n = grp.transform("count")
    scaled = 2 * (r - 1) / (n - 1).where(n > 1) - 1
    scaled = scaled.where(n > 1, 0.0)
    out[cols] = scaled.fillna(0.0)
    return out


def add_macro_interactions(df: pd.DataFrame, chars: list[str], macro: pd.DataFrame,
                           date_col: str = "date") -> tuple[pd.DataFrame, list[str]]:
    """z_{i,t} = (1, x_t) kron c_{i,t}. Macro values at month t are known at t.
    Returns the frame and the list of new interaction columns."""
    m = macro.copy()
    m.index = pd.to_datetime(m.index) + pd.offsets.MonthEnd(0)
    m = m.add_prefix("macro_")
    out = df.merge(m, left_on=date_col, right_index=True, how="left")
    new_cols = {}
    for mc in m.columns:
        mv = out[mc].fillna(out[mc].median()).to_numpy()
        for c in chars:
            new_cols[f"{c}__x__{mc[6:]}"] = out[c].to_numpy() * mv
    inter = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out.drop(columns=m.columns), inter], axis=1)
    return out, list(inter.columns)


def add_industry_dummies(df: pd.DataFrame, col: str = "sic2") -> tuple[pd.DataFrame, list[str]]:
    # drop_first: the full set of dummies sums to 1 and is collinear with the intercept
    d = pd.get_dummies(df[col].astype("Int64").astype(str), prefix="ind", dtype=np.float32, drop_first=True)
    return pd.concat([df, d], axis=1), list(d.columns)
