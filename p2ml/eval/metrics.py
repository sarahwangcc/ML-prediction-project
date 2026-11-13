"""Evaluation: GKX out-of-sample R2, Diebold-Mariano test, rank IC, decile portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def r2_oos(y, yhat) -> float:
    """GKX: benchmark is a zero forecast, not the historical mean."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)


def r2_xs(df: pd.DataFrame, pred: str, y: str = "y", date_col: str = "date") -> float:
    """Cross-sectional R2: demean y and the forecast within each month first.
    Measures stock selection only, ignoring how well the model predicts the
    market-wide level of returns (which dominates R2_oos for large caps)."""
    yd = df[y] - df.groupby(date_col)[y].transform("mean")
    pdm = df[pred] - df.groupby(date_col)[pred].transform("mean")
    return 1.0 - np.sum((yd - pdm) ** 2) / np.sum(yd ** 2)


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    n = len(x)
    s = np.dot(x, x) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * np.dot(x[L:], x[:-L]) / n
    return np.sqrt(s / n)


def diebold_mariano(df: pd.DataFrame, pred_a: str, pred_b: str, y: str = "y",
                    date_col: str = "date", lags: int = 0) -> tuple[float, float]:
    """GKX modified DM: average squared-error difference within each month, then a
    t-test on that monthly series. Positive stat => model B beats model A."""
    d = (df[y] - df[pred_a]) ** 2 - (df[y] - df[pred_b]) ** 2
    d12 = d.groupby(df[date_col]).mean().to_numpy()
    se = _newey_west_se(d12, lags) if lags else d12.std(ddof=1) / np.sqrt(len(d12))
    stat = d12.mean() / se
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def rank_ic(df: pd.DataFrame, pred: str, y: str = "y", date_col: str = "date") -> pd.Series:
    """Monthly Spearman correlation between prediction and realized return."""
    return df.groupby(date_col).apply(lambda g: g[pred].corr(g[y], method="spearman"), include_groups=False)


def decile_portfolios(df: pd.DataFrame, pred: str, y: str = "y", date_col: str = "date",
                      weight: str | None = None, n: int = 10) -> pd.DataFrame:
    """Sort on prediction each month; returns monthly decile returns + H-L column."""
    def one(g):
        q = pd.qcut(g[pred].rank(method="first"), n, labels=False) + 1
        if weight:
            w = g[weight]
            return (g[y] * w).groupby(q).sum() / w.groupby(q).sum()
        return g[y].groupby(q).mean()
    port = df.groupby(date_col).apply(one, include_groups=False)
    if isinstance(port, pd.Series):
        port = port.unstack()
    port["H-L"] = port[n] - port[1]
    return port


def sharpe(r: pd.Series, periods: int = 12) -> float:
    return np.sqrt(periods) * r.mean() / r.std(ddof=1)


def summarize(preds: pd.DataFrame, models: list[str], y: str = "y", weight: str | None = "me") -> pd.DataFrame:
    rows = []
    for m in models:
        sub = preds.dropna(subset=[m])
        row = {"model": m, "R2_oos_%": 100 * r2_oos(sub[y], sub[m]), "R2_xs_%": 100 * r2_xs(sub, m, y)}
        if sub.groupby("date")[m].nunique().max() <= 1:  # e.g. the 'mean' benchmark: no ranking
            rows.append(row)
            continue
        ic = rank_ic(sub, m, y)
        port_ew = decile_portfolios(sub, m, y)
        row.update({"rank_IC": ic.mean(), "IC_t": ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)),
               "HL_EW_mean_%": 100 * port_ew["H-L"].mean(), "HL_EW_Sharpe": sharpe(port_ew["H-L"])})
        if weight and weight in sub:
            port_vw = decile_portfolios(sub, m, y, weight=weight)
            row["HL_VW_Sharpe"] = sharpe(port_vw["H-L"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
