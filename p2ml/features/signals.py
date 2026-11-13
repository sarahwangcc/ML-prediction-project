"""Firm-level signals, grouped under Matt's 8 base factors.

Definitions follow Green, Hand & Zhang (2017) / GKX (2020) where possible.
Signals marked PROXY use monthly data where GKX use daily CRSP; swap in the
daily versions once crsp.dsf is available.

Timing convention: a row (permno, date=t) only uses information known at the
end of month t. The prediction target is the return over t+1 ... t+h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_funda_changes(funda: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year accounting changes, computed before merging to CRSP."""
    f = funda.sort_values(["gvkey", "datadate"]).copy()
    g = f.groupby("gvkey")
    gap_ok = (f["datadate"] - g["datadate"].shift(1)).dt.days.between(300, 430)
    def lag(c):
        return g[c].shift(1).where(gap_ok)
    f["agr"] = f["at"] / lag("at") - 1                          # asset growth
    f["sgr"] = f["sale"] / lag("sale") - 1                      # sales growth
    f["egr"] = f["ceq"] / lag("ceq") - 1                        # book equity growth
    f["chcsho"] = f["csho"] / lag("csho") - 1                   # net share issuance
    f["invest"] = ((f["ppegt"] - lag("ppegt")) + (f["invt"] - lag("invt"))) / lag("at")
    f["at_lag"] = lag("at")
    return f


def _roll(g, col, window, fn, min_periods=None):
    return g[col].transform(lambda s: getattr(s.rolling(window, min_periods=min_periods or window), fn)())


def _cumret(g, col, window, skip):
    """Cumulative return over months t-skip-window+1 ... t-skip."""
    return g[col].transform(
        lambda s: np.expm1(np.log1p(s).shift(skip).rolling(window, min_periods=window).sum()))


def compute_signals(panel: pd.DataFrame, mkt_rf: pd.Series | None = None) -> pd.DataFrame:
    """panel: cleaned CRSP monthly rows merged with lagged Compustat fields.
    Assumes rows are monthly and contiguous within permno (true for CRSP msf)."""
    df = panel.sort_values(["permno", "date"]).copy()
    g = df.groupby("permno", sort=False)
    me = df["me"]

    # ---------------- Momentum ----------------
    df["mom1m"] = df["ret"]                                   # short-term reversal
    df["mom6m"] = _cumret(g, "ret", 5, 1)
    df["mom12m"] = _cumret(g, "ret", 11, 1)
    df["mom36m"] = _cumret(g, "ret", 24, 12)
    df["chmom"] = df["mom6m"] - _cumret(g, "ret", 6, 6)
    df["sic2"] = (df["siccd"] // 100).astype("Int64")
    df["indmom"] = df.groupby(["date", "sic2"])["mom12m"].transform("mean")

    # ---------------- Volatility ----------------
    df["retvol"] = _roll(g, "ret", 12, "std", 6)              # PROXY (GKX: daily, 1m)
    df["maxret"] = _roll(g, "ret", 12, "max", 6)              # PROXY (GKX: max daily ret)
    if mkt_rf is not None:
        df = df.merge(mkt_rf.rename("_mkt"), left_on="date", right_index=True, how="left")
        g = df.groupby("permno", sort=False)
        df["_rm"] = df["ret"] * df["_mkt"]
        cov = _roll(g, "_rm", 60, "mean", 24) - _roll(g, "ret", 60, "mean", 24) * _roll(g, "_mkt", 60, "mean", 24)
        var_m = _roll(g, "_mkt", 60, "var", 24)
        df["beta"] = cov / var_m
        df["betasq"] = df["beta"] ** 2
        var_r = _roll(g, "ret", 60, "var", 24)
        df["idiovol"] = np.sqrt((var_r - df["beta"] ** 2 * var_m).clip(lower=0))
        df = df.drop(columns=["_mkt", "_rm"])
        g = df.groupby("permno", sort=False)

    # ---------------- Size ----------------
    df["mvel1"] = np.log(me)

    # ---------------- Liquidity ----------------
    df["_turn"] = df["vol"] / df["shrout"]
    g = df.groupby("permno", sort=False)
    df["turn"] = _roll(g, "_turn", 3, "mean", 1)
    df["std_turn"] = _roll(g, "_turn", 12, "std", 6)
    df["dolvol"] = np.log((df["prc"].abs() * df["vol"]).replace(0, np.nan))
    df["_illiq"] = df["ret"].abs() / (df["prc"].abs() * df["vol"]).replace(0, np.nan)
    g = df.groupby("permno", sort=False)
    df["ill"] = _roll(g, "_illiq", 12, "mean", 6)             # PROXY Amihud
    df = df.drop(columns=["_turn", "_illiq"])

    # ------- Accounting-based (need merged funda; NaN otherwise) -------
    def has(*cols):
        return all(c in df for c in cols)

    if has("be"):
        df["bm"] = df["be"] / me
    if has("ib"):
        df["ep"] = df["ib"] / me
    if has("sale"):
        df["sp"] = df["sale"] / me
    if has("oancf"):
        df["cfp"] = df["oancf"] / me
    # Yield
    if has("dvt"):
        df["dy"] = df["dvt"] / me
    if has("dvc", "prstkc"):
        df["payout_yld"] = (df["dvc"].fillna(0) + df["prstkc"].fillna(0)) / me
    # Quality
    if has("ib", "at"):
        df["roa"] = df["ib"] / df["at"]
    if has("ib", "be"):
        df["roe"] = df["ib"] / df["be"]
    if has("revt", "cogs", "at"):
        df["gma"] = (df["revt"] - df["cogs"]) / df["at"]
    if has("revt", "cogs", "xsga", "xint", "be"):
        df["operprof"] = (df["revt"] - df["cogs"] - df["xsga"].fillna(0) - df["xint"].fillna(0)) / df["be"]
    if has("ib", "oancf", "at"):
        df["acc"] = (df["ib"] - df["oancf"]) / df["at"]
    if has("lt"):
        df["lev"] = df["lt"] / me
    # Growth columns (agr, sgr, egr, chcsho, invest) come from add_funda_changes.

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
