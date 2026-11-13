"""Mapping from Matt's 8 base factors to the signals we compute.

Add signals here as they are implemented; everything downstream (feature
lists, factor composites, importance by factor) reads from FACTOR_MAP.
The sign is the expected direction of the return relation, used only to
orient composites (e.g. high volatility -> low returns).
"""
from __future__ import annotations

import pandas as pd

FACTOR_MAP: dict[str, dict[str, int]] = {
    "Momentum":   {"mom1m": -1, "mom6m": 1, "mom12m": 1, "mom36m": -1, "chmom": 1, "indmom": 1},
    "Value":      {"bm": 1, "ep": 1, "sp": 1, "cfp": 1},
    "Quality":    {"roa": 1, "roe": 1, "gma": 1, "operprof": 1, "acc": -1, "lev": -1},
    "Yield":      {"dy": 1, "payout_yld": 1},
    "Volatility": {"retvol": -1, "idiovol": -1, "beta": -1, "betasq": -1, "maxret": -1},
    "Size":       {"mvel1": -1},
    "Liquidity":  {"turn": -1, "std_turn": -1, "dolvol": -1, "ill": 1},
    "Growth":     {"agr": -1, "sgr": -1, "egr": -1, "chcsho": -1, "invest": -1},
}

SIGNAL_TO_FACTOR = {s: f for f, sigs in FACTOR_MAP.items() for s in sigs}


def available_signals(df: pd.DataFrame) -> list[str]:
    return [s for s in SIGNAL_TO_FACTOR if s in df.columns]


def factor_composites(df_ranked: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Equal-weight, sign-oriented average of rank-normalized signals per factor.
    Input must already be rank-normalized to [-1, 1]."""
    out = pd.DataFrame(index=df_ranked.index)
    for f, sigs in FACTOR_MAP.items():
        cols = [s for s in sigs if s in signals]
        if cols:
            out[f"F_{f}"] = sum(df_ranked[c] * sigs[c] for c in cols) / len(cols)
    return out
