"""Annual refit with an expanding training window and a rolling validation window."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from p2ml.config import SplitConfig


@dataclass
class Fold:
    train: tuple[pd.Timestamp, pd.Timestamp]
    valid: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


def _ye(y):
    return pd.Timestamp(f"{y}-12-31")


def gkx_folds(cfg: SplitConfig, embargo_months: int | None = None) -> list[Fold]:
    """Fold k: train [first_train_year, V_k-1], valid [V_k, V_k+valid_years-1], test next block.
    V_k moves forward by test_years each refit, so training expands and validation rolls.

    embargo: drop the last `embargo` months of train and valid so that h-month
    targets of the training rows never overlap the following window."""
    emb = cfg.embargo_months if embargo_months is None else embargo_months
    folds = []
    test_start = cfg.first_test_year
    shift = 0
    while test_start <= cfg.last_test_year:
        v0 = cfg.first_valid_year + shift
        v1 = v0 + cfg.valid_years - 1
        t1 = min(test_start + cfg.test_years - 1, cfg.last_test_year)
        tr_end = _ye(v0 - 1) - pd.offsets.MonthEnd(emb)
        va_end = _ye(v1) - pd.offsets.MonthEnd(emb)
        folds.append(Fold(train=(pd.Timestamp(f"{cfg.first_train_year}-01-01"), tr_end),
                          valid=(pd.Timestamp(f"{v0}-01-01"), va_end),
                          test=(pd.Timestamp(f"{test_start}-01-01"), _ye(t1))))
        test_start += cfg.test_years
        shift += cfg.test_years
    return folds


def mask(dates: pd.Series, window: tuple) -> pd.Series:
    return (dates >= window[0]) & (dates <= window[1])
