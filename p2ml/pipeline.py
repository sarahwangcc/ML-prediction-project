"""End-to-end: raw CRSP/Compustat/macro -> GKX features -> rolling ML backtest."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from p2ml.config import Config
from p2ml.data.compustat import add_book_equity, merge_accounting
from p2ml.data.crsp import add_excess_return, add_forward_target, clean_crsp
from p2ml.eval.metrics import r2_oos
from p2ml.eval.splits import gkx_folds, mask
from p2ml.features.factors import SIGNAL_TO_FACTOR, available_signals, factor_composites
from p2ml.features.signals import add_funda_changes, compute_signals
from p2ml.features.transform import add_industry_dummies, add_macro_interactions, rank_normalize
from p2ml.models.registry import MODELS, OLS3_FEATURES


def build_dataset(msf, funda, link, rf, mkt_rf, macro=None, cfg: Config | None = None,
                  feature_set: str = "signals", interactions: bool = True, industries: bool = True):
    """feature_set: 'signals' (all individual signals) or 'factors' (8 composites).
    Returns (frame, feature_columns). Target column is 'y'."""
    cfg = cfg or Config()
    crsp = clean_crsp(msf, cfg.universe)
    crsp = add_excess_return(crsp, rf)
    crsp = add_forward_target(crsp, cfg.horizon)
    acct = add_funda_changes(add_book_equity(funda) if "be" not in funda else funda)
    panel = merge_accounting(crsp, acct, link, lag_months=cfg.lags.annual)
    panel = compute_signals(panel, mkt_rf)
    panel = panel.rename(columns={f"target_{cfg.horizon}m": "y"})
    panel = panel.dropna(subset=["y"])

    sigs = available_signals(panel)
    panel = rank_normalize(panel, sigs)
    if feature_set == "factors":
        comp = factor_composites(panel, sigs)
        panel = pd.concat([panel, comp], axis=1)
        chars = list(comp.columns)
    else:
        chars = sigs
    feats = list(chars)
    if interactions and macro is not None:
        panel, inter = add_macro_interactions(panel, chars, macro)
        feats += inter
    if industries:
        panel, inds = add_industry_dummies(panel)
        feats += inds
    bad = ~np.isfinite(panel[feats].to_numpy(np.float64))
    if bad.any():
        print(f"warning: {bad.sum()} non-finite feature values set to 0 (median)")
        panel[feats] = panel[feats].mask(bad, 0.0)
    keep = ["permno", "date", "y", "me", "ret", "sic2"] + feats
    return panel[keep].reset_index(drop=True), feats


def _xy(df, cols):
    return df[cols].to_numpy(np.float32), df["y"].to_numpy(np.float32)


def fit_tuned(name, train, valid, feats, verbose=False):
    """Pick hyper-parameters by validation R2_oos (GKX do not refit on train+valid)."""
    build, grid = MODELS[name]
    cols = [c for c in OLS3_FEATURES if c in feats] if name == "ols3" else feats
    Xtr, ytr = _xy(train, cols)
    Xva, yva = _xy(valid, cols)
    best = (-np.inf, None, None)
    for p in grid:
        m = build(p)
        if getattr(m, "uses_validation", False):   # NN / boosting: early stopping on valid
            m.fit(Xtr, ytr, Xva, yva)
        else:
            m.fit(Xtr, ytr)
        s = r2_oos(yva, m.predict(Xva))
        if verbose:
            print(f"    {name} {p}: valid R2 {100*s:.3f}%")
        if s > best[0]:
            best = (s, p, m)
    return best[2], best[1], best[0], cols


MODELS.setdefault("ols3", MODELS["ols"])


def run_backtest(data: pd.DataFrame, feats: list[str], model_names: list[str], cfg: Config | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions frame with one column per model, tuning log)."""
    cfg = cfg or Config()
    folds = gkx_folds(cfg.split, embargo_months=max(cfg.split.embargo_months, cfg.horizon - 1))
    preds, log = [], []
    for k, f in enumerate(folds):
        tr, va, te = (data[mask(data["date"], w)] for w in (f.train, f.valid, f.test))
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue
        out = te[["permno", "date", "y", "me"]].copy()
        for name in model_names:
            t0 = time.time()
            m, p, s, cols = fit_tuned(name, tr, va, feats)
            out[name] = m.predict(te[cols].to_numpy(np.float32))
            if not np.isfinite(out[name]).all():
                raise FloatingPointError(f"{name} produced non-finite predictions in fold {k}")
            log.append({"fold": k, "test_start": f.test[0], "model": name, "params": p,
                        "valid_R2_%": 100 * s, "test_R2_%": 100 * r2_oos(out["y"], out[name]),
                        "secs": time.time() - t0})
            if verbose:
                print(f"[fold {k} test {f.test[0].year}] {name:9s} valid R2 {100*s:6.3f}%  "
                      f"test R2 {log[-1]['test_R2_%']:6.3f}%  ({log[-1]['secs']:.1f}s)")
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(log)


def factor_importance(model, data: pd.DataFrame, cols: list[str]) -> pd.Series:
    """GKX-style importance: drop in R2 when all of a factor's signals (and their
    macro interactions) are set to 0, i.e. the cross-sectional median."""
    X = data[cols].to_numpy(np.float32)
    y = data["y"].to_numpy()
    base = r2_oos(y, model.predict(X))
    out = {}
    for fac in sorted(set(SIGNAL_TO_FACTOR.values())):
        idx = [i for i, c in enumerate(cols) if SIGNAL_TO_FACTOR.get(c.split("__x__")[0]) == fac
               or c.split("__x__")[0] == f"F_{fac}"]
        if not idx:
            continue
        Xz = X.copy(); Xz[:, idx] = 0.0
        out[fac] = base - r2_oos(y, model.predict(Xz))
    s = pd.Series(out).clip(lower=0)
    return s / s.sum() if s.sum() > 0 else s
