"""Fake data in the exact CRSP / Compustat / Welch-Goyal layouts.

Lets us build and test the whole pipeline before WRDS access arrives.
Predictability is planted on purpose (latent profitability with a non-linear
effect, plus short-term reversal), so models should reach a small positive R2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n_firms: int = 400, start: str = "1975-01-31", end: str = "2005-12-31",
                   seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="ME")
    T = len(dates)

    # macro (Welch-Goyal style) -------------------------------------------------
    macro = pd.DataFrame(index=dates)
    for name, mu, sd, phi in [("dp", -3.5, 0.05, 0.98), ("ep", -2.8, 0.05, 0.97), ("bm", 0.5, 0.02, 0.98),
                              ("ntis", 0.01, 0.003, 0.95), ("tbl", 0.04, 0.002, 0.99),
                              ("tms", 0.015, 0.002, 0.97), ("dfy", 0.01, 0.001, 0.97),
                              ("svar", 0.002, 0.0008, 0.6)]:
        x = np.empty(T); x[0] = mu
        for t in range(1, T):
            x[t] = mu + phi * (x[t - 1] - mu) + sd * rng.standard_normal()
        macro[name] = x
    mkt = 0.006 + 0.045 * rng.standard_normal(T)

    # firms ------------------------------------------------------------------------
    permnos = 10000 + np.arange(n_firms)
    sic = rng.choice([1311, 2834, 3571, 3674, 4911, 5311, 6021, 7372], n_firms)
    beta = rng.uniform(0.5, 1.5, n_firms)
    birth = rng.integers(0, T // 2, n_firms)
    shares = rng.lognormal(3.0, 1.0, n_firms)  # thousands

    q = rng.standard_normal(n_firms)          # latent profitability
    price = rng.lognormal(3.0, 0.6, n_firms)
    prev_ret = np.zeros(n_firms)
    rows = []
    for t in range(T):
        q = 0.97 * q + np.sqrt(1 - 0.97 ** 2) * rng.standard_normal(n_firms)
        size_z = (np.log(price * shares) - 5) / 2
        mu = 0.002 + 0.004 * np.tanh(q) - 0.002 * size_z * (q > 0) - 0.06 * prev_ret
        ret = mu + beta * (mkt[t] - 0.006) + 0.09 * rng.standard_t(5, n_firms) / np.sqrt(5 / 3)
        ret = np.clip(ret, -0.95, 3.0)
        price = price * (1 + ret)
        vol = shares * rng.lognormal(0.0, 0.5, n_firms) * 0.08 * (1 + np.abs(ret) * 5)
        alive = t >= birth
        for i in np.where(alive)[0]:
            rows.append((permnos[i], dates[t], ret[i], ret[i] - 0.002, price[i] if rng.random() > 0.1 else -price[i],
                         shares[i], vol[i], 11, 1 + (i % 3), sic[i], q[i]))
        prev_ret = ret
    msf = pd.DataFrame(rows, columns=["permno", "date", "ret", "retx", "prc", "shrout", "vol",
                                      "shrcd", "exchcd", "siccd", "_q"])

    # Compustat funda: one record per firm-year at December fiscal year end ----------------
    dec = msf[msf["date"].dt.month == 12].copy()
    me = dec["prc"].abs() * dec["shrout"] / 1000
    at = me * rng.lognormal(0.0, 0.4, len(dec))
    funda = pd.DataFrame({
        "gvkey": dec["permno"].astype(str).str.zfill(6),
        "datadate": dec["date"], "fyear": dec["date"].dt.year,
        "at": at, "lt": at * rng.uniform(0.3, 0.7, len(dec)),
        "sale": at * rng.uniform(0.5, 1.5, len(dec)),
    })
    funda["seq"] = funda["at"] - funda["lt"]
    funda["ceq"] = funda["seq"]
    funda["txditc"] = 0.0
    funda[["pstkrv", "pstkl", "pstk"]] = 0.0
    funda["revt"] = funda["sale"]
    funda["ib"] = funda["at"] * (0.04 + 0.04 * dec["_q"].to_numpy() + 0.01 * rng.standard_normal(len(dec)))
    funda["ni"] = funda["ib"]
    funda["cogs"] = funda["sale"] * rng.uniform(0.5, 0.8, len(dec))
    funda["xsga"] = funda["sale"] * 0.1
    funda["xint"] = funda["lt"] * 0.05
    funda["dp"] = funda["at"] * 0.03
    funda["oancf"] = funda["ib"] + funda["dp"] + 0.01 * funda["at"] * rng.standard_normal(len(dec))
    funda["dvc"] = np.maximum(funda["ib"], 0) * rng.uniform(0, 0.5, len(dec))
    funda["dvt"] = funda["dvc"]
    funda["prstkc"] = np.maximum(funda["ib"], 0) * rng.uniform(0, 0.2, len(dec))
    funda["csho"] = dec["shrout"].to_numpy() / 1000
    funda["act"] = funda["at"] * 0.4
    funda["che"] = funda["at"] * 0.1
    funda["lct"] = funda["lt"] * 0.4
    funda["dlc"] = funda["lt"] * 0.1
    funda["txp"] = 0.0
    funda["invt"] = funda["at"] * 0.1
    funda["capx"] = funda["at"] * 0.05
    funda["ppegt"] = funda["at"] * 0.5
    funda["emp"] = funda["at"] / 50

    link = pd.DataFrame({"gvkey": funda["gvkey"].unique()})
    link["permno"] = link["gvkey"].astype(int)
    link["linktype"], link["linkprim"] = "LC", "P"
    link["linkdt"], link["linkenddt"] = pd.Timestamp("1900-01-01"), pd.NaT

    rf = pd.Series(macro["tbl"].to_numpy() / 12, index=dates, name="RF")
    mkt_rf = pd.Series(mkt - rf.to_numpy(), index=dates, name="Mkt-RF")
    return {"msf": msf.drop(columns="_q"), "funda": funda, "link": link,
            "macro": macro, "rf": rf, "mkt_rf": mkt_rf}
