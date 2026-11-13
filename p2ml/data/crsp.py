"""CRSP monthly stock file loader, for both CRSP formats on WRDS.

CIZ (v2, current default): `crsp.msf_v2` (= stkmthsecuritydata). Columns used:
    permno, mthcaldt, mthret, mthretx, mthprc, shrout, mthvol, siccd,
    primaryexch, sharetype, securitytype, securitysubtype, usincflg,
    issuertype, conditionaltype, tradingstatusflg
  In CIZ, `mthret` already includes delisting returns, so no delisting merge.

SIZ (v1, now under "Legacy" on WRDS): `crsp.msf` joined with `crsp.msenames`:
    permno, date, ret, retx, prc, shrout, vol, shrcd, exchcd, siccd
  plus `crsp.msedelist` (permno, dlstdt, dlret, dlstcd) for delisting returns.

Both are converted to the SIZ-style names used everywhere else in the code:
permno, date, ret, retx, prc, shrout, vol, shrcd, exchcd, siccd.
Works with a wrds.Connection, a CSV/Parquet export, or a DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2ml.config import UniverseConfig
from p2ml.data.dtypes import standardize

CIZ_MSF_SQL = """
select permno, mthcaldt, mthret, mthretx, mthprc, shrout, mthvol, siccd,
       primaryexch, sharetype, securitytype, securitysubtype, usincflg,
       issuertype, conditionaltype, tradingstatusflg
from crsp.msf_v2
where mthcaldt between '{start}' and '{end}'
"""

SIZ_MSF_SQL = """
select a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol,
       b.shrcd, b.exchcd, b.siccd
from crsp.msf as a
left join crsp.msenames as b
  on a.permno = b.permno and b.namedt <= a.date and a.date <= b.nameendt
where a.date between '{start}' and '{end}'
"""
CRSP_MSF_SQL = SIZ_MSF_SQL  # backwards-compatible name

SIZ_DELIST_SQL = "select permno, dlstdt, dlret, dlstcd from crsp.msedelist"

CIZ_EXCH_MAP = {"N": 1, "A": 2, "Q": 3}  # NYSE, NYSE American (AMEX), NASDAQ


def read_table(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def ciz_to_siz(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CIZ columns to SIZ names and rebuild shrcd/exchcd from CIZ flags.

    Common-stock screen = WRDS's recommended CIZ equivalent of shrcd in (10, 11):
    sharetype='NS', securitytype='EQTY', securitysubtype='COM', usincflg='Y',
    issuertype in ('ACOR','CORP'). Exchange screen = primaryexch in N/A/Q with
    conditionaltype='RW' and tradingstatusflg='A' (regular-way, actively traded).
    Rows failing the screens get shrcd/exchcd = NaN, so clean_crsp drops them.
    """
    d = df.copy()
    d.columns = d.columns.str.lower()
    d = d.rename(columns={"mthcaldt": "date", "mthret": "ret", "mthretx": "retx",
                          "mthprc": "prc", "mthvol": "vol"})
    def flag(col, ok):
        return d[col].isin(ok) if col in d else pd.Series(True, index=d.index)
    common = (flag("sharetype", ["NS"]) & flag("securitytype", ["EQTY"]) &
              flag("securitysubtype", ["COM"]) & flag("usincflg", ["Y"]) &
              flag("issuertype", ["ACOR", "CORP"]))
    regular = flag("conditionaltype", ["RW"]) & flag("tradingstatusflg", ["A"])
    d["shrcd"] = np.where(common, 11, np.nan)
    d["exchcd"] = d["primaryexch"].map(CIZ_EXCH_MAP).where(regular) if "primaryexch" in d else np.nan
    return d


def detect_version(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    return "ciz" if {"mthcaldt", "mthret"} & cols else "siz"


def load_crsp_monthly(source, delist_source=None, start="1957-01-01", end="2024-12-31",
                      version: str = "auto") -> pd.DataFrame:
    """Load CRSP monthly data and return it in SIZ-style column names.

    source: wrds.Connection, file path (CSV/Parquet) or DataFrame.
    version: 'ciz' (crsp.msf_v2, default on WRDS since 2024), 'siz' (legacy
    crsp.msf), or 'auto' (wrds: try CIZ, fall back to SIZ; files: detect by columns).
    """
    dl = None
    if hasattr(source, "raw_sql"):  # wrds.Connection
        if version in ("ciz", "auto"):
            try:
                msf = source.raw_sql(CIZ_MSF_SQL.format(start=start, end=end), date_cols=["mthcaldt"])
                version = "ciz"
            except Exception:
                if version == "ciz":
                    raise
                try:  # a failed query leaves the Postgres transaction aborted
                    source.connection.rollback()
                except Exception:
                    pass
                version = "siz"
        if version == "siz":
            msf = source.raw_sql(SIZ_MSF_SQL.format(start=start, end=end), date_cols=["date"])
            dl = source.raw_sql(SIZ_DELIST_SQL, date_cols=["dlstdt"])
    else:
        msf = source.copy() if isinstance(source, pd.DataFrame) else read_table(source)
        if version == "auto":
            version = detect_version(msf)
        if delist_source is not None and version == "siz":
            dl = delist_source.copy() if isinstance(delist_source, pd.DataFrame) else read_table(delist_source)
    msf.columns = msf.columns.str.lower()
    if version == "ciz":
        msf = ciz_to_siz(msf)
    msf = standardize(msf, date_cols=("date",))
    msf["date"] = msf["date"] + pd.offsets.MonthEnd(0)
    if dl is not None:
        dl = standardize(dl, date_cols=("dlstdt",))
        msf = apply_delisting_returns(msf, dl)
    msf.attrs["crsp_version"] = version
    return msf


def apply_delisting_returns(msf: pd.DataFrame, dl: pd.DataFrame) -> pd.DataFrame:
    """Combine regular and delisting returns: (1+ret)(1+dlret)-1.
    Missing performance-related delisting returns are set to -30% (Shumway 1997)."""
    dl = dl.copy()
    dl.columns = dl.columns.str.lower()
    dl["date"] = pd.to_datetime(dl["dlstdt"]) + pd.offsets.MonthEnd(0)
    if "dlstcd" in dl:
        perf = dl["dlstcd"].between(500, 599)
        dl.loc[perf & dl["dlret"].isna(), "dlret"] = -0.30
    out = msf.merge(dl[["permno", "date", "dlret"]], on=["permno", "date"], how="left")
    has = out["dlret"].notna()
    out.loc[has, "ret"] = (1 + out.loc[has, "ret"].fillna(0)) * (1 + out.loc[has, "dlret"]) - 1
    return out.drop(columns="dlret")


def clean_crsp(msf: pd.DataFrame, cfg: UniverseConfig | None = None) -> pd.DataFrame:
    """Apply universe filters and compute market equity (in $ millions)."""
    cfg = cfg or UniverseConfig()
    df = msf[msf["shrcd"].isin(cfg.share_codes) & msf["exchcd"].isin(cfg.exchange_codes)].copy()
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"] = df["prc"].abs()  # negative price = bid/ask midpoint in CRSP
    df["me"] = df["prc"] * df["shrout"] / 1000.0
    df = df[df["me"] > 0]
    if cfg.min_price > 0:
        df = df[df["prc"] >= cfg.min_price]
    df = df.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")
    if cfg.drop_microcaps:
        nyse = df[df["exchcd"] == 1].groupby("date")["me"].quantile(0.2).rename("me_p20")
        df = df.merge(nyse, on="date", how="left")
        df = df[df["me"] >= df["me_p20"]].drop(columns="me_p20")
    return df.reset_index(drop=True)


def add_excess_return(df: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """rf: monthly risk-free rate in decimals indexed by month-end date (e.g. Ken French RF/100)."""
    out = df.merge(rf.rename("rf"), left_on="date", right_index=True, how="left")
    out["exret"] = out["ret"] - out["rf"]
    return out


def add_forward_target(df: pd.DataFrame, horizon: int = 1, col: str = "exret") -> pd.DataFrame:
    """Target at row (i, t) = cumulative excess return over t+1 ... t+horizon.
    Uses calendar months so gaps in a stock's history do not leak the wrong month."""
    df = df.sort_values(["permno", "date"]).copy()
    logr = np.log1p(df[col])
    wide = logr.groupby([df["permno"], df["date"]]).first().unstack("permno").sort_index()
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="ME")
    wide = wide.reindex(full_idx)
    fwd = wide.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    fwd = np.expm1(fwd).stack().rename(f"target_{horizon}m")
    fwd.index.names = ["date", "permno"]
    return df.merge(fwd.reset_index(), on=["date", "permno"], how="left")
