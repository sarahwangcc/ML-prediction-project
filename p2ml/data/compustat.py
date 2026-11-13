"""Compustat annual (funda) / quarterly (fundq) loaders and the CRSP-Compustat link.

Expected `comp.funda` columns (standard filters indfmt='INDL', datafmt='STD',
popsrc='D', consol='C'):
    gvkey, datadate, fyear, at, lt, ceq, seq, txditc, pstkrv, pstkl, pstk,
    sale, revt, cogs, xsga, xint, ib, ni, oancf, dp, dvc, dvt, prstkc, csho,
    act, che, lct, dlc, txp, invt, capx, ppegt, emp
Link table `crsp.ccmxpf_lnkhist`: gvkey, lpermno, linktype, linkprim, linkdt, linkenddt.
"""
from __future__ import annotations

import pandas as pd

from p2ml.data.dtypes import standardize

from p2ml.data.crsp import read_table

FUNDA_COLS = [
    "gvkey", "datadate", "fyear", "at", "lt", "ceq", "seq", "txditc", "pstkrv", "pstkl", "pstk",
    "sale", "revt", "cogs", "xsga", "xint", "ib", "ni", "oancf", "dp", "dvc", "dvt", "prstkc",
    "csho", "act", "che", "lct", "dlc", "txp", "invt", "capx", "ppegt", "emp",
]

FUNDA_SQL = f"""
select {', '.join(FUNDA_COLS)}
from comp.funda
where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
  and datadate between '{{start}}' and '{{end}}'
"""

LINK_SQL = """
select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
from crsp.ccmxpf_lnkhist
where linktype in ('LU','LC') and linkprim in ('P','C')
"""


def load_funda(source, start="1950-01-01", end="2024-12-31") -> pd.DataFrame:
    if hasattr(source, "raw_sql"):
        df = source.raw_sql(FUNDA_SQL.format(start=start, end=end), date_cols=["datadate"])
    else:
        df = source.copy() if isinstance(source, pd.DataFrame) else read_table(source)
    df = standardize(df, date_cols=("datadate",))
    df = df.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    return add_book_equity(df)


def add_book_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Fama-French book equity: SE + deferred taxes - preferred stock."""
    se = df["seq"].fillna(df["ceq"] + df["pstk"].fillna(0)).fillna(df["at"] - df["lt"])
    ps = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"]).fillna(0)
    df["be"] = se + df["txditc"].fillna(0) - ps
    df.loc[df["be"] <= 0, "be"] = pd.NA
    return df


def load_link(source) -> pd.DataFrame:
    if hasattr(source, "raw_sql"):
        link = source.raw_sql(LINK_SQL, date_cols=["linkdt", "linkenddt"])
    else:
        link = source.copy() if isinstance(source, pd.DataFrame) else read_table(source)
        link.columns = link.columns.str.lower()
        link = link.rename(columns={"lpermno": "permno"})
    link = standardize(link, date_cols=("linkdt", "linkenddt"))
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    return link


def merge_accounting(crsp: pd.DataFrame, acct: pd.DataFrame, link: pd.DataFrame | None,
                     lag_months: int = 6) -> pd.DataFrame:
    """Attach the latest accounting record that was public at each CRSP month-end.

    A record with fiscal year end `datadate` becomes usable at
    datadate + `lag_months` (GKX: 6 for annual, 4 for quarterly).
    If `link` is None, `acct` must already carry a `permno` column.
    """
    acct = standardize(acct, date_cols=("datadate",))
    crsp = standardize(crsp, date_cols=("date",))
    if link is not None:
        link = standardize(link, date_cols=("linkdt", "linkenddt"))
        link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
        acct = acct.merge(link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
        acct = acct[(acct["datadate"] >= acct["linkdt"]) & (acct["datadate"] <= acct["linkenddt"])]
        acct = acct.drop(columns=["linkdt", "linkenddt"])
    acct["avail_date"] = acct["datadate"] + pd.offsets.MonthEnd(lag_months)
    acct = acct.sort_values("avail_date")
    left = crsp.sort_values("date")
    out = pd.merge_asof(left, acct.drop(columns=[c for c in ["gvkey"] if c in acct]),
                        left_on="date", right_on="avail_date", by="permno",
                        direction="backward", tolerance=pd.Timedelta(days=550))
    return out.sort_values(["permno", "date"]).reset_index(drop=True)
