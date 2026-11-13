"""Free data we can pull before WRDS access is approved.

- Ken French data library: FF factors (incl. RF) and momentum.
- Welch & Goyal (2008) predictors from Amit Goyal's website (updated yearly).
  Build the 8 GKX macro variables: dp, ep, bm, ntis, tbl, tms, dfy, svar.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}
# Goyal moves this file between years; check https://sites.google.com/view/agoyal145
# for the current link and pass it as `url` if this one breaks.
GOYAL_URL = "https://docs.google.com/spreadsheets/d/1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP/export?format=xlsx"


def _download(url: str) -> bytes:
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the monthly block of a Ken French CSV (stops at the annual block)."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(","))
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            if rows:
                break
            continue
        rows.append(parts)
    cols = ["yyyymm"] + [c.strip() for c in lines[start].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df.pop("yyyymm"), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date").astype(float) / 100.0  # percent -> decimal


def load_french(name: str = "ff5", path=None) -> pd.DataFrame:
    """Monthly French factors in decimals. `path` = local zip to avoid downloading."""
    raw = open(path, "rb").read() if path else _download(FRENCH_BASE + FRENCH_FILES[name])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


def build_gkx_macro(goyal: pd.DataFrame) -> pd.DataFrame:
    """Turn the Welch-Goyal monthly sheet into GKX's 8 macro predictors."""
    g = goyal.copy()
    g.columns = g.columns.str.strip().str.lower()
    g["date"] = pd.to_datetime(g["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    g = g.set_index("date")
    for c in ["index", "d12", "e12", "b/m", "ntis", "tbl", "lty", "baa", "aaa", "svar"]:
        g[c] = pd.to_numeric(g[c].astype(str).str.replace(",", ""), errors="coerce")
    macro = pd.DataFrame(index=g.index)
    macro["dp"] = np.log(g["d12"]) - np.log(g["index"])
    macro["ep"] = np.log(g["e12"]) - np.log(g["index"])
    macro["bm"] = g["b/m"]
    macro["ntis"] = g["ntis"]
    macro["tbl"] = g["tbl"]
    macro["tms"] = g["lty"] - g["tbl"]
    macro["dfy"] = g["baa"] - g["aaa"]
    macro["svar"] = g["svar"]
    return macro


def load_goyal_macro(path=None, url: str = GOYAL_URL) -> pd.DataFrame:
    raw = open(path, "rb").read() if path else _download(url)
    sheet = pd.read_excel(io.BytesIO(raw), sheet_name="Monthly")
    return build_gkx_macro(sheet)
