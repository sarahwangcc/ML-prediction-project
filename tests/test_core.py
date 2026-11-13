import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2ml.config import SplitConfig
from p2ml.data.crsp import add_forward_target
from p2ml.eval.metrics import r2_oos
from p2ml.eval.splits import gkx_folds
from p2ml.features.transform import rank_normalize


def test_rank_normalize_range_and_missing():
    df = pd.DataFrame({"date": ["2000-01-31"] * 5, "x": [10.0, 1.0, np.nan, 5.0, 3.0]})
    out = rank_normalize(df, ["x"])
    assert out["x"].min() == -1 and out["x"].max() == 1
    assert out.loc[2, "x"] == 0.0  # missing -> median -> 0


def test_gkx_default_split_matches_paper():
    folds = gkx_folds(SplitConfig())
    assert len(folds) == 30
    f0, last = folds[0], folds[-1]
    assert f0.train[1] == pd.Timestamp("1974-12-31")
    assert f0.valid == (pd.Timestamp("1975-01-01"), pd.Timestamp("1986-12-31"))
    assert f0.test[0].year == 1987 and last.test[0].year == 2016
    assert last.valid[0].year == 2004  # validation rolls forward, fixed 12 years


def test_embargo_trims_train_end():
    f0 = gkx_folds(SplitConfig(), embargo_months=11)[0]
    assert f0.train[1] == pd.Timestamp("1974-01-31")


def test_forward_target_no_leak_across_gaps():
    df = pd.DataFrame({"permno": [1, 1, 1], "date": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-04-30"]),
                       "exret": [0.1, 0.2, 0.3]})
    out = add_forward_target(df, 1)
    assert np.isclose(out.loc[0, "target_1m"], 0.2)
    assert np.isnan(out.loc[1, "target_1m"])  # March missing -> no target


def test_r2_oos_zero_benchmark():
    y = np.array([0.01, -0.02, 0.03])
    assert r2_oos(y, np.zeros(3)) == 0.0


def _synthetic_as_ciz(msf):
    ciz = msf.rename(columns={"date": "mthcaldt", "ret": "mthret", "retx": "mthretx",
                              "prc": "mthprc", "vol": "mthvol"}).drop(columns=["shrcd", "exchcd"])
    ciz["mthprc"] = ciz["mthprc"].abs()
    ciz["primaryexch"] = msf["exchcd"].map({1: "N", 2: "A", 3: "Q"})
    for c, v in [("sharetype", "NS"), ("securitytype", "EQTY"), ("securitysubtype", "COM"),
                 ("usincflg", "Y"), ("issuertype", "CORP"), ("conditionaltype", "RW"),
                 ("tradingstatusflg", "A")]:
        ciz[c] = v
    return ciz


def test_ciz_loader_matches_siz():
    from p2ml.data.crsp import clean_crsp, load_crsp_monthly
    from p2ml.data.synthetic import make_synthetic
    msf = make_synthetic(n_firms=20, start="1990-01-31", end="1992-12-31")["msf"]
    ciz = _synthetic_as_ciz(msf)
    ciz.loc[ciz.index[:5], "sharetype"] = "AD"      # ADRs must be dropped
    ciz.loc[ciz.index[5:8], "primaryexch"] = "X"    # other exchanges dropped
    out = load_crsp_monthly(ciz)
    assert out.attrs["crsp_version"] == "ciz"
    a = clean_crsp(out)
    b = clean_crsp(load_crsp_monthly(msf))
    assert len(a) == len(b) - 8
    m = a.merge(b, on=["permno", "date"], suffixes=("_c", "_s"))
    assert np.allclose(m["ret_c"], m["ret_s"]) and np.allclose(m["me_c"], m["me_s"])


def test_wrds_nullable_dtypes_merge():
    """WRDS/parquet return nullable Int64/Float64 and mixed datetime units; the
    real-data run failed in merge_asof on Int64 vs Float64 permno."""
    from p2ml.data.compustat import load_funda, load_link
    from p2ml.data.crsp import load_crsp_monthly
    from p2ml.data.synthetic import make_synthetic
    from p2ml.config import Config, SplitConfig
    from p2ml.pipeline import build_dataset
    raw = make_synthetic(n_firms=30, start="1980-01-31", end="1990-12-31")
    ciz = _synthetic_as_ciz(raw["msf"]).convert_dtypes()
    ciz.loc[ciz.index[0], "mthret"] = pd.NA
    ciz["mthcaldt"] = ciz["mthcaldt"].astype("datetime64[us]")
    funda = raw["funda"].convert_dtypes()
    funda["datadate"] = funda["datadate"].astype("datetime64[us]")
    link = raw["link"].copy()
    link["permno"] = link["permno"].astype("Float64")
    link = link.rename(columns={"permno": "lpermno"})
    msf = load_crsp_monthly(ciz)
    assert msf["permno"].dtype == np.int64 and msf["ret"].dtype == np.float64
    cfg = Config(split=SplitConfig(first_train_year=1980, first_valid_year=1984, valid_years=2,
                                   first_test_year=1986, last_test_year=1989))
    data, feats = build_dataset(msf, load_funda(funda), load_link(link), raw["rf"], raw["mkt_rf"],
                                macro=raw["macro"], cfg=cfg)
    assert len(data) > 0 and data["bm"].abs().sum() > 0
    assert np.isfinite(data[feats].to_numpy(np.float32)).all()
