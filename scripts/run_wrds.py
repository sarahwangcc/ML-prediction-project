"""Run the pipeline on real WRDS data (CRSP CIZ + Compustat North America).

Step 1, download once and cache to data/raw/*.parquet (asks for your WRDS password):
    python scripts/run_wrds.py download --wrds-user miaoaria

Step 2, build features and run the backtest from the cache (no WRDS needed):
    python scripts/run_wrds.py run --models ols3 enet rf gbrt nn3 --goyal-path data/raw/goyal.xlsx

Smoke test on a small universe first (top 500 stocks by market cap each month):
    python scripts/run_wrds.py run --models ols3 ridge gbrt --top-n 500 --no-interactions
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2ml import quiet_warnings  # noqa: E402

quiet_warnings()

import pandas as pd  # noqa: E402

from p2ml.config import RAW_DIR, Config, SplitConfig  # noqa: E402
from p2ml.data.compustat import load_funda, load_link  # noqa: E402
from p2ml.data.crsp import load_crsp_monthly  # noqa: E402
from p2ml.data.free import load_french, load_goyal_macro  # noqa: E402
from p2ml.eval.metrics import diebold_mariano, summarize  # noqa: E402
from p2ml.pipeline import build_dataset, run_backtest  # noqa: E402


def download(args):
    import wrds
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=args.wrds_user)
    print("CRSP monthly (CIZ msf_v2, falls back to legacy SIZ msf) ...")
    msf = load_crsp_monthly(db, start=args.start, end=args.end, version=args.crsp_version)
    print(f"  {len(msf):,} rows, format = {msf.attrs.get('crsp_version')}")
    msf.to_parquet(RAW_DIR / "crsp_msf.parquet")
    print("Compustat funda ...")
    funda = load_funda(db, start="1950-01-01", end=args.end)
    print(f"  {len(funda):,} rows")
    funda.to_parquet(RAW_DIR / "comp_funda.parquet")
    print("CCM link table ...")
    load_link(db).to_parquet(RAW_DIR / "ccm_link.parquet")
    db.close()
    try:
        load_french("ff5").to_parquet(RAW_DIR / "ff5.parquet")
        print("Ken French FF5 saved")
    except Exception as e:  # noqa: BLE001
        print(f"Ken French download failed ({e}); download the FF5 CSV zip by hand and pass --ff-path")
    print(f"done -> {RAW_DIR}")


def run(args):
    msf = load_crsp_monthly(pd.read_parquet(RAW_DIR / "crsp_msf.parquet"))  # converts raw CIZ exports too
    funda = pd.read_parquet(RAW_DIR / "comp_funda.parquet")
    link = load_link(pd.read_parquet(RAW_DIR / "ccm_link.parquet"))
    ff = load_french("ff5", path=args.ff_path) if args.ff_path else pd.read_parquet(RAW_DIR / "ff5.parquet")
    macro = None
    if not args.no_interactions:
        if args.goyal_path:
            macro = load_goyal_macro(path=args.goyal_path)
        else:
            print("no --goyal-path given: running without macro interactions")

    if args.top_n:
        me = msf["prc"].abs() * msf["shrout"]
        rank = me.groupby(msf["date"]).rank(ascending=False)
        msf = msf[rank <= args.top_n]

    split = SplitConfig(last_test_year=args.last_test_year)
    cfg = Config(split=split, horizon=args.horizon)
    data, feats = build_dataset(msf, funda, link, rf=ff["RF"], mkt_rf=ff["Mkt-RF"], macro=macro,
                                cfg=cfg, feature_set=args.feature_set)
    print(f"panel: {len(data):,} stock-months, {data['permno'].nunique():,} stocks, {len(feats)} features")
    if not args.full_grid:
        from p2ml.models.registry import set_fast_mode
        set_fast_mode()
    preds, log = run_backtest(data, feats, args.models, cfg)

    out = Path(args.out); out.mkdir(exist_ok=True)
    preds.to_parquet(out / "predictions.parquet")
    log.to_csv(out / "tuning_log.csv", index=False)
    table = summarize(preds, args.models)
    table.to_csv(out / "summary.csv")
    print(table.round(3).to_string())
    base = "ols3" if "ols3" in args.models else args.models[0]
    for m in args.models:
        if m not in (base, "mean"):
            s, p = diebold_mariano(preds, base, m)
            print(f"DM {m} vs {base}: {s:.2f} (p={p:.3f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--wrds-user", default="miaoaria")
    d.add_argument("--start", default="1955-01-01")
    d.add_argument("--end", default="2024-12-31")
    d.add_argument("--crsp-version", choices=["auto", "ciz", "siz"], default="auto")
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=["mean", "ols3", "enet", "pls", "rf", "gbrt", "nn3"])
    r.add_argument("--feature-set", choices=["signals", "factors"], default="signals")
    r.add_argument("--no-interactions", action="store_true")
    r.add_argument("--goyal-path", help="Welch-Goyal xlsx downloaded from Amit Goyal's website")
    r.add_argument("--ff-path", help="local FF5 CSV zip, if the cached parquet is missing")
    r.add_argument("--horizon", type=int, default=1)
    r.add_argument("--last-test-year", type=int, default=2016, help="2016 = same as GKX; raise to extend")
    r.add_argument("--top-n", type=int, default=0, help="keep only the N largest stocks per month")
    r.add_argument("--full-grid", action="store_true")
    r.add_argument("--out", default="outputs_wrds")
    args = ap.parse_args()
    download(args) if args.cmd == "download" else run(args)


if __name__ == "__main__":
    main()
