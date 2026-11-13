"""Run the full pipeline on synthetic CRSP/Compustat-shaped data.

    python scripts/run_demo.py                      # quick: linear + trees + NN3
    python scripts/run_demo.py --models ols ridge enet rf gbrt nn1 nn3 --feature-set factors
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2ml import quiet_warnings  # noqa: E402

quiet_warnings()

from p2ml.config import Config, SplitConfig  # noqa: E402
from p2ml.data.synthetic import make_synthetic  # noqa: E402
from p2ml.eval.metrics import diebold_mariano, summarize  # noqa: E402
from p2ml.pipeline import build_dataset, run_backtest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ols", "ols3", "ridge", "enet", "pls", "rf", "gbrt", "nn3"])
    ap.add_argument("--feature-set", choices=["signals", "factors"], default="signals")
    ap.add_argument("--no-interactions", action="store_true")
    ap.add_argument("--firms", type=int, default=400)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--full-grid", action="store_true", help="use full GKX-style grids (slow)")
    args = ap.parse_args()
    if not args.full_grid:
        from p2ml.models.registry import set_fast_mode
        set_fast_mode()

    raw = make_synthetic(n_firms=args.firms, start="1975-01-31", end="2005-12-31")
    # Compressed version of the GKX split for 31 years of fake data.
    cfg = Config(split=SplitConfig(first_train_year=1975, first_valid_year=1985, valid_years=6,
                                   first_test_year=1991, last_test_year=2004, test_years=2))
    data, feats = build_dataset(raw["msf"], raw["funda"], raw["link"], raw["rf"], raw["mkt_rf"],
                                macro=None if args.no_interactions else raw["macro"], cfg=cfg,
                                feature_set=args.feature_set)
    print(f"panel: {len(data):,} stock-months, {data['permno'].nunique()} firms, {len(feats)} features")

    preds, log = run_backtest(data, feats, args.models, cfg)
    table = summarize(preds, args.models)
    print("\nOut-of-sample summary (synthetic data):")
    print(table.round(3).to_string())
    if "ols" in args.models and len(args.models) > 1:
        print("\nDiebold-Mariano vs OLS (positive = better than OLS):")
        for m in args.models:
            if m != "ols":
                s, p = diebold_mariano(preds, "ols", m)
                print(f"  {m:9s} DM={s:6.2f}  p={p:.3f}")

    out = Path(args.out); out.mkdir(exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)
    log.to_csv(out / "tuning_log.csv", index=False)
    table.to_csv(out / "summary.csv")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
