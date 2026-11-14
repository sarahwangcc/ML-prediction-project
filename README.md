# SGC Project 2: Machine-Learning Return Prediction (replicating Gu, Kelly & Xiu 2020)

Team: Sarah · Aria · Javier. We build multiple signals under Matt's 8 base factors (Momentum, Value,
Quality, Yield, Volatility, Size, Liquidity, Growth) and, following the GKX (2020) methodology, compare how
well linear models, tree models and neural networks predict next-month excess returns of individual stocks.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # unit tests
python scripts/run_demo.py                # full pipeline on synthetic data (a few minutes, CPU is enough)
python scripts/run_demo.py --feature-set factors --models ols ridge rf gbrt nn3   # 8-factor composite version
```

By default `run_demo.py` runs in "fast mode" (small grids, small ensembles) and is only meant to check
that the pipeline works; add `--full-grid` for real results.

## Directory layout

```
p2ml/
  config.py            all settings: sample splits, reporting lags, stock universe, etc.
  data/
    crsp.py            CRSP monthly file loader (CIZ msf_v2 by default, SIZ as fallback), cleaning, excess returns and h-month-ahead targets
    compustat.py       Compustat funda loader, Fama-French book equity, CCM link table, merge with a 6-month lag
    free.py            free data: Ken French factors, Welch-Goyal macro variables -> GKX's 8 macro predictors
    synthetic.py       synthetic data with exactly the same fields as CRSP/Compustat (for use before WRDS access)
  features/
    signals.py         signal construction (momentum, volatility, size, liquidity, value, yield, quality, growth)
    factors.py         FACTOR_MAP: mapping of the 8 base factors -> signals, plus factor composite scores
    transform.py       GKX cross-sectional rank normalization to [-1, 1], median imputation, macro interactions, industry dummies
  models/
    registry.py        model zoo + hyperparameter grids: ols, ols3, ols_huber, ridge, lasso, enet, pcr, pls, rf, gbrt, nn1-nn5
    nn.py              GKX neural networks (32-16-8-4-2, BatchNorm, Adam, L1, early stopping, multi-seed ensembles)
  eval/
    splits.py          expanding training window + rolling validation window + annual refit (default matches the paper: 1957-74 / 1975-86 / 1987-2016)
    metrics.py         out-of-sample R² (zero benchmark), modified Diebold-Mariano test, rank IC, decile long-short portfolios and Sharpe ratios
  pipeline.py          build_dataset() (raw data -> feature matrix) and run_backtest() (rolling training, tuning, prediction)
scripts/run_demo.py    end-to-end example (synthetic data)
scripts/run_wrds.py    real data: `download` fetches and caches, `run` runs the backtest
tests/                 unit tests (normalization, splits, no look-ahead in targets, R² definition, CIZ and SIZ give identical results)
```

## Running on real WRDS data

UofT subscribes to CRSP and Compustat North America (not Compustat Global, which this project does not need).
CRSP now defaults to the **CIZ (v2) format**, and the code reads `crsp.msf_v2` directly; if that query fails it
automatically falls back to the legacy SIZ tables `crsp.msf` + `crsp.msenames` + `crsp.msedelist`. CIZ's
`mthret` already includes delisting returns, so no separate delisting merge is needed.
CIZ has no `shrcd`/`exchcd`, so the code rebuilds them with the equivalent filters recommended by WRDS:
common stock = sharetype='NS', securitytype='EQTY', securitysubtype='COM', usincflg='Y',
issuertype in ('ACOR','CORP'); exchange = primaryexch in N/A/Q, with conditionaltype='RW' and
tradingstatusflg='A'.

```bash
pip install -r requirements.txt

# Step 1: download once from WRDS and cache to data/raw/*.parquet (prompts for your WRDS password and Duo)
python scripts/run_wrds.py download --wrds-user <your_wrds_username>

# Step 2: quick trial on a small sample (top 500 stocks by market cap each month, no macro interactions); results in a few minutes
python scripts/run_wrds.py run --models ols3 ridge gbrt --top-n 500 --no-interactions

# Step 3: full sample + macro interactions (first download the Welch-Goyal xlsx from Amit Goyal's website into data/raw/)
python scripts/run_wrds.py run --models ols3 enet pls rf gbrt nn3 --goyal-path data/raw/goyal.xlsx --full-grid
```

Results go to `outputs_wrds/`: `summary.csv` (out-of-sample R², rank IC, long-short Sharpe), `tuning_log.csv`
(hyperparameters per model per year), and `predictions.parquet` (stock-month predictions). First sanity check:
the monthly R² of OLS-3 should be close to the paper's roughly 0.16%.

The default test period 1987-2016 matches the paper; use `--last-test-year 2023` to extend it, and
`--horizon 3/6/12` to change the forecast horizon.

Note on data size: the full CRSP sample is about 3 million rows × 920 columns, roughly 11 GB in float32. Get the
pipeline running first with `--top-n`, `--no-interactions` or `--feature-set factors` before adding all
interaction terms; NN and GBRT are best run on a GPU or cluster.

## Known simplifications (TODO)

- `retvol`, `maxret` and `ill` are currently approximated from monthly data; GKX uses daily CRSP (crsp.dsf), to be swapped in once available.
- Only annual Compustat is used; quarterly signals (roaq, roeq, etc., with a 4-month lag) are still to be added.
- There are about 33 signals; the goal is 5-8 per factor, about 55 in total. Add new signals to `signals.py` and register them in `FACTOR_MAP`.
- GLM (splines + group lasso) is not implemented yet.
- Portfolio returns do not yet deduct transaction costs; to be added once we connect to Project 1's backtesting framework.

## Reference

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial
Studies*, 33(5), 2223-2273. Free versions: [NBER w25398](https://www.nber.org/papers/w25398.pdf),
[SSRN 3159577](https://papers.ssrn.com/abstract=3159577).
