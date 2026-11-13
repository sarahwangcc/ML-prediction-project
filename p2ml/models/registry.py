"""Model zoo with GKX-style hyper-parameter grids.

Every entry: name -> (builder(params) -> estimator, grid: list[dict]).
Estimators expose fit(X, y[, X_val, y_val]) and predict(X).
Grids are deliberately small; widen them once real data is in.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


class _Flat:
    """Wrap estimators whose predict returns 2-D output (PLS)."""
    def __init__(self, est):
        self.est = est
    def fit(self, X, y, *_):
        self.est.fit(X, y); return self
    def predict(self, X):
        return np.ravel(self.est.predict(X))


class _LGBM:
    uses_validation = True

    def __init__(self, **params):
        import lightgbm as lgb
        self.p = params
        self.m = lgb.LGBMRegressor(verbose=-1, **params)
    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb
        if X_val is not None:
            self.m.fit(X, y, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(20, verbose=False)])
        else:
            self.m.fit(X, y)
        return self
    def predict(self, X):
        return self.m.predict(X)


def _nn(n_layers):
    def build(p):
        from p2ml.models.nn import GKXNet
        return GKXNet(n_layers=n_layers, **p)
    return build, _grid(l1=[1e-5, 1e-4, 1e-3], lr=[1e-4, 1e-3])


MODELS = {
    # Benchmark: forecast every stock at the training-sample average return.
    # Its R2_oos shows how much of a model's R2 is just the market-wide level.
    "mean":     (lambda p: DummyRegressor(strategy="mean"), [{}]),
    "ols":      (lambda p: LinearRegression(), [{}]),
    "ols_huber": (lambda p: SGDRegressor(loss="huber", penalty=None, epsilon=p["epsilon"], max_iter=50, tol=1e-4),
                  _grid(epsilon=[0.05, 0.1])),
    "ridge":    (lambda p: Ridge(alpha=p["alpha"]), _grid(alpha=10.0 ** np.arange(-2, 5))),
    "lasso":    (lambda p: Lasso(alpha=p["alpha"], max_iter=5000), _grid(alpha=10.0 ** np.arange(-6, -2))),
    "enet":     (lambda p: ElasticNet(alpha=p["alpha"], l1_ratio=0.5, max_iter=5000),
                 _grid(alpha=10.0 ** np.arange(-6, -2))),
    "pcr":      (lambda p: make_pipeline(PCA(n_components=p["k"]), LinearRegression()), _grid(k=[1, 3, 5, 10, 20])),
    "pls":      (lambda p: _Flat(PLSRegression(n_components=p["k"], scale=False)), _grid(k=[1, 2, 3, 5, 10])),
    "rf":       (lambda p: RandomForestRegressor(n_estimators=300, max_depth=p["depth"], max_features=p["mf"],
                                                 min_samples_leaf=50, n_jobs=-1, random_state=0),
                 _grid(depth=[2, 4, 6], mf=[0.1, 0.3])),
    "gbrt":     (lambda p: _LGBM(n_estimators=1000, max_depth=p["depth"], num_leaves=2 ** p["depth"],
                                 learning_rate=p["lr"], objective=p["obj"], min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5),
                 _grid(depth=[1, 2], lr=[0.01, 0.1], obj=["huber"])),
    **{f"nn{k}": _nn(k) for k in range(1, 6)},
}

# OLS-3 uses only size, book-to-market and momentum (GKX benchmark).
OLS3_FEATURES = ["mvel1", "bm", "mom12m"]


def set_fast_mode():
    """Tiny grids and small ensembles for smoke tests / laptops. Not for results."""
    from sklearn.ensemble import RandomForestRegressor as RF
    MODELS["rf"] = (lambda p: RF(n_estimators=50, max_depth=p["depth"], max_features=0.3,
                                 min_samples_leaf=100, n_jobs=-1, random_state=0), _grid(depth=[3, 6]))
    MODELS["enet"] = (MODELS["enet"][0], _grid(alpha=[1e-4, 1e-3]))
    MODELS["lasso"] = (MODELS["lasso"][0], _grid(alpha=[1e-4, 1e-3]))
    MODELS["gbrt"] = (MODELS["gbrt"][0], _grid(depth=[2], lr=[0.05], obj=["huber"]))
    for k in range(1, 6):
        def build(p, k=k):
            from p2ml.models.nn import GKXNet
            return GKXNet(n_layers=k, n_ensemble=2, max_epochs=30, **p)
        MODELS[f"nn{k}"] = (build, _grid(l1=[1e-4, 1e-3], lr=[1e-3]))
