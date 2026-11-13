"""GKX feed-forward networks NN1-NN5 (32-16-8-4-2 pyramid) in PyTorch.

ReLU, batch norm, Adam, L1 weight penalty, early stopping on the validation
set, and an ensemble over random seeds (GKX use 10).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

HIDDEN = [32, 16, 8, 4, 2]


class _MLP(nn.Module):
    def __init__(self, n_in: int, n_layers: int):
        super().__init__()
        layers, d = [], n_in
        for h in HIDDEN[:n_layers]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GKXNet:
    uses_validation = True

    def __init__(self, n_layers: int = 3, l1: float = 1e-5, lr: float = 1e-3, batch_size: int = 1024,
                 max_epochs: int = 100, patience: int = 5, n_ensemble: int = 5, seed: int = 0,
                 device: str | None = None):
        self.n_layers, self.l1, self.lr = n_layers, l1, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.n_ensemble, self.seed = n_ensemble, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models_: list[_MLP] = []

    def _fit_one(self, X, y, Xv, yv, seed):
        torch.manual_seed(seed)
        m = _MLP(X.shape[1], self.n_layers).to(self.device)
        with torch.no_grad():
            # Start from the "predict the mean" model: returns are mostly noise, so a
            # randomly initialised output layer would begin far worse than zero.
            m.net[-1].weight.mul_(0.01)
            m.net[-1].bias.fill_(y.mean().item())
        opt = torch.optim.Adam(m.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        n = len(X)
        for _ in range(self.max_epochs):
            m.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = ((m(X[idx]) - y[idx]) ** 2).mean()
                l1 = sum(p.abs().sum() for name, p in m.named_parameters() if "weight" in name)
                (loss + self.l1 * l1).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = ((m(Xv) - yv) ** 2).mean().item()
            if v < best - 1e-10:
                best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        m.load_state_dict(best_state)
        m.eval()
        return m

    def fit(self, X, y, X_val=None, y_val=None):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        X = np.asarray(X, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6   # train-only input scaling
        X, y = t((X - self.mu_) / self.sd_), t(y)
        if X_val is None:  # fall back to last 20% as early-stopping set
            k = int(len(X) * 0.8)
            X, Xv, y, yv = X[:k], X[k:], y[:k], y[k:]
        else:
            Xv, yv = t((np.asarray(X_val, dtype=np.float32) - self.mu_) / self.sd_), t(y_val)
        self.models_ = [self._fit_one(X, y, Xv, yv, self.seed + s) for s in range(self.n_ensemble)]
        return self

    def predict(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return np.mean([m(X).cpu().numpy() for m in self.models_], axis=0)
