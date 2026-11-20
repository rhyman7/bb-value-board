"""Chance a player goes over a prop line, from his projection.

Each stat is treated as a negative binomial count with the projection as its mean and variance
mean + k * mean^2, with k fitted per stat on the tuning seasons' misses (2021-23, before any prop
line we test on). Then a bet is worth making only when the chance beats the price:
    edge = chance * decimal_price - 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).resolve().parent.parent / "docs" / "nba" / "prop_dist.json"
MAXX = 90


def fit_k(mu, actual):
    """Method of moments: E[(a - mu)^2] = mu + k mu^2, least squares for k over player-games."""
    mu, a = np.asarray(mu, float), np.asarray(actual, float)
    resid = (a - mu) ** 2 - mu
    return max(float((resid * mu ** 2).sum() / (mu ** 4).sum()), 1e-4)


def cdf_table(mu, k):
    """P(X <= x) for x = 0..MAXX, one row per mean."""
    mu = np.maximum(np.asarray(mu, float), 0.05)[:, None]
    r = 1.0 / k
    p = r / (r + mu)
    pmf = np.empty((mu.shape[0], MAXX + 1))
    pmf[:, :1] = p ** r
    for x in range(MAXX):
        pmf[:, x + 1:x + 2] = pmf[:, x:x + 1] * (x + r) / (x + 1) * (1 - p)
    return np.cumsum(pmf, axis=1)


def chances(mu, line, k):
    """(P over, P under, P push) for each (mean, line)."""
    line = np.asarray(line, float)
    c = cdf_table(mu, k)
    lo = np.clip(np.floor(line - 1e-9).astype(int), -1, MAXX)          # highest count that is under
    whole = np.isclose(line, np.round(line))
    idx = np.arange(len(line))
    under_cdf = np.where(lo >= 0, c[idx, np.clip(lo, 0, MAXX)], 0.0)    # P(X < line) for whole lines too
    at = np.where(whole, c[idx, np.clip(np.round(line).astype(int), 0, MAXX)] - under_cdf, 0.0)
    p_under = under_cdf
    p_over = 1.0 - under_cdf - at
    return p_over, p_under, at


def load():
    return json.loads(PARAMS.read_text())["k"]


def edge(p_win, p_push, dec):
    """Expected profit per unit at decimal odds, pushes returning the stake."""
    return p_win * (dec - 1.0) - (1.0 - p_win - p_push)
