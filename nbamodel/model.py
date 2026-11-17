"""The NBA game model: the ratings' projection plus schedule spots, refitted each season on earlier
seasons only, then blended with the market by the weight the model has earned.

    margin = a * ratings margin + rest and back-to-back terms + missing players (home minus away)
             (no intercept: home court is in the ratings)
    total  = b0 + b1 * ratings total + back-to-back terms + missing players (both teams)

projection = line + w * (model - line), with w measured on earlier seasons against closing lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN_FEATURES = ["r_margin", "rest_diff", "b2b_h", "b2b_a", "g7_diff", "miss_diff"]
TOTAL_FEATURES = ["r_total", "b2b_h", "b2b_a", "both_b2b", "miss_sum"]
RIDGE = 1.0


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rh, ra = df.rest_h.clip(upper=3), df.rest_a.clip(upper=3)
    df["rest_diff"] = rh - ra
    df["b2b_h"] = (df.rest_h == 0).astype(float)
    df["b2b_a"] = (df.rest_a == 0).astype(float)
    df["both_b2b"] = df.b2b_h * df.b2b_a
    df["g7_diff"] = df.g7_h - df.g7_a
    if "miss_h" in df:
        df["miss_diff"] = df.miss_h.fillna(0) - df.miss_a.fillna(0)
        df["miss_sum"] = df.miss_h.fillna(0) + df.miss_a.fillna(0)
    else:
        df["miss_diff"] = 0.0
        df["miss_sum"] = 0.0
    return df


def _ridge(X, y, intercept):
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    pen = np.full(X.shape[1], RIDGE)
    if intercept:
        pen[0] = 0.0
    return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)


def fit(train: pd.DataFrame) -> dict:
    """Coefficients from games in `train` (enough games played that the ratings mean something)."""
    t = train[train.n_this >= 60]
    bm = _ridge(t[MARGIN_FEATURES].to_numpy(float), t.margin.to_numpy(float), intercept=False)
    bt = _ridge(t[TOTAL_FEATURES].to_numpy(float), t.total.to_numpy(float), intercept=True)
    return {"margin": dict(zip(MARGIN_FEATURES, bm)), "total": dict(zip(["const"] + TOTAL_FEATURES, bt))}


def predict(df: pd.DataFrame, coef: dict) -> pd.DataFrame:
    df = df.copy()
    df["m_margin"] = sum(df[k] * v for k, v in coef["margin"].items())
    df["m_total"] = coef["total"]["const"] + sum(df[k] * v for k, v in coef["total"].items() if k != "const")
    return df


def market_weight(df: pd.DataFrame, model_col: str, line_col: str, actual_col: str):
    """How far toward the model the line should move: least squares of (actual - line) on
    (model - line), through the origin. Returns (w, standard error, games)."""
    d = df.dropna(subset=[line_col])
    x = (d[model_col] - d[line_col]).to_numpy(float)
    y = (d[actual_col] - d[line_col]).to_numpy(float)
    if len(x) < 200 or not np.any(x):
        return 0.0, None, len(x)
    w = float(x @ y / (x @ x))
    resid = y - w * x
    se = float(np.sqrt(resid @ resid / (len(x) - 1) / (x @ x)))
    return w, se, len(x)
