"""The NBA game model: the ratings' projection plus schedule spots, refitted each season on earlier
seasons only, then blended with the market by the weight the model has earned.

    margin = a * ratings margin + c * market-implied margin + rest and back-to-back terms
             + missing players (home minus away)      (no intercept: home court is in the ratings)
    total  = b0 + b1 * ratings total + b2 * market-implied total + back-to-back terms + missing players

The market-implied numbers come from earlier closing lines (market.py). Where there are none
(before the 2020-21 season, or a live season whose closes aren't saved yet) they fall back to the
box-score ratings' own numbers.

projection = line + w * (model - line), with w measured on earlier seasons against closing lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN_FEATURES = ["r_margin", "mk_margin", "rest_diff", "b2b_h", "b2b_a", "g7_diff", "miss_diff", "imp_diff"]
TOTAL_FEATURES = ["r_total", "mk_total", "b2b_h", "b2b_a", "both_b2b", "miss_sum"]
RIDGE = 1.0


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rh, ra = df.rest_h.clip(upper=3), df.rest_a.clip(upper=3)
    df["rest_diff"] = rh - ra
    df["b2b_h"] = (df.rest_h == 0).astype(float)
    df["b2b_a"] = (df.rest_a == 0).astype(float)
    df["both_b2b"] = df.b2b_h * df.b2b_a
    df["g7_diff"] = df.g7_h - df.g7_a
    df["has_mk"] = df["mk_margin"].notna() if "mk_margin" in df else False
    df["mk_margin"] = df["mk_margin"].fillna(df.r_margin) if "mk_margin" in df else df.r_margin
    df["mk_total"] = df["mk_total"].fillna(df.r_total) if "mk_total" in df else df.r_total
    if "miss_h" in df:
        df["miss_diff"] = df.miss_h.fillna(0) - df.miss_a.fillna(0)
        df["miss_sum"] = df.miss_h.fillna(0) + df.miss_a.fillna(0)
    else:
        df["miss_diff"] = 0.0
        df["miss_sum"] = 0.0
    # impact-rated lineup strength, home minus away (impact.py); neutral when unknown
    df["imp_diff"] = df["imp_diff"].fillna(0.0) if "imp_diff" in df else 0.0
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
    # without market ratings (early season, or no closes yet): the model without them
    fm = [f for f in MARGIN_FEATURES if f != "mk_margin"]
    ft = [f for f in TOTAL_FEATURES if f != "mk_total"]
    out = {"margin_base": dict(zip(fm, _ridge(t[fm].to_numpy(float), t.margin.to_numpy(float), intercept=False))),
           "total_base": dict(zip(["const"] + ft, _ridge(t[ft].to_numpy(float), t.total.to_numpy(float), intercept=True)))}
    if "has_mk" in t and t.has_mk.sum() >= 1000:     # with them: fit on games that had them
        tm = t[t.has_mk]
        out["margin"] = dict(zip(MARGIN_FEATURES, _ridge(tm[MARGIN_FEATURES].to_numpy(float), tm.margin.to_numpy(float), intercept=False)))
        out["total"] = dict(zip(["const"] + TOTAL_FEATURES, _ridge(tm[TOTAL_FEATURES].to_numpy(float), tm.total.to_numpy(float), intercept=True)))
    else:
        out["margin"], out["total"] = out["margin_base"], out["total_base"]
    return out


def predict(df: pd.DataFrame, coef: dict) -> pd.DataFrame:
    df = df.copy()
    def lin(c, total):
        return (c["const"] if total else 0.0) + sum(df[k] * v for k, v in c.items() if k != "const")
    has = df["has_mk"].astype(bool) if "has_mk" in df else pd.Series(False, index=df.index)
    df["m_margin"] = np.where(has, lin(coef["margin"], False), lin(coef.get("margin_base", coef["margin"]), False))
    df["m_total"] = np.where(has, lin(coef["total"], True), lin(coef.get("total_base", coef["total"]), True))
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
