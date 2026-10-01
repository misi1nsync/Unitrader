"""Linear-regression alpha model and its walk-forward backtests.

Model: next-hour log return ~ const + last 1h return + last 24h return
+ log(volume / 24h average volume), fitted by OLS.

Backtests: walk forward bar by bar with an expanding window (at least
MIN_TRAIN bars of history, never any future bar), trade the sign of the
forecast when it clears trading costs, and split the out-of-sample period
into N_BACKTESTS consecutive slices. The live forecast is the same model
fitted on the whole 30-day window, so the backtests test exactly what trades.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURES = ["const", "ret_1h", "ret_24h", "log_volume_ratio"]
COST = 0.0005  # 0.05% per side
MIN_TRAIN = 168  # 7 days of hourly bars before the first out-of-sample bar
N_BACKTESTS = 5
BARS_PER_YEAR = 24 * 365
_RIDGE = 1e-10


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Return features X (one row per bar), target y (next-bar log return), timestamps."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    log_close = np.log(df["close"])
    ret_1h = log_close.diff()
    ret_24h = log_close.diff(24)
    vol = df["volume"].astype(float)
    log_vol_ratio = np.log((vol + 1e-12) / (vol.rolling(24).mean() + 1e-12))
    X = np.column_stack([np.ones(len(df)), ret_1h, ret_24h, log_vol_ratio])
    y = ret_1h.shift(-1).to_numpy()
    return X, y, df["timestamp"]


def _position(pred: float) -> int:
    return 1 if pred > COST else -1 if pred < -COST else 0


def _sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0


def walk_forward(X: np.ndarray, y: np.ndarray, timestamps: pd.Series, *, sharpe_min: float) -> list[dict]:
    has_x = np.isfinite(X).all(axis=1)
    trainable = has_x & np.isfinite(y)
    Xz = np.where(trainable[:, None], np.nan_to_num(X), 0.0)
    yz = np.where(trainable, np.nan_to_num(y), 0.0)
    # Running sums of X'X and X'y let every refit be a 4x4 solve.
    xtx = np.cumsum(np.einsum("ni,nj->nij", Xz, Xz), axis=0)
    xty = np.cumsum(Xz * yz[:, None], axis=0)
    n_train = np.cumsum(trainable)

    rows, rets, prev = [], [], 0
    for t in range(1, len(y)):
        # Rows < t are known after bar t closes (row t-1's target is bar t's return).
        if not trainable[t] or n_train[t - 1] < MIN_TRAIN:
            continue
        beta = np.linalg.solve(xtx[t - 1] + _RIDGE * np.eye(len(FEATURES)), xty[t - 1])
        pos = _position(float(X[t] @ beta))
        rets.append(pos * y[t] - COST * abs(pos - prev))
        rows.append(t)
        prev = pos

    if len(rets) < N_BACKTESTS * 2:
        return []
    out = []
    for chunk in np.array_split(np.arange(len(rets)), N_BACKTESTS):
        r = np.asarray(rets)[chunk]
        out.append({
            "from": timestamps.iloc[rows[chunk[0]]].isoformat(),
            "to": timestamps.iloc[rows[chunk[-1]]].isoformat(),
            "bars": int(len(r)),
            "sharpe": round(_sharpe(r), 2),
            "return_pct": round(float(np.expm1(r.sum()) * 100), 3),
            "passed": bool(_sharpe(r) > sharpe_min),
        })
    return out


def fit(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS on every bar with a known target; forecast from the latest bar."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    Xt, yt = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    resid = yt - Xt @ beta
    dof = max(len(yt) - len(FEATURES), 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(Xt.T @ Xt)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    latest = X[np.isfinite(X).all(axis=1)][-1]
    forecast = float(latest @ beta)
    return {
        "train_bars": int(len(yt)),
        "coefficients": {
            name: {"coef": float(f"{b:.4g}"), "t_stat": round(float(b / s), 2) if s > 0 else None}
            for name, b, s in zip(FEATURES, beta, se)
        },
        "r_squared": round(float(1 - (resid @ resid) / ss_tot), 4) if ss_tot > 0 else 0.0,
        "forecast_next_1h_pct": round(forecast * 100, 4),
        "direction": {1: "long", -1: "short", 0: "flat"}[_position(forecast)],
    }


def analyze(df: pd.DataFrame, *, sharpe_min: float, passes_required: int) -> dict:
    X, y, ts = design_matrix(df)
    model = fit(X, y)
    backtests = walk_forward(X, y, ts, sharpe_min=sharpe_min)
    passed = sum(b["passed"] for b in backtests)
    return {
        **model,
        "backtests": backtests,
        "backtests_passed": passed,
        "gate_passed": len(backtests) == N_BACKTESTS and passed >= passes_required,
    }
