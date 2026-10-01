import numpy as np

from conftest import make_bars
from unitrader import research


def test_planted_autocorrelation_passes_gate():
    out = research.analyze(make_bars("X", ar=0.4, seed=3), sharpe_min=1.5, passes_required=3)
    assert len(out["backtests"]) == research.N_BACKTESTS
    assert out["coefficients"]["ret_1h"]["coef"] > 0.2
    assert out["coefficients"]["ret_1h"]["t_stat"] > 5
    assert out["backtests_passed"] >= 3 and out["gate_passed"]


def test_random_walk_rarely_passes_gate():
    passed = [
        research.analyze(make_bars("X", seed=s), sharpe_min=1.5, passes_required=3)["gate_passed"]
        for s in range(10)
    ]
    assert sum(passed) <= 2


def test_walk_forward_uses_no_future_data():
    # Changing the final bar must not change any backtest except the last slice.
    df = make_bars("X", ar=0.4, seed=5)
    base = research.analyze(df, sharpe_min=1.5, passes_required=3)["backtests"]
    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] *= 1.5
    moved = research.analyze(df2, sharpe_min=1.5, passes_required=3)["backtests"]
    assert base[:-1] == moved[:-1]


def test_forecast_direction_follows_last_return():
    df = make_bars("X", ar=0.4, seed=3)
    X, y, _ = research.design_matrix(df)
    out = research.fit(X, y)
    last_ret = np.log(df.close.iloc[-1] / df.close.iloc[-2])
    if abs(out["forecast_next_1h_pct"]) > research.COST * 100:
        assert (out["forecast_next_1h_pct"] > 0) == (last_ret > 0)
