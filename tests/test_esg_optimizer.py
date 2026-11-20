"""Tests for the ESG-efficient frontier optimizer (offline, synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_frontend = Path(__file__).resolve().parent.parent / "frontend"
if str(_frontend) not in sys.path:
    sys.path.insert(0, str(_frontend))

from esg_optimizer import (  # noqa: E402
    calc_stats,
    frontier_points,
    max_sharpe_portfolio,
    optimize_esg_frontier,
)

TICKERS = ["A", "B", "C", "D"]
ESG = np.array([0.55, 0.62, 0.70, 0.78])


@pytest.fixture
def market():
    """Mean excess returns and covariance for four assets with distinct Sharpe ratios."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=252)
    drift = np.array([0.0009, 0.0006, 0.0004, 0.0002])
    vol = np.array([0.020, 0.015, 0.012, 0.010])
    returns = pd.DataFrame(drift + rng.standard_normal((252, 4)) * vol, index=days, columns=TICKERS)
    prices = 100 * (1 + returns).cumprod()
    rf = pd.Series(0.0001, index=days)
    return calc_stats(prices, rf)


def test_calc_stats_forward_fills_missing_risk_free_days():
    days = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 101, 102, 101, 103]}, index=days, dtype=float)
    rf = pd.Series([0.0001, 0.0002, np.nan, np.nan, 0.0003], index=days)
    mean_excess, cov = calc_stats(prices, rf)
    returns = prices["A"].pct_change().dropna()
    expected = (returns - rf.ffill().reindex(returns.index)).mean()
    assert mean_excess["A"] == pytest.approx(expected)
    assert cov.shape == (1, 1)


@pytest.mark.parametrize("target", [0.60, 0.66, 0.72])
def test_frontier_portfolio_meets_constraints(market, target):
    mean_excess, cov = market
    result = optimize_esg_frontier(mean_excess, cov, ESG, target, min_allocation=0.01)
    weights = result.weights.values
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0.01 - 1e-8).all()
    assert float(weights @ ESG) == pytest.approx(target, abs=1e-4)


def test_tangency_sharpe_is_an_upper_bound_for_the_frontier(market):
    mean_excess, cov = market
    tangency = max_sharpe_portfolio(mean_excess, cov, min_allocation=0.01)
    points = list(frontier_points(mean_excess, cov, ESG, min_allocation=0.01, step=0.02))
    assert points, "frontier should not be empty"
    assert max(p.sharpe for p in points) <= tangency.sharpe + 1e-4


def test_frontier_is_empty_when_all_esg_scores_are_equal(market):
    mean_excess, cov = market
    assert list(frontier_points(mean_excess, cov, np.full(4, 0.7), min_allocation=0.01)) == []
