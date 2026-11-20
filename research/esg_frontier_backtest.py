"""ESG-efficient frontier and quarterly-rebalanced backtest.

This is the research script behind the dashboard. It does two things:

1. Frontier: for one estimation window, find the maximum-Sharpe long-only
   portfolio at each portfolio ESG target, and compare it with the
   unconstrained (Markowitz) tangency portfolio and each single asset.
2. Backtest: rebalance at the start of every quarter to the maximum-Sharpe
   portfolio at a fixed ESG target, estimating returns and covariances from the
   previous quarter only (no look-ahead). Compare against equal weight, the
   unconstrained max-Sharpe portfolio, and SPY.

Usage:
    python research/esg_frontier_backtest.py
    python research/esg_frontier_backtest.py --tickers AAPL MSFT NVDA --esg-scores 0.70 0.80 0.60

Figures are written to docs/images/ and the results table to docs/results.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))

from esg_optimizer import (  # noqa: E402
    asset_sharpes,
    calc_stats,
    fetch_price_history,
    fetch_risk_free_rate,
    frontier_points,
    max_sharpe_portfolio,
    optimize_esg_frontier,
)

# Universe and normalized ESG scores (0-1, higher is better) from the original SIG notebook.
DEFAULT_TICKERS = ["CROX", "DECK", "EME", "COST", "DE"]
DEFAULT_ESG_SCORES = [0.62, 0.63, 0.56, 0.72, 0.73]
TRADING_DAYS = 252


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--esg-scores", nargs="+", type=float, default=DEFAULT_ESG_SCORES,
                   help="one normalized ESG score per ticker, in the same order")
    p.add_argument("--frontier-start", type=dt.date.fromisoformat, default=dt.date(2025, 1, 1))
    p.add_argument("--frontier-end", type=dt.date.fromisoformat, default=dt.date(2026, 1, 1))
    p.add_argument("--backtest-start", type=dt.date.fromisoformat, default=dt.date(2019, 10, 1))
    p.add_argument("--backtest-end", type=dt.date.fromisoformat, default=dt.date.today())
    p.add_argument("--targets", nargs="+", type=float, default=[0.65, 0.70],
                   help="portfolio ESG targets to backtest")
    p.add_argument("--min-allocation", type=float, default=0.01, help="long-only floor per asset")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "docs")
    args = p.parse_args()
    if len(args.tickers) != len(args.esg_scores):
        p.error(f"got {len(args.tickers)} tickers but {len(args.esg_scores)} ESG scores")
    return args


def plot_frontier(prices: pd.DataFrame, rf: pd.Series, esg: pd.Series, args, path: pathlib.Path) -> None:
    window = prices.loc[args.frontier_start:args.frontier_end]
    mean_excess, cov = calc_stats(window, rf)
    points = list(frontier_points(mean_excess, cov, esg.values, args.min_allocation, step=0.005))
    tangency = max_sharpe_portfolio(mean_excess, cov, args.min_allocation)
    tangency_esg = float(np.dot(tangency.weights, esg))
    singles = asset_sharpes(mean_excess, cov, esg.index)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([pt.target_esg for pt in points], [pt.sharpe for pt in points], lw=2.2, label="ESG-efficient frontier")
    ax.scatter(esg, singles, s=60, color="#555", zorder=3, label="Single assets")
    for ticker in esg.index:
        ax.annotate(ticker, (esg[ticker], singles[ticker]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.scatter([tangency_esg], [tangency.sharpe], marker="*", s=220, color="#d62728", zorder=4,
               label=f"Max-Sharpe (no ESG target), ESG = {tangency_esg:.2f}")
    ax.set_xlabel("Portfolio ESG score")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title(f"ESG-efficient frontier, {args.frontier_start:%b %Y} to {args.frontier_end:%b %Y}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def quarter_starts(index: pd.Index, start: dt.date) -> list:
    """First trading day of each calendar quarter on or after start."""
    days = pd.Series(pd.to_datetime(index), index=index)
    days = days[days.index >= start]
    first = days.groupby(days.dt.to_period("Q")).head(1)
    return list(first.index)


def backtest(prices: pd.DataFrame, rf: pd.Series, choose_weights, rebalance_dates: list) -> tuple[pd.Series, int]:
    """Daily returns of a portfolio rebalanced on rebalance_dates.

    Weights chosen at rebalance date d_i use prices from d_{i-1} through d_i and
    are held from the day after d_i through d_{i+1}.
    """
    returns = prices.pct_change()
    pieces, fallbacks, weights = [], 0, None
    for prev, cur, nxt in zip(rebalance_dates[:-1], rebalance_dates[1:], rebalance_dates[2:] + [None]):
        mean_excess, cov = calc_stats(prices.loc[prev:cur], rf)
        try:
            weights = choose_weights(mean_excess, cov)
        except RuntimeError:
            fallbacks += 1  # keep last quarter's weights if the optimizer fails
            if weights is None:
                weights = pd.Series(1 / len(prices.columns), index=prices.columns)
        held = returns.loc[cur:nxt].iloc[1:] if nxt is not None else returns.loc[cur:].iloc[1:]
        pieces.append(held[weights.index].fillna(0) @ weights)
    daily = pd.concat(pieces)
    return daily[~daily.index.duplicated()], fallbacks


def summarize(daily: pd.Series, rf: pd.Series) -> dict:
    growth = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    excess = daily - rf.reindex(daily.index).ffill().fillna(0)
    return {
        "Total return": growth.iloc[-1] - 1,
        "Annual return": growth.iloc[-1] ** (1 / years) - 1,
        "Annual volatility": daily.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": excess.mean() / daily.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (growth / growth.cummax() - 1).min(),
    }


def main() -> None:
    args = parse_args()
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    esg = pd.Series(args.esg_scores, index=args.tickers)
    start = min(args.frontier_start, args.backtest_start)
    end = max(args.frontier_end, args.backtest_end)

    prices = fetch_price_history(args.tickers, start, end).dropna()
    rf = fetch_risk_free_rate(start, end)
    bench = fetch_price_history([args.benchmark], start, end)[args.benchmark].pct_change()

    plot_frontier(prices, rf, esg, args, images / "esg_frontier.png")

    rebal = quarter_starts(prices.loc[args.backtest_start:args.backtest_end].index, args.backtest_start)
    prices = prices.loc[:args.backtest_end]
    strategies, fallbacks = {}, {}
    for target in args.targets:
        name = f"ESG target {target:.2f}"
        strategies[name], fallbacks[name] = backtest(
            prices, rf,
            lambda m, c, t=target: optimize_esg_frontier(m, c, esg.values, t, args.min_allocation).weights,
            rebal)
    strategies["Max-Sharpe, no ESG target"], fallbacks["Max-Sharpe, no ESG target"] = backtest(
        prices, rf, lambda m, c: max_sharpe_portfolio(m, c, args.min_allocation).weights, rebal)
    strategies["Equal weight"], _ = backtest(
        prices, rf, lambda m, c: pd.Series(1 / len(args.tickers), index=args.tickers), rebal)
    first_day = next(iter(strategies.values())).index
    strategies[f"{args.benchmark} (buy and hold)"] = bench.reindex(first_day).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, daily in strategies.items():
        ax.plot((1 + daily).cumprod(), lw=1.6, label=name)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"Quarterly-rebalanced backtest, {first_day[0]:%b %Y} to {first_day[-1]:%b %Y}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(images / "backtest.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame({name: summarize(daily, rf) for name, daily in strategies.items()}).T
    shown = table.copy()
    for col in ["Total return", "Annual return", "Annual volatility", "Max drawdown"]:
        shown[col] = shown[col].map("{:.1%}".format)
    shown["Sharpe"] = shown["Sharpe"].map("{:.2f}".format)
    universe = ", ".join(f"{t} ({s:.2f})" for t, s in esg.items())
    report = (
        "# Backtest results\n\n"
        f"Generated by `research/esg_frontier_backtest.py` on {dt.date.today():%Y-%m-%d}.\n\n"
        f"- Universe (normalized ESG score): {universe}\n"
        f"- Period: {first_day[0]:%Y-%m-%d} to {first_day[-1]:%Y-%m-%d}, rebalanced quarterly "
        f"({len(rebal) - 1} rebalances), each using only the previous quarter's data\n"
        f"- Long only, at least {args.min_allocation:.0%} per asset; no transaction costs\n"
        f"- Optimizer fallbacks (kept previous weights): {sum(fallbacks.values())}\n\n"
        + shown.to_markdown() + "\n\n"
        "Five stocks over one sample period is a small test. Read these numbers as a demonstration "
        "of the method, not as evidence that an ESG target improves returns. Equal weight often beats "
        "mean-variance portfolios out of sample when expected returns come from a short window "
        "(DeMiguel, Garlappi and Uppal, 2009), because noisy return estimates dominate the optimization.\n"
    )
    (args.out / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
