import pandas as pd
import numpy as np


def run_backtest(
    returns,
    expected_returns,
    covs,
    weight_func,
    transaction_cost=0.001
):
    portfolio_returns = []
    result_dates = []

    dates = sorted(covs.keys())
    dates = [d for d in dates if d in expected_returns.index]

    assert all(dates[i] < dates[i+1] for i in range(len(dates)-1))
    prev_weights = None

    for date, next_date in zip(dates, dates[1:]):
        if next_date not in returns.index:
            continue

        mu = expected_returns.loc[date].copy()
        if mu.isna().any():
            continue

        cov = covs[date]

        weights = weight_func(mu, cov)
        if weights.isna().any():
            continue

        realized = returns.loc[next_date]

        gross = (weights * realized).sum()
        cost = 0 if prev_weights is None else transaction_cost * (weights - prev_weights).abs().sum()

        portfolio_returns.append(gross - cost)
        result_dates.append(next_date)
        prev_weights = weights

    return pd.Series(portfolio_returns, index=result_dates)
