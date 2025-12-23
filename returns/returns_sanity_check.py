import pandas as pd
from returns.expected_returns import (
    load_weekly_returns,
    compute_expected_returns
)

returns = load_weekly_returns()

# pick a date safely after lookback
evaluation_date = returns.index[30]

er_mean = compute_expected_returns(
    returns,
    evaluation_date,
    method="mean",
    lookback=12
)

er_momentum = compute_expected_returns(
    returns,
    evaluation_date,
    method="momentum",
    lookback=12
)

print(f"\nExpected returns on {evaluation_date.date()} (rolling mean):")
print(er_mean)

print(f"\nExpected returns on {evaluation_date.date()} (momentum):")
print(er_momentum)
