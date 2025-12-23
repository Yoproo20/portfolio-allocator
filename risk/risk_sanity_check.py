import pandas as pd
import numpy as np

from risk.covariance import rolling_ledoit_wolf_covariance
from risk.volatility import rolling_volatility


def main():
    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    covs = rolling_ledoit_wolf_covariance(returns)
    vols = rolling_volatility(returns)

    # Pick a recent date
    latest_date = list(covs.keys())[-1]
    cov = covs[latest_date]

    print(f"\nCovariance matrix on {latest_date.date()}:\n")
    print(cov.round(6))

    print("\nEigenvalues (should be >= 0):")
    print(np.linalg.eigvals(cov))

    print("\nLatest annualized volatility:")
    print(vols.loc[latest_date].round(3))


if __name__ == "__main__":
    main()
