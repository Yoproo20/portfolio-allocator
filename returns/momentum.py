import pandas as pd


def rolling_mean_returns(
    returns: pd.DataFrame,
    lookback: int
) -> pd.Series:
    """
    Expected returns via rolling mean.

    Parameters
    ----------
    returns : pd.DataFrame -> Weekly log returns up to time t (index = date)
    lookback : int -> Number of weeks to look back

    Returns
    -------
    pd.Series -> Expected return per asset for time t+1
    """
    if len(returns) < lookback:
        raise ValueError("Not enough data for rolling mean lookback")

    window = returns.iloc[-lookback:]
    return window.mean()


def simple_momentum_returns(
    returns: pd.DataFrame,
    lookback: int
) -> pd.Series:
    """
    Expected returns via cumulative momentum.

    Parameters
    ----------
    returns : pd.DataFrame -> Weekly log returns up to time t
    lookback : int -> Number of weeks to look back

    Returns
    -------
    pd.Series -> Expected return per asset for time t+1
    """
    if len(returns) < lookback:
        raise ValueError("Not enough data for momentum lookback")

    window = returns.iloc[-lookback:]
    return window.sum() / lookback
