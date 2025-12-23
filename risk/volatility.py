import pandas as pd
import numpy as np


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 52
) -> pd.DataFrame:
    """
    Rolling annualized volatility.
    """
    weekly_vol = returns.rolling(window).std()
    annualized_vol = weekly_vol * np.sqrt(52)

    return annualized_vol
