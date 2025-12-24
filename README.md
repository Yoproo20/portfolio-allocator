# Portfolio Allocator

A research-focused toolkit for building and backtesting portfolio allocation strategies on a small ETF universe. The project covers data ingestion, return forecasting, risk modeling, portfolio construction, and backtesting with transaction costs.

## Capabilities
- Downloads and cleans adjusted close data from Yahoo Finance via `yfinance`.
- Computes daily and weekly log returns.
- Estimates expected returns using rolling mean or momentum.
- Builds rolling Ledoit-Wolf covariance matrices.
- Optional ML return forecasting with Gradient Boosting regressors.
- Allocates portfolios using Equal Weight, Risk Parity, or Mean-Variance optimization.
- Runs weekly-rebalanced backtests with transaction costs and performance metrics.
- Plots cumulative performance by strategy.

## Asset Universe and Frequency
- Assets: SPY, QQQ, IWM, TLT, GLD, VNQ
- Frequency: Weekly, rebalanced on Fridays

## Project Structure
```
allocation/
  equal_weight.py
  risk_parity.py
  mean_variance.py
  allocation_sanity_check.py
backtest/
  engine.py
  metrics.py
  plotting.py
  backtest.py
optimization/
  predict_returns.py
returns/
  expected_returns.py
  momentum.py
  generate_expected_returns.py
  returns_sanity_check.py
risk/
  covariance.py
  volatility.py
  generate_covariances.py
  risk_sanity_check.py
data/
  prices/
    raw_prices.csv
    prices_clean.csv
    returns_daily.csv
    returns_weekly.csv
    expected_returns_weekly.csv
    predicted_returns_weekly.csv
  covariances.pkl
get_data.py
validate_data.py
requirements.txt
Makefile
README.md
```

## Quickstart
1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Download and prepare data:

```bash
python get_data.py
python validate_data.py
```

3) Generate expected returns and covariances:

```bash
python returns\\generate_expected_returns.py
python risk\\generate_covariances.py
```

4) (Optional) Generate ML-based return predictions:

```bash
python optimization\\predict_returns.py
```

5) Run the backtest and plot results:

```bash
python backtest\\backtest.py
```

## Makefile Shortcuts
If you have `make` available:

```bash
make venv
make data
make expected
make covariances
make ml
make backtest
make sanity
```

## Workflow Details
- Data pipeline writes cleaned prices and returns to `data/prices/`.
- Expected returns are shifted by one period in `backtest/backtest.py` to prevent lookahead bias.
- ML predictions are ranked cross-sectionally and z-scored before use.
- Risk model uses a 52-week rolling Ledoit-Wolf covariance estimator.
- Allocation strategies are plug-in functions that map (mu, cov) -> weights.

## Strategy Configuration
Edit `backtest/backtest.py` to control expected return sources and blending:
- `MU_MODE`: "stat" (rolling mean), "ml_rank" (ML ranks), or "blend".
- `ALPHA_STAT`: weight on the statistical signal in blend mode.

## Outputs
- CSVs of prices, returns, and expected returns in `data/prices/`.
- Rolling covariance matrices in `data/covariances.pkl`.
- Console metrics: Annual Return, Annual Volatility, Sharpe, Max Drawdown.
- Cumulative returns plot for each strategy.

## Notes
- Data is pulled from Yahoo Finance; network access is required.
- Mean-Variance optimization uses long-only weights with a max position cap.
- The ML model is trained on the full available history (research setting).

## Sanity Checks
- `allocation/allocation_sanity_check.py` prints sample weights and sum checks.
- `risk/risk_sanity_check.py` inspects covariance properties and volatility.
- `returns/returns_sanity_check.py` validates expected return calculations.
