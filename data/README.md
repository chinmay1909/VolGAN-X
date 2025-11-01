# Data Directory

This folder contains option-market datasets used for training and evaluation.

- **spx_options_sample.csv** — toy dataset derived from S&P 500 option quotes.
- Replace or extend with proprietary data from WRDS / OptionMetrics when running private experiments.
- Columns:
  - `maturity` – time to expiration (in years)
  - `strike` – option strike price
  - `spot` – underlying asset spot price
  - `implied_vol` – Black-Scholes implied volatility
  - `call_price`, `put_price` – corresponding option mid prices