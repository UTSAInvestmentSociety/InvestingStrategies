# Investing Strategies (Bloomberg blpapi Demos)

## Overview
This repository collects a suite of end-to-end Python demonstrations for quantitative investment strategies that source pricing and fundamental data exclusively through Bloomberg's Desktop API (`blpapi`). Each script focuses on a different portfolio construction technique, wraps its own configuration via a `dataclass`, and exports analytics such as weights, returns, and optional plots to a dedicated output folder.【F:strategy_01_momentum.py†L3-L64】【F:strategy_mean_variance_opt.py†L3-L93】【F:strategy_value_screens.py†L3-L82】

> **Bloomberg access required** – All strategies assume that the Bloomberg Terminal is running locally with Desktop API permissions and that the `blpapi` Python package is installed alongside common scientific libraries (NumPy, pandas, matplotlib, etc.).【F:strategy_01_momentum.py†L7-L13】【F:strategy_mean_variance_opt.py†L20-L27】【F:strategy_value_screens.py†L51-L81】

## Repository Structure

| File | Description |
| --- | --- |
| `all_strategies.py` | Command-line runner that sequentially executes the core strategy scripts, streams their output to the console, and persists per-script logs under `./logs/`, with options for timeouts and error handling.【F:all_strategies.py†L3-L131】 |
| `strategy_01_momentum.py` | Cross-sectional 12–1 momentum strategy on a diversified ETF universe, including Bloomberg data ingestion, monthly rebalance logic, turnover-aware performance, and optional equity curve plotting.【F:strategy_01_momentum.py†L3-L206】【F:strategy_01_momentum.py†L254-L336】 |
| `strategy_mean_variance_opt.py` | Rolling mean-variance optimization demo that contrasts global minimum variance, max-Sharpe, and naïve equal-weight portfolios using Bloomberg total-return history and convex optimization via CVXPy.【F:strategy_mean_variance_opt.py†L3-L109】【F:strategy_mean_variance_opt.py†L156-L200】 |
| `strategy_value_screens.py` | Sector-neutral value screens that assemble composite valuation scores (E/P, B/M, EBITDA/EV), form long-short decile portfolios with turnover costs, and export monthly performance artefacts.【F:strategy_value_screens.py†L3-L199】 |
| `strategy_events.py` | Dividend ex-date event study measuring abnormal and cumulative abnormal returns around corporate actions, complete with CAPM-based expected return estimation and optional visualization.【F:strategy_events.py†L3-L200】 |
| `strategy_factors.py` | Factor investing toolkit that builds Bloomberg-based SMB, HML, UMD, and Quality factors, runs rolling regressions for factor betas, and backtests beta-sorted long/short portfolios.【F:strategy_factors.py†L3-L200】 |
| `strategy_index_beta.py` | Smart beta index-tracking example for the RGUSTSC semiconductor index, featuring tracking-error minimization, equal-weight, and inverse-volatility portfolios with monthly rebalancing.【F:strategy_index_beta.py†L3-L200】 |
| `strategy_pairs.py` | Pairs trading (Engle–Granger) workflow that filters sector-consistent cointegrated pairs, runs a rolling z-score trading rule with transaction costs, and aggregates pair-level returns.【F:strategy_pairs.py†L3-L200】 |

## Getting Started
1. **Install prerequisites**
   ```bash
   pip install blpapi pandas numpy matplotlib cvxpy statsmodels
   ```
   Adjust the list according to the strategy you intend to run (e.g., `cvxpy` for mean-variance optimization, `statsmodels` for pairs trading).【F:strategy_mean_variance_opt.py†L20-L27】【F:strategy_pairs.py†L20-L27】

2. **Configure Bloomberg Desktop API**
   Ensure the Bloomberg Terminal is active on the same machine and that your user has Desktop API entitlements. Scripts connect to `localhost:8194` by default but expose `host` and `port` fields in their configuration dataclasses for custom deployments.【F:strategy_01_momentum.py†L31-L34】【F:strategy_pairs.py†L53-L58】

3. **Customize strategy parameters**
   Open the relevant script and edit its `Config` dataclass to adjust universes, date ranges, rebalancing rules, and output folders before execution.【F:strategy_value_screens.py†L38-L82】【F:strategy_index_beta.py†L46-L78】

## Running Individual Scripts
Each script can be executed directly:
```bash
python strategy_01_momentum.py
```
Most strategies create an output directory (e.g., `./out_momentum_blpapi`) populated with intermediate data, weights, and performance CSV files; some also save plots if `save_plots` is enabled.【F:strategy_01_momentum.py†L60-L75】【F:strategy_01_momentum.py†L285-L332】

### Batch Execution
To run the core trio (`momentum`, `mean_variance_opt`, `value_screens`) in sequence, use the orchestrator:
```bash
python all_strategies.py --base-dir . --continue-on-error
```
Optional flags include `--timeout` (per-script seconds) and `--python` (alternative interpreter). The runner halts on the first failure unless `--continue-on-error` is specified and prints a summary table at completion.【F:all_strategies.py†L14-L131】

## Outputs & Logging
- **CSV artifacts** – Each module writes cleaned price histories, signals, weights, and performance series to its `out_*` directory for further analysis or visualization.【F:strategy_value_screens.py†L14-L80】【F:strategy_pairs.py†L89-L118】
- **Plots (optional)** – Many scripts toggle matplotlib chart generation via the `save_plots` flag, saving PNG equity curves or factor diagnostics when enabled.【F:strategy_01_momentum.py†L312-L330】【F:strategy_index_beta.py†L75-L78】
- **Run logs** – `all_strategies.py` captures timestamped stdout/stderr streams for each child process under `./logs/`, aiding reproducibility and troubleshooting.【F:all_strategies.py†L44-L95】

## Strategy Highlights
- **Momentum** – Cross-sectional 12-month momentum with one-month skip, dollar-neutral long/short weighting, and turnover cost adjustments.【F:strategy_01_momentum.py†L49-L122】【F:strategy_01_momentum.py†L217-L308】
- **Mean-Variance Optimization** – Rolling 60-month estimation window, shrinkage controls for mean and covariance, and CVXPy-based solver for long-only GMV and max-Sharpe portfolios.【F:strategy_mean_variance_opt.py†L76-L200】
- **Value Screens** – Sector-relative z-score composites across valuation ratios with decile portfolios and 1-month fundamental lag to mitigate look-ahead bias.【F:strategy_value_screens.py†L7-L76】
- **Event Study** – Dividend ex-date abnormal return analysis using CAPM-based expected returns and configurable event/estimation windows.【F:strategy_events.py†L6-L85】
- **Factor Investing** – Construction of SMB, HML, UMD, and Quality factors plus beta-sorted long/short overlays on Bloomberg universes.【F:strategy_factors.py†L7-L113】
- **Index Tracking / Smart Beta** – RGUSTSC semiconductor constituents with tracking-error minimization, equal-weight, and inverse-volatility portfolios.【F:strategy_index_beta.py†L6-L90】
- **Pairs Trading** – Sector-aware cointegration screening with Engle–Granger tests, rolling z-score entries/exits, and turnover-aware portfolio aggregation.【F:strategy_pairs.py†L6-L118】

## Adapting the Templates
Because the scripts isolate Bloomberg connectivity and data wrangling utilities (e.g., historical price fetchers, reference data loaders), you can readily port the helper classes and configuration blocks into bespoke research notebooks or production pipelines. Adjust universes, add new factors, or swap event definitions while preserving the provided logging and output conventions.【F:strategy_value_screens.py†L103-L199】【F:strategy_factors.py†L120-L200】

## Disclaimer
These scripts are for educational and prototyping purposes only. They assume access to licensed Bloomberg data; confirm compliance with your organization's data usage policies before running or distributing outputs.【F:strategy_factors.py†L20-L26】【F:strategy_events.py†L18-L27】
