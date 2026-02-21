# MODEL

## Scope
Resurgence estimates portfolio tail risk (VaR/CVaR) for short horizons using a regime-switching Monte Carlo model calibrated on historical daily returns.

## Return Distribution Assumptions
- Base daily returns are summarized by sample mean (`mu`) and sample volatility (`sigma`) from observed portfolio returns.
- Path evolution uses log-return dynamics with Gaussian shocks per step:
  - `r_t = mu_state - 0.5 * sigma_state^2 + sigma_state * z_t`
  - `z_t ~ N(0, 1)`
- Losses are long-only (`loss = max(-(V_T - 1), 0)`), so loss support is non-negative.

## Correlation Modeling Approach
- The current engine runs on **portfolio-level returns** after aggregating constituent assets in the data layer.
- Cross-asset correlation is therefore represented **implicitly** through the historical portfolio series, not via an explicit time-varying covariance matrix.
- Practical implication: this is robust and simple, but does not decompose marginal contribution or model dynamic correlation breakdown explicitly.

## Monte Carlo Methodology
- Regime state follows a 3-state Markov chain with configurable transition matrix.
- Each state modifies drift and volatility with state-specific adjustments/multipliers.
- For each simulation path:
  1. Sample next regime state from transition probabilities.
  2. Draw Gaussian innovation.
  3. Update portfolio value via log-return compounding.
- VaR is the empirical quantile of simulated losses; CVaR is the conditional mean beyond VaR.

## Random Seed Handling
- Python and Rust engines accept an explicit seed.
- Python path generation uses NumPy `default_rng(seed)`.
- Rust path generation deterministically derives per-path seeds from a base seed for reproducible parallel runs.
- If seed is omitted, the default seed (`42`) is used.

## Known Limitations
- Gaussian innovations understate extreme tail behavior vs heavy-tailed/stochastic-volatility processes.
- Markov regimes are fixed-parameter and not re-estimated online.
- Transaction costs, slippage, liquidity stress, and jump processes are not explicitly modeled.
- Portfolio-level aggregation hides instrument-level factor and concentration effects.
- Backtest quality depends on data completeness and market regime representativeness.

## When NOT to Use This Model
- Intraday risk control requiring microstructure-aware dynamics.
- Portfolios with strong option convexity/path dependence without dedicated Greeks/revaluation.
- Illiquid or discontinuous markets where jump/default dynamics dominate.
- Regulatory capital workflows that require approved model governance beyond this implementation.
