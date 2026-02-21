from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from resurgence_py.engine import simulate_regime_var_cvar_python
from resurgence_py.models import (
    default_state_drift_adjustments,
    default_state_vol_multipliers,
    default_transition_matrix,
)
from validation.tests import (
    ChristoffersenIndependenceResult,
    KupiecPOFResult,
    christoffersen_independence_test,
    kupiec_proportion_of_failures,
)

try:
    import resurgence_core
except Exception:  # noqa: BLE001
    resurgence_core = None


def default_sample_tickers() -> list[str]:
    return ["SPY", "QQQ", "TLT", "GLD"]


def default_sample_weights() -> list[float]:
    return [0.40, 0.25, 0.20, 0.15]


class BacktestConfig(BaseModel):
    """Configuration for rolling historical backtesting."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=default_sample_tickers, min_length=1)
    weights: list[float] | None = None
    start_date: date = date(2012, 1, 1)
    end_date: date = date.today()
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    rolling_window: int = Field(default=252, ge=30)
    simulations: int = Field(default=10_000, ge=1_000, le=1_000_000)
    horizon_days: int = Field(default=1, ge=1, le=60)
    seed: int | None = Field(default=42, ge=0)
    method: str = Field(default="monte_carlo")
    use_rust_if_available: bool = True

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for ticker in tickers:
            clean = ticker.strip().upper()
            if not clean:
                continue
            if clean not in seen:
                seen.add(clean)
                normalized.append(clean)

        if not normalized:
            msg = "At least one ticker is required"
            raise ValueError(msg)

        return normalized

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: list[float] | None, info: ValidationInfo) -> list[float] | None:
        if weights is None:
            return None

        tickers = info.data.get("tickers", [])
        if len(weights) != len(tickers):
            msg = "weights length must match tickers length"
            raise ValueError(msg)
        if any(weight <= 0.0 for weight in weights):
            msg = "weights must be positive"
            raise ValueError(msg)

        total = sum(weights)
        if total <= 0.0:
            msg = "weights must sum to a positive value"
            raise ValueError(msg)

        return [weight / total for weight in weights]

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, end_date: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start is not None and start > end_date:
            msg = "start_date must be on or before end_date"
            raise ValueError(msg)
        return end_date

    @field_validator("method")
    @classmethod
    def validate_method(cls, method: str) -> str:
        normalized = method.strip().lower()
        if normalized not in {"historical", "monte_carlo"}:
            msg = "method must be either 'historical' or 'monte_carlo'"
            raise ValueError(msg)
        return normalized


class BacktestObservation(BaseModel):
    """One-step-ahead VaR forecast versus realized portfolio return."""

    model_config = ConfigDict(extra="forbid")

    date: date
    realized_return: float
    realized_loss: float = Field(ge=0.0)
    predicted_var: float = Field(ge=0.0)
    predicted_cvar: float = Field(ge=0.0)
    breach: bool


class BacktestSummary(BaseModel):
    """Aggregated validation summary for a rolling VaR/CVaR backtest."""

    model_config = ConfigDict(extra="forbid")

    generated_at_utc: datetime
    config: BacktestConfig
    observation_count: int = Field(ge=1)
    breaches: int = Field(ge=0)
    breach_rate: float = Field(ge=0.0, le=1.0)
    expected_breach_rate: float = Field(ge=0.0, le=1.0)
    mean_predicted_var: float = Field(ge=0.0)
    mean_predicted_cvar: float = Field(ge=0.0)
    mean_realized_loss: float = Field(ge=0.0)
    kupiec_pof: KupiecPOFResult
    christoffersen: ChristoffersenIndependenceResult


class BacktestRunResult(BaseModel):
    """Backtest payload containing summary, detailed rows, and output locations."""

    model_config = ConfigDict(extra="forbid")

    summary: BacktestSummary
    observations: list[BacktestObservation]
    csv_path: str = Field(min_length=1)
    json_path: str = Field(min_length=1)


def _normalize_weights(tickers: list[str], weights: list[float] | None) -> list[float]:
    if weights is not None:
        return weights
    size = len(tickers)
    return [1.0 / size for _ in range(size)]


def load_sample_portfolio_returns(config: BacktestConfig, timeout_s: float = 20.0) -> pd.Series:
    """Fetch a sample multi-asset portfolio and return daily equal/weighted returns."""
    end_exclusive = config.end_date + timedelta(days=1)

    frame = yf.download(
        tickers=config.tickers,
        start=config.start_date.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        timeout=timeout_s,
        group_by="column",
        threads=True,
    )

    if frame.empty:
        msg = "No market data returned for backtest"
        raise RuntimeError(msg)

    prices = _extract_price_frame(frame, config.tickers)
    returns = prices.pct_change().dropna(how="any")

    if returns.empty:
        msg = "Backtest return series is empty after normalization"
        raise RuntimeError(msg)

    weights = np.array(_normalize_weights(config.tickers, config.weights), dtype=np.float64)
    portfolio = returns.mul(weights, axis=1).sum(axis=1)
    portfolio.name = "portfolio_return"
    return portfolio.astype(np.float64)


def _extract_price_frame(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if "Adj Close" in frame.columns.get_level_values(0):
            prices = frame["Adj Close"]
        elif "Close" in frame.columns.get_level_values(0):
            prices = frame["Close"]
        else:
            msg = "Expected Close or Adj Close columns in yfinance payload"
            raise RuntimeError(msg)
    else:
        if "Adj Close" in frame.columns:
            prices = frame[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in frame.columns:
            prices = frame[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            msg = "Expected Close or Adj Close columns in yfinance payload"
            raise RuntimeError(msg)

    prices = prices.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        msg = f"Missing price columns for tickers: {missing}"
        raise RuntimeError(msg)

    return prices[tickers]


def _historical_var_cvar(losses: np.ndarray, confidence_level: float) -> tuple[float, float]:
    if losses.size == 0:
        msg = "losses must contain at least one observation"
        raise ValueError(msg)

    var = float(np.quantile(losses, confidence_level))
    tail_losses = losses[losses >= var]
    cvar = float(np.mean(tail_losses)) if tail_losses.size > 0 else var
    return var, cvar


def _monte_carlo_var_cvar(
    window_returns: np.ndarray,
    confidence_level: float,
    simulations: int,
    horizon_days: int,
    seed: int | None,
    use_rust_if_available: bool,
) -> tuple[float, float]:
    transition = default_transition_matrix()
    vol_multipliers = default_state_vol_multipliers()
    drift_adjustments = default_state_drift_adjustments()

    if use_rust_if_available and resurgence_core is not None and hasattr(resurgence_core, "simulate_regime_var_cvar"):
        result, _ = resurgence_core.simulate_regime_var_cvar(
            window_returns.tolist(),
            confidence_level,
            simulations,
            horizon_days,
            transition,
            vol_multipliers,
            drift_adjustments,
            2,
            seed,
            None,
        )
        return float(result.var), float(result.cvar)

    risk, _ = simulate_regime_var_cvar_python(
        returns=window_returns.tolist(),
        confidence_level=confidence_level,
        simulations=simulations,
        horizon_days=horizon_days,
        transition_matrix=transition,
        state_vol_multipliers=vol_multipliers,
        state_drift_adjustments=drift_adjustments,
        initial_state=2,
        seed=seed,
    )
    return float(risk.var), float(risk.cvar)


def run_backtest_on_returns(
    portfolio_returns: pd.Series,
    config: BacktestConfig,
) -> tuple[list[BacktestObservation], BacktestSummary]:
    """Run rolling VaR/CVaR forecasts and breach tracking against a return series."""
    cleaned = portfolio_returns.dropna().astype(np.float64)
    if len(cleaned) <= config.rolling_window:
        msg = "Not enough observations for the configured rolling window"
        raise ValueError(msg)

    values = cleaned.to_numpy(dtype=np.float64)
    dates = cleaned.index

    observations: list[BacktestObservation] = []
    breaches: list[bool] = []

    for idx in range(config.rolling_window, len(values)):
        window = values[idx - config.rolling_window : idx]
        realized_return = float(values[idx])
        realized_loss = max(-realized_return, 0.0)

        if config.method == "historical":
            var, cvar = _historical_var_cvar(np.maximum(-window, 0.0), config.confidence_level)
        else:
            step_seed = None if config.seed is None else config.seed + idx
            var, cvar = _monte_carlo_var_cvar(
                window_returns=window,
                confidence_level=config.confidence_level,
                simulations=config.simulations,
                horizon_days=config.horizon_days,
                seed=step_seed,
                use_rust_if_available=config.use_rust_if_available,
            )

        breach = realized_loss > var
        breaches.append(breach)

        timestamp = pd.Timestamp(dates[idx])
        observations.append(
            BacktestObservation(
                date=timestamp.date(),
                realized_return=realized_return,
                realized_loss=realized_loss,
                predicted_var=var,
                predicted_cvar=cvar,
                breach=breach,
            )
        )

    kupiec = kupiec_proportion_of_failures(breaches, config.confidence_level)
    christoffersen = christoffersen_independence_test(breaches)

    predicted_vars = np.array([row.predicted_var for row in observations], dtype=np.float64)
    predicted_cvars = np.array([row.predicted_cvar for row in observations], dtype=np.float64)
    realized_losses = np.array([row.realized_loss for row in observations], dtype=np.float64)

    summary = BacktestSummary(
        generated_at_utc=datetime.now(UTC),
        config=config,
        observation_count=len(observations),
        breaches=kupiec.breaches,
        breach_rate=kupiec.observed_breach_rate,
        expected_breach_rate=kupiec.expected_breach_rate,
        mean_predicted_var=float(np.mean(predicted_vars)),
        mean_predicted_cvar=float(np.mean(predicted_cvars)),
        mean_realized_loss=float(np.mean(realized_losses)),
        kupiec_pof=kupiec,
        christoffersen=christoffersen,
    )
    return observations, summary


def write_backtest_results(
    observations: list[BacktestObservation],
    summary: BacktestSummary,
    output_dir: str | Path,
) -> tuple[str, str]:
    """Write detailed rows to CSV and summary to JSON for reproducible review."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "rolling_backtest.csv"
    json_path = out_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "realized_return",
                "realized_loss",
                "predicted_var",
                "predicted_cvar",
                "breach",
            ],
        )
        writer.writeheader()
        for row in observations:
            writer.writerow(row.model_dump(mode="json"))

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.model_dump(mode="json"), handle, indent=2)

    return str(csv_path), str(json_path)


def run_historical_backtest(
    config: BacktestConfig,
    output_dir: str | Path = "validation/results",
) -> BacktestRunResult:
    """Run a complete validation pass for a sample multi-asset portfolio."""
    portfolio_returns = load_sample_portfolio_returns(config=config)
    observations, summary = run_backtest_on_returns(portfolio_returns=portfolio_returns, config=config)
    csv_path, json_path = write_backtest_results(observations, summary, output_dir=output_dir)

    return BacktestRunResult(
        summary=summary,
        observations=observations,
        csv_path=csv_path,
        json_path=json_path,
    )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run rolling VaR/CVaR validation backtest")
    parser.add_argument("--tickers", type=str, default=",".join(default_sample_tickers()))
    parser.add_argument("--start-date", type=_parse_date, default=date(2012, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=date.today())
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="monte_carlo", choices=["historical", "monte_carlo"])
    parser.add_argument("--output-dir", type=str, default="validation/results")
    parser.add_argument("--disable-rust", action="store_true")
    args = parser.parse_args()

    tickers = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
    config = BacktestConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        confidence_level=args.confidence_level,
        rolling_window=args.rolling_window,
        simulations=args.simulations,
        horizon_days=args.horizon_days,
        seed=args.seed,
        method=args.method,
        use_rust_if_available=not args.disable_rust,
    )

    result = run_historical_backtest(config=config, output_dir=args.output_dir)
    print(result.summary.model_dump_json(indent=2))
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")


if __name__ == "__main__":
    cli()
