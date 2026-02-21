from __future__ import annotations

import asyncio
import logging
from math import exp

import numpy as np

from resurgence_py.models import EngineInput, FlowInput, PortfolioSeries, RiskMetrics

logger = logging.getLogger(__name__)

try:
    import resurgence_core
except Exception:  # noqa: BLE001
    resurgence_core = None


def simulate_var_cvar_python(
    returns: list[float],
    confidence_level: float,
    simulations: int,
    horizon_days: int,
    seed: int | None,
) -> RiskMetrics:
    """Reference Python implementation for benchmarking and fallback mode."""
    rng = np.random.default_rng(seed if seed is not None else 42)

    values = np.array(returns, dtype=float)
    if len(values) < 2:
        msg = "Need at least two return observations"
        raise ValueError(msg)

    drift = float(np.mean(values))
    volatility = float(np.std(values, ddof=1))
    if volatility <= np.finfo(float).eps:
        msg = "Degenerate volatility encountered"
        raise ValueError(msg)

    shocks = rng.standard_normal(size=(simulations, horizon_days))
    log_returns = drift - 0.5 * volatility * volatility + volatility * shocks
    terminal_values = np.exp(log_returns.sum(axis=1))
    pnl = terminal_values - 1.0
    losses = np.maximum(-pnl, 0.0)

    losses_sorted = np.sort(losses)
    var_index = max(min(int(np.ceil(confidence_level * simulations)) - 1, simulations - 1), 0)
    var = float(losses_sorted[var_index])
    cvar = float(np.mean(losses_sorted[var_index:]))

    return RiskMetrics(
        var=var,
        cvar=cvar,
        mean_loss=float(np.mean(losses_sorted)),
        loss_stddev=float(np.std(losses_sorted, ddof=1)),
        confidence_level=confidence_level,
        simulations=simulations,
        horizon_days=horizon_days,
        estimated_drift=drift,
        estimated_volatility=volatility,
    )


class Engine:
    """Node 2: run Monte Carlo VaR/CVaR from the Rust extension."""

    async def run(self, payload: FlowInput, portfolio: PortfolioSeries) -> RiskMetrics:
        engine_input = EngineInput(
            returns=portfolio.returns,
            confidence_level=payload.confidence_level,
            simulations=payload.simulations,
            horizon_days=payload.horizon_days,
            seed=payload.seed,
        )

        if resurgence_core is None:
            logger.warning(
                "Rust extension unavailable; using Python fallback for risk simulation"
            )
            return await asyncio.to_thread(
                simulate_var_cvar_python,
                engine_input.returns,
                engine_input.confidence_level,
                engine_input.simulations,
                engine_input.horizon_days,
                engine_input.seed,
            )

        try:
            result = await asyncio.to_thread(
                resurgence_core.simulate_var_cvar,
                engine_input.returns,
                engine_input.confidence_level,
                engine_input.simulations,
                engine_input.horizon_days,
                engine_input.seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rust simulation failed")
            msg = f"Rust Monte Carlo failure: {exc}"
            raise RuntimeError(msg) from exc

        return RiskMetrics(
            var=float(result.var),
            cvar=float(result.cvar),
            mean_loss=float(result.mean_loss),
            loss_stddev=float(result.loss_stddev),
            confidence_level=float(result.confidence_level),
            simulations=int(result.simulations),
            horizon_days=int(result.horizon_days),
            estimated_drift=float(result.estimated_drift),
            estimated_volatility=float(result.estimated_volatility),
        )
