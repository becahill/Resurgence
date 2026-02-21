from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from resurgence_py.models import EngineInput, FlowInput, PortfolioSeries, RiskMetrics

logger = logging.getLogger(__name__)

try:
    import resurgence_core
except Exception:  # noqa: BLE001
    resurgence_core = None


@dataclass(slots=True)
class EngineRunArtifacts:
    """Runtime artifacts from Engine execution."""

    risk: RiskMetrics
    losses: np.ndarray
    latency_ms: float


def simulate_regime_var_cvar_python(
    returns: list[float],
    confidence_level: float,
    simulations: int,
    horizon_days: int,
    transition_matrix: list[list[float]],
    state_vol_multipliers: list[float],
    state_drift_adjustments: list[float],
    initial_state: int,
    seed: int | None,
) -> tuple[RiskMetrics, np.ndarray]:
    """Reference Python implementation of regime-switching VaR/CVaR."""
    rng = np.random.default_rng(seed if seed is not None else 42)

    values = np.array(returns, dtype=np.float64)
    if len(values) < 2:
        msg = "Need at least two return observations"
        raise ValueError(msg)

    drift = float(np.mean(values))
    volatility = float(np.std(values, ddof=1))
    if volatility <= np.finfo(float).eps:
        msg = "Degenerate volatility encountered"
        raise ValueError(msg)

    transition = np.array(transition_matrix, dtype=np.float64)
    vol_mult = np.array(state_vol_multipliers, dtype=np.float64)
    drift_adj = np.array(state_drift_adjustments, dtype=np.float64)

    state = np.full(shape=simulations, fill_value=initial_state, dtype=np.int64)
    log_growth = np.zeros(shape=simulations, dtype=np.float64)
    state_counts = np.zeros(shape=3, dtype=np.float64)

    for _ in range(horizon_days):
        draws = rng.random(simulations)
        next_state = np.empty_like(state)

        for source_state in range(3):
            mask = state == source_state
            if not np.any(mask):
                continue
            cdf = np.cumsum(transition[source_state])
            next_state[mask] = np.searchsorted(cdf, draws[mask], side="right")

        state = next_state
        for observed_state in range(3):
            state_counts[observed_state] += float(np.sum(state == observed_state))

        sigma = volatility * vol_mult[state]
        mu = drift + drift_adj[state]
        z = rng.standard_normal(simulations)
        log_growth += mu - 0.5 * sigma * sigma + sigma * z

    terminal_values = np.exp(log_growth)
    pnl = terminal_values - 1.0
    losses = np.maximum(-pnl, 0.0)
    losses_sorted = np.sort(losses)

    var_index = max(min(int(np.ceil(confidence_level * simulations)) - 1, simulations - 1), 0)
    var = float(losses_sorted[var_index])
    cvar = float(np.mean(losses_sorted[var_index:]))
    mean_loss = float(np.mean(losses_sorted))
    loss_stddev = float(np.std(losses_sorted, ddof=1))

    pnl_stddev = float(np.std(pnl, ddof=1))
    annualization = np.sqrt(252.0 / float(horizon_days))
    sharpe = float((float(np.mean(pnl)) / pnl_stddev) * annualization) if pnl_stddev > 0 else 0.0

    max_loss = float(losses_sorted[-1])
    max_loss_zscore = float((max_loss - mean_loss) / max(loss_stddev, 1e-9))

    state_occupancy = (state_counts / (simulations * horizon_days)).tolist()

    risk = RiskMetrics(
        var=var,
        cvar=cvar,
        mean_loss=mean_loss,
        loss_stddev=loss_stddev,
        confidence_level=confidence_level,
        simulations=simulations,
        horizon_days=horizon_days,
        estimated_drift=drift,
        estimated_volatility=volatility,
        sharpe_ratio=sharpe,
        max_loss_zscore=max_loss_zscore,
        state_occupancy=state_occupancy,
        regime_transition_matrix=transition_matrix,
        state_vol_multipliers=state_vol_multipliers,
    )
    return risk, losses_sorted


def simulate_var_cvar_python(
    returns: list[float],
    confidence_level: float,
    simulations: int,
    horizon_days: int,
    seed: int | None,
) -> RiskMetrics:
    """Backwards-compatible wrapper that uses the default HMM regime settings."""
    risk, _ = simulate_regime_var_cvar_python(
        returns=returns,
        confidence_level=confidence_level,
        simulations=simulations,
        horizon_days=horizon_days,
        transition_matrix=[
            [0.92, 0.04, 0.04],
            [0.10, 0.80, 0.10],
            [0.18, 0.14, 0.68],
        ],
        state_vol_multipliers=[0.70, 1.85, 1.00],
        state_drift_adjustments=[0.0003, -0.0008, 0.0],
        initial_state=2,
        seed=seed,
    )
    return risk


class Engine:
    """Node 2: run regime-switching Monte Carlo VaR/CVaR from the Rust extension."""

    async def run(
        self,
        payload: FlowInput,
        portfolio: PortfolioSeries,
        rayon_threads: int | None,
    ) -> EngineRunArtifacts:
        engine_input = EngineInput(
            returns=portfolio.returns,
            confidence_level=payload.confidence_level,
            simulations=payload.simulations,
            horizon_days=payload.horizon_days,
            transition_matrix=payload.hmm_transition_matrix,
            state_vol_multipliers=payload.state_vol_multipliers,
            state_drift_adjustments=payload.state_drift_adjustments,
            initial_state=payload.initial_state,
            seed=payload.seed,
            rayon_threads=rayon_threads,
        )

        start = perf_counter()

        if resurgence_core is None or not hasattr(resurgence_core, "simulate_regime_var_cvar"):
            logger.warning("Rust regime extension unavailable; using Python fallback simulation")
            risk, losses = await asyncio.to_thread(
                simulate_regime_var_cvar_python,
                engine_input.returns,
                engine_input.confidence_level,
                engine_input.simulations,
                engine_input.horizon_days,
                engine_input.transition_matrix,
                engine_input.state_vol_multipliers,
                engine_input.state_drift_adjustments,
                engine_input.initial_state,
                engine_input.seed,
            )
            latency_ms = (perf_counter() - start) * 1_000.0
            return EngineRunArtifacts(risk=risk, losses=losses, latency_ms=latency_ms)

        try:
            result, losses = await asyncio.to_thread(
                resurgence_core.simulate_regime_var_cvar,
                engine_input.returns,
                engine_input.confidence_level,
                engine_input.simulations,
                engine_input.horizon_days,
                engine_input.transition_matrix,
                engine_input.state_vol_multipliers,
                engine_input.state_drift_adjustments,
                engine_input.initial_state,
                engine_input.seed,
                engine_input.rayon_threads,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rust regime simulation failed")
            msg = f"Rust regime Monte Carlo failure: {exc}"
            raise RuntimeError(msg) from exc

        latency_ms = (perf_counter() - start) * 1_000.0

        risk = RiskMetrics(
            var=float(result.var),
            cvar=float(result.cvar),
            mean_loss=float(result.mean_loss),
            loss_stddev=float(result.loss_stddev),
            confidence_level=float(result.confidence_level),
            simulations=int(result.simulations),
            horizon_days=int(result.horizon_days),
            estimated_drift=float(result.estimated_drift),
            estimated_volatility=float(result.estimated_volatility),
            sharpe_ratio=float(result.sharpe_ratio),
            max_loss_zscore=float(result.max_loss_zscore),
            state_occupancy=[float(value) for value in result.state_occupancy],
            regime_transition_matrix=engine_input.transition_matrix,
            state_vol_multipliers=engine_input.state_vol_multipliers,
        )

        # This preserves the Rust-owned buffer when already a NumPy ndarray.
        losses_vector = np.asarray(losses)
        return EngineRunArtifacts(risk=risk, losses=losses_vector, latency_ms=latency_ms)
