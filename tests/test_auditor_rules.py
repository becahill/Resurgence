from __future__ import annotations

import numpy as np

from resurgence_py.auditor import LLMAuditor
from resurgence_py.models import RiskMetrics


def _base_risk() -> RiskMetrics:
    return RiskMetrics(
        var=0.05,
        cvar=0.08,
        mean_loss=0.02,
        loss_stddev=0.02,
        confidence_level=0.95,
        simulations=10_000,
        horizon_days=10,
        estimated_drift=0.0001,
        estimated_volatility=0.03,
        sharpe_ratio=0.8,
        max_loss_zscore=2.5,
    )


def _rule_ids(anomalies: list) -> set[str]:
    return {anomaly.rule_id for anomaly in anomalies}


def test_var_monotonicity_consistency_rule() -> None:
    risk = _base_risk().model_copy(update={"var": 0.001, "cvar": 0.002})
    losses = np.array([0.0, 0.002, 0.004, 0.01, 0.03, 0.05, 0.08, 0.12], dtype=np.float64)

    anomalies = LLMAuditor.evaluate_deterministic_anomalies(risk=risk, losses=losses)

    assert "risk.var_consistency" in _rule_ids(anomalies)


def test_variance_non_negative_rule() -> None:
    risk = _base_risk()
    losses = np.array([1e308, 1e308 - 1e300, 1e308 - 2e300], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        anomalies = LLMAuditor.evaluate_deterministic_anomalies(risk=risk, losses=losses)

    assert "variance.non_negative" in _rule_ids(anomalies)


def test_distribution_mean_vol_bounds_rule() -> None:
    risk = _base_risk().model_copy(update={"mean_loss": 1.2, "loss_stddev": 1.1})
    losses = np.array([0.01, 0.02, 0.015, 0.008, 0.025, 0.018], dtype=np.float64)

    anomalies = LLMAuditor.evaluate_deterministic_anomalies(risk=risk, losses=losses)
    rule_ids = _rule_ids(anomalies)

    assert "distribution.mean_bounds" in rule_ids
    assert "distribution.volatility_bounds" in rule_ids


def test_mad_outlier_rule() -> None:
    rng = np.random.default_rng(19)
    baseline = rng.normal(loc=0.01, scale=0.001, size=200)
    losses = np.concatenate([baseline, np.array([0.2])]).astype(np.float64)

    anomalies = LLMAuditor.evaluate_deterministic_anomalies(risk=_base_risk(), losses=losses)

    assert "distribution.mad_outlier" in _rule_ids(anomalies)
