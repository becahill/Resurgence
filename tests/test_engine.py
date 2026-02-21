from __future__ import annotations

import numpy as np

from resurgence_py.engine import simulate_var_cvar_python


def test_python_engine_generates_consistent_risk_metrics() -> None:
    returns = np.random.default_rng(123).normal(0.0005, 0.01, 500).tolist()

    result = simulate_var_cvar_python(
        returns=returns,
        confidence_level=0.95,
        simulations=10_000,
        horizon_days=10,
        seed=42,
    )

    assert result.var >= 0
    assert result.cvar >= result.var
    assert result.loss_stddev >= 0
