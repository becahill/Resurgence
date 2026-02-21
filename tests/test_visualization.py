from __future__ import annotations

import numpy as np

from resurgence_py.visualization import rolling_var_frame, scenario_comparison_table
from validation.backtest import BacktestObservation


def test_rolling_var_frame_normalizes_models_and_dicts() -> None:
    observations = [
        BacktestObservation(
            date="2024-01-03",
            realized_return=-0.01,
            realized_loss=0.01,
            predicted_var=0.015,
            predicted_cvar=0.021,
            breach=False,
        ),
        {
            "date": "2024-01-04",
            "realized_return": -0.02,
            "realized_loss": 0.02,
            "predicted_var": 0.017,
            "predicted_cvar": 0.023,
            "breach": True,
        },
    ]

    frame = rolling_var_frame(observations)

    assert list(frame.columns) == ["date", "realized_loss", "predicted_var", "predicted_cvar", "breach"]
    assert len(frame) == 2
    assert bool(frame.loc[1, "breach"])


def test_scenario_comparison_table_computes_tail_metrics() -> None:
    scenarios = {
        "baseline": np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64),
        "stress": np.array([0.02, 0.03, 0.06, 0.09], dtype=np.float64),
    }

    summary = scenario_comparison_table(scenarios, confidence_level=0.75)

    assert list(summary["scenario"]) == ["baseline", "stress"]
    assert float(summary.loc[1, "var"]) >= float(summary.loc[0, "var"])
    assert float(summary.loc[1, "cvar"]) >= float(summary.loc[1, "var"])
