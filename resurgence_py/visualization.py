from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None


def _require_matplotlib() -> None:
    if plt is None:
        msg = "matplotlib is required for plotting. Install with `python3 -m pip install matplotlib`."
        raise RuntimeError(msg)


def _normalize_observation_row(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "model_dump"):
        payload = observation.model_dump(mode="python")
    elif isinstance(observation, Mapping):
        payload = dict(observation)
    else:
        msg = "Observations must be mapping-like or Pydantic models with model_dump()"
        raise TypeError(msg)

    required = {"date", "realized_loss", "predicted_var", "predicted_cvar", "breach"}
    missing = required.difference(payload)
    if missing:
        msg = f"Observation is missing required keys: {sorted(missing)}"
        raise ValueError(msg)

    return payload


def rolling_var_frame(observations: Sequence[Any]) -> pd.DataFrame:
    """Build a tidy frame for rolling VaR/CVaR visualization."""
    rows = [_normalize_observation_row(observation) for observation in observations]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame[["date", "realized_loss", "predicted_var", "predicted_cvar", "breach"]]


def scenario_comparison_table(
    scenario_losses: Mapping[str, Sequence[float]],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute comparable tail-risk summaries across named scenarios."""
    rows: list[dict[str, float | str]] = []

    for name, losses in scenario_losses.items():
        values = np.asarray(losses, dtype=np.float64)
        if values.size == 0:
            msg = f"Scenario '{name}' must include at least one loss value"
            raise ValueError(msg)

        var = float(np.quantile(values, confidence_level))
        tail = values[values >= var]
        cvar = float(np.mean(tail)) if tail.size > 0 else var

        rows.append(
            {
                "scenario": name,
                "mean_loss": float(np.mean(values)),
                "var": var,
                "cvar": cvar,
                "max_loss": float(np.max(values)),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("scenario").reset_index(drop=True)


def plot_loss_distribution(
    losses: Sequence[float],
    output_path: str | Path,
    bins: int = 50,
    confidence_level: float = 0.95,
) -> str:
    """Render a histogram of the simulated loss distribution."""
    _require_matplotlib()

    values = np.asarray(losses, dtype=np.float64)
    if values.size == 0:
        msg = "losses must contain at least one value"
        raise ValueError(msg)

    var = float(np.quantile(values, confidence_level))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.hist(values, bins=bins, color="#1769aa", alpha=0.82, edgecolor="white", linewidth=0.5)
    axis.axvline(var, color="#c62828", linestyle="--", linewidth=1.5, label=f"VaR {confidence_level:.0%}")
    axis.set_title("Simulated Loss Distribution")
    axis.set_xlabel("Loss")
    axis.set_ylabel("Frequency")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_rolling_var(
    observations: Sequence[Any],
    output_path: str | Path,
) -> str:
    """Plot realized losses versus rolling VaR/CVaR forecasts."""
    _require_matplotlib()
    frame = rolling_var_frame(observations)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(frame["date"], frame["predicted_var"], label="Predicted VaR", linewidth=1.4, color="#1565c0")
    axis.plot(frame["date"], frame["predicted_cvar"], label="Predicted CVaR", linewidth=1.4, color="#00897b")
    axis.plot(frame["date"], frame["realized_loss"], label="Realized Loss", linewidth=1.0, color="#37474f")

    breaches = frame[frame["breach"]]
    if not breaches.empty:
        axis.scatter(
            breaches["date"],
            breaches["realized_loss"],
            color="#d32f2f",
            s=20,
            label="VaR Breach",
            zorder=5,
        )

    axis.set_title("Rolling VaR/CVaR Backtest")
    axis.set_xlabel("Date")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_scenario_comparison(
    scenario_losses: Mapping[str, Sequence[float]],
    output_path: str | Path,
    confidence_level: float = 0.95,
) -> str:
    """Render scenario comparison chart for mean loss, VaR, and CVaR."""
    _require_matplotlib()

    summary = scenario_comparison_table(
        scenario_losses=scenario_losses,
        confidence_level=confidence_level,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(summary))
    width = 0.25

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width, summary["mean_loss"], width=width, label="Mean Loss", color="#546e7a")
    axis.bar(x, summary["var"], width=width, label=f"VaR {confidence_level:.0%}", color="#1e88e5")
    axis.bar(x + width, summary["cvar"], width=width, label=f"CVaR {confidence_level:.0%}", color="#43a047")

    axis.set_xticks(x)
    axis.set_xticklabels(summary["scenario"], rotation=0)
    axis.set_ylabel("Loss")
    axis.set_title("Scenario Tail-Risk Comparison")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)
