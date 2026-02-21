from __future__ import annotations

import argparse
import time

import numpy as np

from resurgence_py.engine import simulate_var_cvar_python

try:
    import resurgence_core
except Exception:  # noqa: BLE001
    resurgence_core = None


def run_benchmark(simulations: int, horizon_days: int, sample_points: int) -> dict[str, float]:
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.0004, scale=0.0125, size=sample_points).tolist()

    t0 = time.perf_counter()
    python_metrics = simulate_var_cvar_python(
        returns=returns,
        confidence_level=0.95,
        simulations=simulations,
        horizon_days=horizon_days,
        seed=42,
    )
    python_time = time.perf_counter() - t0

    rust_time = float("nan")
    rust_var = float("nan")
    rust_cvar = float("nan")

    if resurgence_core is not None:
        t1 = time.perf_counter()
        rust_result = resurgence_core.simulate_var_cvar(
            returns,
            0.95,
            simulations,
            horizon_days,
            42,
        )
        rust_time = time.perf_counter() - t1
        rust_var = float(rust_result.var)
        rust_cvar = float(rust_result.cvar)

    speedup = python_time / rust_time if rust_time and rust_time > 0 else float("nan")

    return {
        "python_time_s": python_time,
        "python_var": python_metrics.var,
        "python_cvar": python_metrics.cvar,
        "rust_time_s": rust_time,
        "rust_var": rust_var,
        "rust_cvar": rust_cvar,
        "speedup_x": speedup,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Rust vs Python VaR/CVaR engine")
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--sample-points", type=int, default=2_500)
    args = parser.parse_args()

    result = run_benchmark(
        simulations=args.simulations,
        horizon_days=args.horizon_days,
        sample_points=args.sample_points,
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    cli()
