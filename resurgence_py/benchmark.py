from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from resurgence_py.engine import simulate_regime_var_cvar_python
from resurgence_py.models import (
    default_state_drift_adjustments,
    default_state_vol_multipliers,
    default_transition_matrix,
)

try:
    import resurgence_core
except Exception:  # noqa: BLE001
    resurgence_core = None


DEFAULT_PATH_COUNTS = [1_000, 10_000, 100_000, 1_000_000]


@dataclass(slots=True)
class BenchmarkCaseResult:
    engine: str
    simulations: int
    runtime_s: float
    peak_rss_mb: float
    delta_rss_mb: float
    var: float
    cvar: float
    available: bool
    error: str | None = None


def _max_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss) / (1024.0 * 1024.0)
    return float(max_rss) / 1024.0


def _generate_returns(sample_points: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0004, scale=0.0125, size=sample_points).tolist()


def _run_python_engine(
    returns: list[float],
    simulations: int,
    horizon_days: int,
    seed: int,
) -> tuple[float, float]:
    result, _ = simulate_regime_var_cvar_python(
        returns=returns,
        confidence_level=0.95,
        simulations=simulations,
        horizon_days=horizon_days,
        transition_matrix=default_transition_matrix(),
        state_vol_multipliers=default_state_vol_multipliers(),
        state_drift_adjustments=default_state_drift_adjustments(),
        initial_state=2,
        seed=seed,
    )
    return result.var, result.cvar


def _run_rust_engine(
    returns: list[float],
    simulations: int,
    horizon_days: int,
    seed: int,
) -> tuple[float, float]:
    if resurgence_core is None or not hasattr(resurgence_core, "simulate_regime_var_cvar"):
        msg = "resurgence_core extension is unavailable"
        raise RuntimeError(msg)

    result, _ = resurgence_core.simulate_regime_var_cvar(
        returns,
        0.95,
        simulations,
        horizon_days,
        default_transition_matrix(),
        default_state_vol_multipliers(),
        default_state_drift_adjustments(),
        2,
        seed,
        None,
    )
    return float(result.var), float(result.cvar)


def _run_worker_case(
    engine: str,
    simulations: int,
    horizon_days: int,
    sample_points: int,
    seed: int,
) -> BenchmarkCaseResult:
    returns = _generate_returns(sample_points=sample_points, seed=seed)
    baseline_rss_mb = _max_rss_mb()
    start = time.perf_counter()

    try:
        if engine == "python":
            var, cvar = _run_python_engine(
                returns=returns,
                simulations=simulations,
                horizon_days=horizon_days,
                seed=seed,
            )
        elif engine == "rust":
            var, cvar = _run_rust_engine(
                returns=returns,
                simulations=simulations,
                horizon_days=horizon_days,
                seed=seed,
            )
        else:
            msg = f"Unknown engine '{engine}'"
            raise ValueError(msg)
    except Exception as exc:  # noqa: BLE001
        return BenchmarkCaseResult(
            engine=engine,
            simulations=simulations,
            runtime_s=0.0,
            peak_rss_mb=_max_rss_mb(),
            delta_rss_mb=0.0,
            var=float("nan"),
            cvar=float("nan"),
            available=False,
            error=str(exc),
        )

    runtime_s = time.perf_counter() - start
    peak_rss_mb = _max_rss_mb()

    return BenchmarkCaseResult(
        engine=engine,
        simulations=simulations,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
        delta_rss_mb=max(peak_rss_mb - baseline_rss_mb, 0.0),
        var=float(var),
        cvar=float(cvar),
        available=True,
        error=None,
    )


def _invoke_worker(
    engine: str,
    simulations: int,
    horizon_days: int,
    sample_points: int,
    seed: int,
) -> BenchmarkCaseResult:
    command = [
        sys.executable,
        "-m",
        "resurgence_py.benchmark",
        "--worker",
        "--engine",
        engine,
        "--simulations",
        str(simulations),
        "--horizon-days",
        str(horizon_days),
        "--sample-points",
        str(sample_points),
        "--seed",
        str(seed),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    if completed.returncode != 0:
        return BenchmarkCaseResult(
            engine=engine,
            simulations=simulations,
            runtime_s=0.0,
            peak_rss_mb=0.0,
            delta_rss_mb=0.0,
            var=float("nan"),
            cvar=float("nan"),
            available=False,
            error=completed.stderr.strip() or "Worker benchmark failed",
        )

    raw = completed.stdout.strip().splitlines()[-1]
    payload = json.loads(raw)
    return BenchmarkCaseResult(**payload)


def _format_simulations(simulations: int) -> str:
    if simulations >= 1_000_000:
        return f"{simulations / 1_000_000:.1f}M"
    if simulations >= 1_000:
        return f"{simulations / 1_000:.0f}k"
    return str(simulations)


def _render_markdown(
    records: list[BenchmarkCaseResult],
    path_counts: list[int],
    sample_points: int,
    horizon_days: int,
    seed: int,
    generated_at: datetime,
) -> str:
    by_case: dict[tuple[int, str], BenchmarkCaseResult] = {
        (record.simulations, record.engine): record for record in records
    }

    lines = [
        "# Benchmarks",
        "",
        f"Generated at: {generated_at.isoformat()}",
        f"Host: {platform.platform()}",
        f"Python: {platform.python_version()}",
        "",
        "Configuration:",
        f"- Path counts: {', '.join(_format_simulations(value) for value in path_counts)}",
        f"- Horizon days: {horizon_days}",
        f"- Sample points: {sample_points}",
        f"- Seed: {seed}",
        "",
        "| Paths | Python runtime (s) | Python peak RSS (MB) | Rust runtime (s) | Rust peak RSS (MB) | Speedup (x) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for simulations in path_counts:
        python_case = by_case.get((simulations, "python"))
        rust_case = by_case.get((simulations, "rust"))

        python_runtime = "n/a"
        python_memory = "n/a"
        rust_runtime = "n/a"
        rust_memory = "n/a"
        speedup = "n/a"

        if python_case is not None and python_case.available:
            python_runtime = f"{python_case.runtime_s:.4f}"
            python_memory = f"{python_case.peak_rss_mb:.2f}"

        if rust_case is not None and rust_case.available:
            rust_runtime = f"{rust_case.runtime_s:.4f}"
            rust_memory = f"{rust_case.peak_rss_mb:.2f}"

        if (
            python_case is not None
            and rust_case is not None
            and python_case.available
            and rust_case.available
            and rust_case.runtime_s > 0
        ):
            speedup_value = python_case.runtime_s / rust_case.runtime_s
            speedup = f"{speedup_value:.2f}"

        lines.append(
            f"| {_format_simulations(simulations)} | {python_runtime} | {python_memory} | "
            f"{rust_runtime} | {rust_memory} | {speedup} |"
        )

    rust_errors = [
        record.error for record in records if record.engine == "rust" and not record.available and record.error
    ]
    if rust_errors:
        lines.extend(
            [
                "",
                "Rust notes:",
                f"- {rust_errors[0]}",
            ]
        )

    lines.extend(
        [
            "",
            "Raw machine-readable output: `benchmarks/results.json`.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_benchmark_suite(
    path_counts: list[int] | None = None,
    horizon_days: int = 10,
    sample_points: int = 2_500,
    seed: int = 7,
    output_json: str | Path = "benchmarks/results.json",
    output_markdown: str | Path = "benchmarks/results.md",
) -> dict[str, object]:
    selected_paths = DEFAULT_PATH_COUNTS if path_counts is None else path_counts

    records: list[BenchmarkCaseResult] = []
    for simulations in selected_paths:
        for engine in ["python", "rust"]:
            result = _invoke_worker(
                engine=engine,
                simulations=simulations,
                horizon_days=horizon_days,
                sample_points=sample_points,
                seed=seed,
            )
            records.append(result)

    generated_at = datetime.now(UTC)
    payload = {
        "generated_at_utc": generated_at.isoformat(),
        "path_counts": selected_paths,
        "horizon_days": horizon_days,
        "sample_points": sample_points,
        "seed": seed,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "results": [asdict(record) for record in records],
    }

    output_json_path = Path(output_json)
    output_markdown_path = Path(output_markdown)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    markdown = _render_markdown(
        records=records,
        path_counts=selected_paths,
        sample_points=sample_points,
        horizon_days=horizon_days,
        seed=seed,
        generated_at=generated_at,
    )
    output_markdown_path.write_text(markdown, encoding="utf-8")

    return payload


def run_benchmark(simulations: int, horizon_days: int, sample_points: int) -> dict[str, float]:
    """Backwards-compatible single-case benchmark helper."""
    python_case = _invoke_worker(
        engine="python",
        simulations=simulations,
        horizon_days=horizon_days,
        sample_points=sample_points,
        seed=7,
    )
    rust_case = _invoke_worker(
        engine="rust",
        simulations=simulations,
        horizon_days=horizon_days,
        sample_points=sample_points,
        seed=7,
    )

    rust_runtime = rust_case.runtime_s if rust_case.available else float("nan")
    speedup = (
        python_case.runtime_s / rust_runtime
        if np.isfinite(rust_runtime) and rust_runtime > 0
        else float("nan")
    )

    return {
        "python_time_s": python_case.runtime_s,
        "python_var": python_case.var,
        "python_cvar": python_case.cvar,
        "rust_time_s": rust_runtime,
        "rust_var": rust_case.var,
        "rust_cvar": rust_case.cvar,
        "speedup_x": speedup,
    }


def _worker_cli(args: argparse.Namespace) -> None:
    result = _run_worker_case(
        engine=args.engine,
        simulations=args.simulations,
        horizon_days=args.horizon_days,
        sample_points=args.sample_points,
        seed=args.seed,
    )
    print(json.dumps(asdict(result)))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Rust vs Python VaR/CVaR engine")
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--sample-points", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--paths", type=str, default="1000,10000,100000,1000000")
    parser.add_argument("--output-json", type=str, default="benchmarks/results.json")
    parser.add_argument("--output-markdown", type=str, default="benchmarks/results.md")
    parser.add_argument("--engine", type=str, choices=["python", "rust"], default="python")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker_cli(args)
        return

    if args.paths:
        path_counts = [int(value.strip()) for value in args.paths.split(",") if value.strip()]
    else:
        path_counts = [args.simulations]

    payload = run_benchmark_suite(
        path_counts=path_counts,
        horizon_days=args.horizon_days,
        sample_points=args.sample_points,
        seed=args.seed,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    cli()
