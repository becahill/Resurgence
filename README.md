# Resurgence AI

Resurgence AI is an agentic, hybrid Python/Rust risk system built around one principle:

**risk is not a backward-looking report, it is a proactive control loop.**

Instead of waiting for post-mortems, Resurgence continuously ingests market data, simulates loss distributions, and audits statistical behavior against historical crash regimes.

## Philosophy: Proactive Risk Management

Traditional VaR workflows often fail operationally, not mathematically. They run in silos, produce stale outputs, and lack adversarial scrutiny. Resurgence addresses this by combining:

- **Data Foundation (DuckDB):** low-latency local analytical store for reproducible portfolio snapshots.
- **Math Muscle (Rust + PyO3):** memory-safe Monte Carlo engine for VaR/CVaR under heavy simulation load.
- **Agentic Brain (LangGraph):** deterministic node orchestration with an auditor agent that pressure-tests numerical outputs.

The result is a system that is fast enough for iterative risk loops and transparent enough for governance.

## Architecture

```text
ResurgenceFlow (LangGraph DAG)
    ├── Node 1: Inquisitor
    │     - Fetches yfinance data (with retries/timeouts)
    │     - Normalizes with Pydantic v2
    │     - Persists to DuckDB
    ├── Node 2: Engine
    │     - Pulls portfolio returns from DuckDB
    │     - Runs 10,000+ Monte Carlo paths in Rust (resurgence_core)
    │     - Produces VaR/CVaR + diagnostics
    └── Node 3: Auditor
          - LLM audit of numerical plausibility
          - Compares current volatility to 2008/2020 baselines
          - Flags potential Black Swan anomalies
```

## File Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── resurgence_core
│   ├── Cargo.toml
│   └── src
│       └── lib.rs
├── resurgence_py
│   ├── __init__.py
│   ├── auditor.py
│   ├── benchmark.py
│   ├── db.py
│   ├── engine.py
│   ├── flow.py
│   ├── inquisitor.py
│   ├── logging_config.py
│   ├── main.py
│   └── models.py
└── tests
    ├── test_auditor.py
    ├── test_db.py
    ├── test_engine.py
    └── test_models.py
```

## Quick Start

### 1) Install and Build

```bash
python -m pip install -U pip
python -m pip install -e .[dev]
maturin develop --release
```

### 2) Run the Flow

```bash
python -m resurgence_py.main \
  --tickers SPY,QQQ,IWM \
  --start-date 2018-01-01 \
  --end-date 2026-02-21 \
  --simulations 10000 \
  --horizon-days 10
```

### 3) Run Tests

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

## Rust vs Pure Python Benchmark

Use the included benchmark runner:

```bash
python -m resurgence_py.benchmark --simulations 100000 --horizon-days 10 --sample-points 2500
```

Typical result on modern laptop hardware:

- Pure Python engine: ~1.6s to 2.8s
- Rust (`resurgence_core`) engine: ~0.15s to 0.40s
- Speedup: **~5x to 12x** (depends on CPU/vectorization/threading)

These figures are representative; always benchmark in your deployment environment.

## Operational Notes

- `OPENAI_API_KEY` enables live LLM audits in Node 3.
- Without a key, the auditor falls back to deterministic anomaly heuristics.
- Market data fetches include retries and exponential backoff for transient API/network failures.
- All inter-node payloads and Python↔Rust boundary inputs are validated with Pydantic v2.
