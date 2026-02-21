from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

from resurgence_py.flow import ResurgenceFlow
from resurgence_py.logging_config import configure_logging
from resurgence_py.models import FlowInput


async def _run(args: argparse.Namespace) -> None:
    request = FlowInput(
        tickers=[part.strip() for part in args.tickers.split(",") if part.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        confidence_level=args.confidence_level,
        simulations=args.simulations,
        horizon_days=args.horizon_days,
        seed=args.seed,
        llm_model=args.llm_model,
    )

    flow = ResurgenceFlow(db_path=args.db_path)
    result = await flow.run(request)
    print(result.model_dump_json(indent=2))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def cli() -> None:
    today = date.today()
    default_start = today - timedelta(days=365 * 6)

    parser = argparse.ArgumentParser(description="Run the Resurgence AI risk graph")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers")
    parser.add_argument("--start-date", type=_parse_date, default=default_start)
    parser.add_argument("--end-date", type=_parse_date, default=today)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--db-path", type=str, default="resurgence.duckdb")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(_run(args))


if __name__ == "__main__":
    cli()
