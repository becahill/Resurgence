from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from resurgence_py.db import init_db, read_crash_profile, read_portfolio_series, write_market_data
from resurgence_py.models import (
    CrashVolProfile,
    DataPullConfig,
    FlowInput,
    MarketDataRecord,
    PortfolioSeries,
)

logger = logging.getLogger(__name__)


class DataFetchError(RuntimeError):
    """Raised when all fetch retries fail for a ticker."""


class Inquisitor:
    """Node 1: pull market data and persist to DuckDB."""

    def __init__(
        self,
        db_path: str,
        request_timeout_s: float = 20.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
    ) -> None:
        self.db_path = db_path
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        init_db(self.db_path)

    async def run(
        self,
        payload: FlowInput,
        pull_config: DataPullConfig,
    ) -> tuple[PortfolioSeries, CrashVolProfile]:
        records = await self._fetch_all(payload, pull_config)
        await asyncio.to_thread(write_market_data, self.db_path, records)

        portfolio = await asyncio.to_thread(read_portfolio_series, self.db_path, payload.tickers)
        crash_profile = await asyncio.to_thread(read_crash_profile, self.db_path, payload.tickers)
        return portfolio, crash_profile

    async def _fetch_all(self, payload: FlowInput, pull_config: DataPullConfig) -> list[MarketDataRecord]:
        tasks = [
            self._fetch_with_retry(
                ticker=ticker,
                start_date=pull_config.start_date,
                end_date=pull_config.end_date,
                timeout_s=pull_config.request_timeout_s,
            )
            for ticker in payload.tickers
        ]

        batches = await asyncio.gather(*tasks)
        records = [record for batch in batches for record in batch]
        logger.info("Inquisitor collected %d normalized rows", len(records))

        if not records:
            msg = "Inquisitor produced no market records"
            raise DataFetchError(msg)

        return records

    async def _fetch_with_retry(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeout_s: float,
    ) -> list[MarketDataRecord]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._fetch_ticker_history,
                    ticker,
                    start_date,
                    end_date,
                    timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    logger.exception(
                        "Ticker %s failed after %d attempts",
                        ticker,
                        self.max_retries,
                    )
                    raise DataFetchError(f"Failed to fetch {ticker}") from exc

                sleep_s = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Fetch attempt %d/%d failed for %s (%s). retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    ticker,
                    exc,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)

        msg = f"Unreachable retry state for {ticker}"
        raise DataFetchError(msg)

    def _fetch_ticker_history(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeout_s: float,
    ) -> list[MarketDataRecord]:
        # yfinance uses an exclusive end-date.
        end_exclusive = end_date + timedelta(days=1)
        frame = yf.Ticker(ticker).history(
            start=start_date.isoformat(),
            end=end_exclusive.isoformat(),
            interval="1d",
            auto_adjust=False,
            timeout=timeout_s,
        )

        if frame.empty:
            msg = f"No rows returned for ticker {ticker}"
            raise DataFetchError(msg)

        normalized = self._normalize_frame(ticker, frame)
        if len(normalized) < 2:
            msg = f"Insufficient rows returned for ticker {ticker}"
            raise DataFetchError(msg)

        return normalized

    def _normalize_frame(self, ticker: str, frame: pd.DataFrame) -> list[MarketDataRecord]:
        frame = frame.copy()
        close_col = "Close"
        adj_col = "Adj Close" if "Adj Close" in frame.columns else "Close"

        frame["date"] = frame.index.date
        frame["daily_return"] = frame[close_col].pct_change()

        records: list[MarketDataRecord] = []
        for _, row in frame.iterrows():
            daily_return = row["daily_return"]
            records.append(
                MarketDataRecord(
                    ticker=ticker,
                    date=row["date"],
                    close=float(row[close_col]),
                    adj_close=float(row[adj_col]),
                    volume=int(row.get("Volume", 0) or 0),
                    daily_return=None if pd.isna(daily_return) else float(daily_return),
                )
            )

        return records
