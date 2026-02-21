from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from math import sqrt

import duckdb
import numpy as np
import pandas as pd

from resurgence_py.models import CrashVolProfile, MarketDataRecord, PortfolioSeries

logger = logging.getLogger(__name__)


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database=db_path)
    connection.execute("PRAGMA threads=4;")
    return connection


def init_db(db_path: str) -> None:
    """Initialize DuckDB schema if missing."""
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                close DOUBLE NOT NULL,
                adj_close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                daily_return DOUBLE,
                PRIMARY KEY (ticker, date)
            );
            """
        )
    logger.info("DuckDB schema initialized at %s", db_path)


def write_market_data(db_path: str, records: Sequence[MarketDataRecord]) -> None:
    """Upsert normalized market rows into DuckDB."""
    if not records:
        msg = "No records supplied for write_market_data"
        raise ValueError(msg)

    frame = pd.DataFrame([record.model_dump() for record in records])
    with _connect(db_path) as connection:
        connection.register("market_df", frame)
        connection.execute(
            """
            INSERT OR REPLACE INTO price_history (ticker, date, close, adj_close, volume, daily_return)
            SELECT ticker, date, close, adj_close, volume, daily_return
            FROM market_df;
            """
        )
        connection.unregister("market_df")

    logger.info("Persisted %d rows into %s", len(records), db_path)


def _query_portfolio_returns(db_path: str, tickers: Sequence[str]) -> list[float]:
    placeholders = ", ".join(["?"] * len(tickers))
    sql = f"""
        WITH portfolio_daily AS (
            SELECT date, AVG(daily_return) AS portfolio_return
            FROM price_history
            WHERE ticker IN ({placeholders})
              AND daily_return IS NOT NULL
            GROUP BY date
        )
        SELECT portfolio_return
        FROM portfolio_daily
        ORDER BY date;
    """

    with _connect(db_path) as connection:
        rows = connection.execute(sql, list(tickers)).fetchall()

    returns = [float(row[0]) for row in rows if row[0] is not None]
    if len(returns) < 2:
        msg = "Insufficient return history after ingestion"
        raise ValueError(msg)

    return returns


def _annualized_volatility(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(np.array(returns, dtype=float), ddof=1) * sqrt(252.0))


def read_portfolio_series(db_path: str, tickers: Sequence[str]) -> PortfolioSeries:
    """Build portfolio return series and realized volatility from DuckDB."""
    returns = _query_portfolio_returns(db_path, tickers)
    return PortfolioSeries(
        returns=returns,
        annualized_volatility=_annualized_volatility(returns),
        observation_count=len(returns),
    )


def _query_date_window_returns(
    db_path: str,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
) -> list[float]:
    placeholders = ", ".join(["?"] * len(tickers))
    sql = f"""
        WITH portfolio_daily AS (
            SELECT date, AVG(daily_return) AS portfolio_return
            FROM price_history
            WHERE ticker IN ({placeholders})
              AND daily_return IS NOT NULL
              AND date BETWEEN ? AND ?
            GROUP BY date
        )
        SELECT portfolio_return
        FROM portfolio_daily
        ORDER BY date;
    """

    params = [*tickers, start_date.isoformat(), end_date.isoformat()]
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [float(row[0]) for row in rows if row[0] is not None]


def read_crash_profile(db_path: str, tickers: Sequence[str]) -> CrashVolProfile:
    """Compute realized volatility anchors from 2008 and 2020 market regimes."""
    returns_2008 = _query_date_window_returns(
        db_path=db_path,
        tickers=tickers,
        start_date=date(2008, 1, 1),
        end_date=date(2008, 12, 31),
    )
    returns_2020 = _query_date_window_returns(
        db_path=db_path,
        tickers=tickers,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )

    profile = CrashVolProfile(
        volatility_2008=_annualized_volatility(returns_2008),
        volatility_2020=_annualized_volatility(returns_2020),
        sample_size_2008=len(returns_2008),
        sample_size_2020=len(returns_2020),
    )
    logger.info(
        "Crash profile computed: vol_2008=%.4f vol_2020=%.4f",
        profile.volatility_2008,
        profile.volatility_2020,
    )
    return profile
