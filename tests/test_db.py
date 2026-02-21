from __future__ import annotations

from datetime import date

from resurgence_py.db import init_db, read_portfolio_series, write_market_data
from resurgence_py.models import MarketDataRecord


def test_db_roundtrip(tmp_path) -> None:
    db_path = str(tmp_path / "resurgence_test.duckdb")
    init_db(db_path)

    rows = [
        MarketDataRecord(
            ticker="SPY",
            date=date(2024, 1, 2),
            close=470.0,
            adj_close=470.0,
            volume=100,
            daily_return=None,
        ),
        MarketDataRecord(
            ticker="SPY",
            date=date(2024, 1, 3),
            close=472.0,
            adj_close=472.0,
            volume=100,
            daily_return=0.004255,
        ),
        MarketDataRecord(
            ticker="SPY",
            date=date(2024, 1, 4),
            close=468.0,
            adj_close=468.0,
            volume=105,
            daily_return=-0.008475,
        ),
        MarketDataRecord(
            ticker="QQQ",
            date=date(2024, 1, 2),
            close=400.0,
            adj_close=400.0,
            volume=80,
            daily_return=None,
        ),
        MarketDataRecord(
            ticker="QQQ",
            date=date(2024, 1, 3),
            close=404.0,
            adj_close=404.0,
            volume=80,
            daily_return=0.01,
        ),
        MarketDataRecord(
            ticker="QQQ",
            date=date(2024, 1, 4),
            close=398.0,
            adj_close=398.0,
            volume=90,
            daily_return=-0.014851,
        ),
    ]

    write_market_data(db_path, rows)
    portfolio = read_portfolio_series(db_path, ["SPY", "QQQ"])

    assert portfolio.observation_count == 2
    assert len(portfolio.returns) == 2
