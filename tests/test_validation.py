from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from validation.backtest import BacktestConfig, run_backtest_on_returns, write_backtest_results
from validation.tests import christoffersen_independence_test, kupiec_proportion_of_failures


def test_kupiec_pof_detects_excessive_breaches() -> None:
    breaches = [False] * 80 + [True] * 20
    result = kupiec_proportion_of_failures(breaches, confidence_level=0.95)

    assert result.observations == 100
    assert result.breaches == 20
    assert result.reject_at_95 is True


def test_kupiec_pof_accepts_calibrated_sequence() -> None:
    breaches = [False] * 95 + [True] * 5
    result = kupiec_proportion_of_failures(breaches, confidence_level=0.95)

    assert result.reject_at_95 is False
    assert abs(result.observed_breach_rate - 0.05) < 1e-12


def test_christoffersen_detects_clustered_breaches() -> None:
    clustered = [False] * 90 + [True] * 10
    result = christoffersen_independence_test(clustered)

    assert result.applicable is True
    assert result.reject_at_95 is True


def test_run_backtest_on_returns_writes_json_and_csv(tmp_path) -> None:
    rng = np.random.default_rng(11)
    returns = pd.Series(
        rng.normal(loc=0.0003, scale=0.011, size=420),
        index=pd.bdate_range(start=datetime(2022, 1, 3), periods=420),
        name="portfolio_return",
    )

    config = BacktestConfig(
        tickers=["SPY", "QQQ", "TLT"],
        weights=[0.5, 0.3, 0.2],
        confidence_level=0.95,
        rolling_window=126,
        method="historical",
        simulations=1_000,
        seed=7,
    )

    observations, summary = run_backtest_on_returns(returns, config)
    csv_path, json_path = write_backtest_results(
        observations=observations,
        summary=summary,
        output_dir=tmp_path / "validation_results",
    )

    assert len(observations) == len(returns) - config.rolling_window
    assert summary.observation_count == len(observations)
    assert 0.0 <= summary.breach_rate <= 1.0

    with open(csv_path, encoding="utf-8") as handle:
        csv_rows = handle.readlines()
    assert len(csv_rows) == len(observations) + 1

    with open(json_path, encoding="utf-8") as handle:
        summary_json = json.load(handle)
    assert summary_json["observation_count"] == len(observations)
