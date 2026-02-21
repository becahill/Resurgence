from __future__ import annotations

import asyncio
from datetime import date

import numpy as np

from resurgence_py.auditor import LLMAuditor
from resurgence_py.models import CrashVolProfile, DataPullConfig, PortfolioSeries, RiskMetrics


def test_fallback_auditor_flags_extreme_regime() -> None:
    auditor = LLMAuditor(model_name="gpt-4o-mini", allow_live_llm=False)

    risk = RiskMetrics(
        var=0.08,
        cvar=0.18,
        mean_loss=0.03,
        loss_stddev=0.04,
        confidence_level=0.95,
        simulations=10_000,
        horizon_days=10,
        estimated_drift=0.0001,
        estimated_volatility=0.03,
    )
    portfolio = PortfolioSeries(
        returns=[0.01, -0.03, 0.02, -0.01],
        annualized_volatility=0.85,
        observation_count=4,
    )
    crash_profile = CrashVolProfile(
        volatility_2008=0.55,
        volatility_2020=0.60,
        sample_size_2008=200,
        sample_size_2020=240,
    )
    pull_config = DataPullConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
        request_timeout_s=20.0,
    )
    losses = np.array([0.01, 0.04, 0.05, 0.07, 0.12], dtype=np.float64)

    report = asyncio.run(
        auditor.run(
            risk=risk,
            portfolio=portfolio,
            crash_profile=crash_profile,
            losses=losses,
            pull_config=pull_config,
            rerun_count=0,
            max_reruns=2,
        )
    )

    assert report.flagged is True
    assert report.severity in {"warning", "critical"}
    assert report.black_swan_ratio > 1.0
    assert isinstance(report.anomalies, list)


def test_auditor_requests_rerun_on_hallucination() -> None:
    auditor = LLMAuditor(model_name="gpt-4o-mini", allow_live_llm=False)

    risk = RiskMetrics(
        var=0.04,
        cvar=0.08,
        mean_loss=0.02,
        loss_stddev=0.01,
        confidence_level=0.95,
        simulations=10_000,
        horizon_days=10,
        estimated_drift=0.0001,
        estimated_volatility=0.03,
        sharpe_ratio=9.0,
        max_loss_zscore=12.0,
    )
    portfolio = PortfolioSeries(
        returns=[0.01, -0.03, 0.02, -0.01],
        annualized_volatility=0.35,
        observation_count=4,
    )
    crash_profile = CrashVolProfile(
        volatility_2008=0.55,
        volatility_2020=0.60,
        sample_size_2008=200,
        sample_size_2020=240,
    )
    pull_config = DataPullConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
        request_timeout_s=20.0,
    )
    losses = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)

    report = asyncio.run(
        auditor.run(
            risk=risk,
            portfolio=portfolio,
            crash_profile=crash_profile,
            losses=losses,
            pull_config=pull_config,
            rerun_count=0,
            max_reruns=2,
        )
    )

    assert report.hallucination_detected is True
    assert report.requires_rerun is True
    assert report.suggested_lookback_extension_days >= 252
    assert any(anomaly.rule_id == "metrics.sharpe_bounds" for anomaly in report.anomalies)
