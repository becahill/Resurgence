from __future__ import annotations

from resurgence_py.auditor import LLMAuditor
from resurgence_py.models import CrashVolProfile, PortfolioSeries, RiskMetrics


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

    report = __import__("asyncio").run(
        auditor.run(risk=risk, portfolio=portfolio, crash_profile=crash_profile)
    )

    assert report.flagged is True
    assert report.severity in {"warning", "critical"}
    assert report.black_swan_ratio > 1.0
