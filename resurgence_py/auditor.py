from __future__ import annotations

import json
import logging
import os

from resurgence_py.models import AuditReport, AuditSeverity, CrashVolProfile, PortfolioSeries, RiskMetrics

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment,misc]


class LLMAuditor:
    """Node 3: audit numerical risk output for Black Swan anomalies."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        allow_live_llm: bool = True,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.allow_live_llm = allow_live_llm
        self._llm = self._build_llm() if allow_live_llm else None

    def _build_llm(self) -> ChatOpenAI | None:
        if ChatOpenAI is None:
            logger.warning("langchain-openai not installed; auditor will use deterministic fallback")
            return None
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY missing; auditor will use deterministic fallback")
            return None

        return ChatOpenAI(model=self.model_name, temperature=self.temperature)

    async def run(
        self,
        risk: RiskMetrics,
        portfolio: PortfolioSeries,
        crash_profile: CrashVolProfile,
    ) -> AuditReport:
        baseline = max(crash_profile.volatility_2008, crash_profile.volatility_2020, 1e-9)
        black_swan_ratio = portfolio.annualized_volatility / baseline

        if self._llm is not None:
            llm_report = await self._run_llm_audit(risk, portfolio, crash_profile, black_swan_ratio)
            if llm_report is not None:
                return llm_report

        return self._fallback_audit(risk, portfolio, crash_profile, black_swan_ratio)

    async def _run_llm_audit(
        self,
        risk: RiskMetrics,
        portfolio: PortfolioSeries,
        crash_profile: CrashVolProfile,
        black_swan_ratio: float,
    ) -> AuditReport | None:
        assert self._llm is not None

        prompt = (
            "You are the Auditor node in a financial risk pipeline. "
            "Given these metrics, decide if output is anomalous and return strict JSON only "
            "with keys: severity (pass|warning|critical), flagged (bool), summary, rationale, "
            "recommended_actions (array of short strings).\n\n"
            f"risk_metrics={risk.model_dump_json()}\n"
            f"portfolio_metrics={portfolio.model_dump_json()}\n"
            f"crash_profile={crash_profile.model_dump_json()}\n"
            f"black_swan_ratio={black_swan_ratio:.6f}\n"
            "Evaluate against crash regimes from 2008 and 2020, and flag mathematical anomalies."
        )

        try:
            response = await self._llm.ainvoke(prompt)
            response_text = str(response.content).strip()
            parsed = json.loads(self._extract_json(response_text))
            parsed["black_swan_ratio"] = black_swan_ratio
            return AuditReport.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM audit parse failed (%s); fallback heuristic engaged", exc)
            return None

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            msg = "No JSON object found in LLM response"
            raise ValueError(msg)
        return raw_text[start : end + 1]

    @staticmethod
    def _fallback_audit(
        risk: RiskMetrics,
        portfolio: PortfolioSeries,
        crash_profile: CrashVolProfile,
        black_swan_ratio: float,
    ) -> AuditReport:
        tail_ratio = risk.cvar / max(risk.var, 1e-9)
        crash_max = max(crash_profile.volatility_2008, crash_profile.volatility_2020, 1e-9)

        if black_swan_ratio >= 1.25 or tail_ratio >= 2.0:
            severity = AuditSeverity.CRITICAL
            flagged = True
            summary = "Black Swan alert: tail risk exceeds stress anchors"
        elif black_swan_ratio >= 0.90 or tail_ratio >= 1.6:
            severity = AuditSeverity.WARNING
            flagged = True
            summary = "Elevated tail behavior relative to historical crash baselines"
        else:
            severity = AuditSeverity.PASS
            flagged = False
            summary = "Risk profile is within historical stress boundaries"

        rationale = (
            f"Annualized volatility={portfolio.annualized_volatility:.4f}, "
            f"stress baseline={crash_max:.4f}, black_swan_ratio={black_swan_ratio:.4f}, "
            f"CVaR/ VaR ratio={tail_ratio:.4f}."
        )

        actions: list[str] = []
        if flagged:
            actions = [
                "Increase hedge ratio on highest-beta holdings",
                "Run intraday re-pricing with reduced liquidity assumptions",
                "Escalate to risk committee with scenario replay",
            ]

        return AuditReport(
            severity=severity,
            flagged=flagged,
            summary=summary,
            rationale=rationale,
            black_swan_ratio=black_swan_ratio,
            recommended_actions=actions,
        )
