from __future__ import annotations

import json
import logging
import os

import numpy as np

from resurgence_py.models import (
    AuditReport,
    AuditSeverity,
    CrashVolProfile,
    DataPullConfig,
    PortfolioSeries,
    RiskMetrics,
)

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment,misc]


class LLMAuditor:
    """Node 3: audit numerical risk output for Black Swan and hallucination anomalies."""

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
        losses: np.ndarray,
        pull_config: DataPullConfig,
        rerun_count: int,
        max_reruns: int,
    ) -> AuditReport:
        baseline = max(crash_profile.volatility_2008, crash_profile.volatility_2020, 1e-9)
        black_swan_ratio = portfolio.annualized_volatility / baseline
        tail_ratio = risk.cvar / max(risk.var, 1e-9)

        hallucination_reasons = self._detect_hallucinations(risk, losses)
        hallucination_detected = len(hallucination_reasons) > 0

        if hallucination_detected:
            severity = AuditSeverity.CRITICAL
            flagged = True
            summary = "Quantitative hallucination detected in simulation output"
        elif black_swan_ratio >= 1.25 or tail_ratio >= 2.0:
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

        requires_rerun = hallucination_detected and rerun_count < max_reruns
        suggested_extension_days = 252 * (rerun_count + 1) if requires_rerun else 0
        suggested_timeout_s = min(pull_config.request_timeout_s + 5.0, 60.0) if requires_rerun else None

        rationale = (
            f"Annualized volatility={portfolio.annualized_volatility:.4f}, baseline={baseline:.4f}, "
            f"black_swan_ratio={black_swan_ratio:.4f}, tail_ratio={tail_ratio:.4f}, "
            f"sharpe={risk.sharpe_ratio:.4f}, max_loss_zscore={risk.max_loss_zscore:.4f}."
        )
        if hallucination_reasons:
            rationale = f"{rationale} Hallucination triggers: {'; '.join(hallucination_reasons)}"

        recommended_actions: list[str] = []
        if flagged:
            recommended_actions.extend(
                [
                    "Increase hedge ratio on highest-beta holdings",
                    "Run intraday re-pricing with reduced liquidity assumptions",
                    "Escalate to risk committee with scenario replay",
                ]
            )
        if requires_rerun:
            recommended_actions.insert(0, "Rerun Inquisitor with expanded lookback and elevated timeout")

        llm_overlay = await self._run_llm_overlay(
            risk=risk,
            black_swan_ratio=black_swan_ratio,
            hallucination_reasons=hallucination_reasons,
            default_summary=summary,
            default_rationale=rationale,
            default_actions=recommended_actions,
        )
        if llm_overlay is not None:
            summary = llm_overlay["summary"]
            rationale = llm_overlay["rationale"]
            recommended_actions = llm_overlay["recommended_actions"]

        return AuditReport(
            severity=severity,
            flagged=flagged,
            summary=summary,
            rationale=rationale,
            black_swan_ratio=black_swan_ratio,
            hallucination_detected=hallucination_detected,
            requires_rerun=requires_rerun,
            suggested_lookback_extension_days=suggested_extension_days,
            suggested_timeout_s=suggested_timeout_s,
            detected_sharpe=risk.sharpe_ratio,
            detected_zscore=risk.max_loss_zscore,
            recommended_actions=recommended_actions,
        )

    @staticmethod
    def _detect_hallucinations(risk: RiskMetrics, losses: np.ndarray) -> list[str]:
        reasons: list[str] = []

        if not np.isfinite(risk.sharpe_ratio) or abs(risk.sharpe_ratio) > 6.0:
            reasons.append("nonsensical Sharpe ratio")

        if not np.isfinite(risk.max_loss_zscore) or abs(risk.max_loss_zscore) > 10.0:
            reasons.append("extreme loss z-score")

        if losses.size == 0:
            reasons.append("empty loss vector")
        else:
            if np.isnan(losses).any() or np.isinf(losses).any():
                reasons.append("NaN/Inf in loss distribution")
            if float(np.min(losses)) < 0.0:
                reasons.append("negative losses in long-only loss vector")

        if risk.cvar < risk.var:
            reasons.append("CVaR lower than VaR")

        if risk.loss_stddev <= 0.0 and risk.mean_loss > 0.0:
            reasons.append("positive mean loss with zero dispersion")

        return reasons

    async def _run_llm_overlay(
        self,
        risk: RiskMetrics,
        black_swan_ratio: float,
        hallucination_reasons: list[str],
        default_summary: str,
        default_rationale: str,
        default_actions: list[str],
    ) -> dict[str, str | list[str]] | None:
        if self._llm is None:
            return None

        prompt = (
            "You are the Auditor node in a financial risk pipeline. Return strict JSON only with keys "
            "summary, rationale, recommended_actions. Keep each field concise and operationally clear.\n\n"
            f"risk_metrics={risk.model_dump_json()}\n"
            f"black_swan_ratio={black_swan_ratio:.6f}\n"
            f"hallucination_reasons={json.dumps(hallucination_reasons)}\n"
            f"default_summary={default_summary}\n"
            f"default_rationale={default_rationale}\n"
            f"default_actions={json.dumps(default_actions)}\n"
        )

        try:
            response = await self._llm.ainvoke(prompt)
            response_text = str(response.content).strip()
            parsed = json.loads(self._extract_json(response_text))
            summary = str(parsed.get("summary", default_summary)).strip() or default_summary
            rationale = str(parsed.get("rationale", default_rationale)).strip() or default_rationale

            actions_raw = parsed.get("recommended_actions", default_actions)
            if isinstance(actions_raw, list):
                actions = [str(action).strip() for action in actions_raw if str(action).strip()]
            else:
                actions = default_actions
            if not actions:
                actions = default_actions

            return {
                "summary": summary,
                "rationale": rationale,
                "recommended_actions": actions,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM overlay parse failed (%s); deterministic fallback retained", exc)
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
