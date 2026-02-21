from __future__ import annotations

import json
import logging
import os

import numpy as np

from resurgence_py.models import (
    AuditAnomaly,
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

        anomalies = self.evaluate_deterministic_anomalies(risk=risk, losses=losses)
        critical_anomalies = [anomaly for anomaly in anomalies if anomaly.severity == AuditSeverity.CRITICAL]
        warning_anomalies = [anomaly for anomaly in anomalies if anomaly.severity == AuditSeverity.WARNING]
        hallucination_detected = len(critical_anomalies) > 0

        if hallucination_detected:
            severity = AuditSeverity.CRITICAL
            flagged = True
            summary = "Quantitative hallucination detected in simulation output"
        elif black_swan_ratio >= 1.25 or tail_ratio >= 2.0:
            severity = AuditSeverity.CRITICAL
            flagged = True
            summary = "Black Swan alert: tail risk exceeds stress anchors"
        elif warning_anomalies or black_swan_ratio >= 0.90 or tail_ratio >= 1.6:
            severity = AuditSeverity.WARNING
            flagged = True
            if warning_anomalies:
                summary = "Deterministic audit warnings detected in simulation output"
            else:
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
            f"sharpe={risk.sharpe_ratio:.4f}, max_loss_zscore={risk.max_loss_zscore:.4f}, "
            f"anomalies={len(anomalies)}."
        )
        if anomalies:
            anomaly_descriptions = "; ".join(
                f"{anomaly.rule_id}: {anomaly.message}" for anomaly in anomalies[:4]
            )
            if len(anomalies) > 4:
                anomaly_descriptions = f"{anomaly_descriptions}; +{len(anomalies) - 4} more"
            rationale = f"{rationale} Deterministic triggers: {anomaly_descriptions}"

        recommended_actions: list[str] = []
        if flagged:
            recommended_actions.extend(
                [
                    "Increase hedge ratio on highest-beta holdings",
                    "Run intraday re-pricing with reduced liquidity assumptions",
                    "Escalate to risk committee with scenario replay",
                ]
            )
        if anomalies:
            recommended_actions.insert(0, "Review structured anomaly report for deterministic rule breaches")
        if requires_rerun:
            recommended_actions.insert(0, "Rerun Inquisitor with expanded lookback and elevated timeout")

        llm_overlay = await self._run_llm_overlay(
            risk=risk,
            black_swan_ratio=black_swan_ratio,
            hallucination_reasons=[anomaly.message for anomaly in critical_anomalies],
            deterministic_anomalies=[
                {
                    "rule_id": anomaly.rule_id,
                    "severity": anomaly.severity.value,
                    "message": anomaly.message,
                }
                for anomaly in anomalies
            ],
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
            anomalies=anomalies,
            recommended_actions=recommended_actions,
        )

    @staticmethod
    def evaluate_deterministic_anomalies(risk: RiskMetrics, losses: np.ndarray) -> list[AuditAnomaly]:
        """Run deterministic diagnostics and return structured anomaly payloads."""
        anomalies: list[AuditAnomaly] = []
        loss_values = np.asarray(losses, dtype=np.float64)

        anomalies.extend(LLMAuditor._check_scalar_metric_bounds(risk))
        anomalies.extend(LLMAuditor._check_loss_vector_integrity(loss_values))

        if loss_values.size == 0 or np.isnan(loss_values).any() or np.isinf(loss_values).any():
            return anomalies

        anomalies.extend(LLMAuditor._check_var_monotonicity(risk, loss_values))
        anomalies.extend(LLMAuditor._check_variance_sanity(risk, loss_values))
        anomalies.extend(LLMAuditor._check_distribution_sanity(risk, loss_values))
        anomalies.extend(LLMAuditor._check_outliers(loss_values))

        return anomalies

    @staticmethod
    def _anomaly(
        rule_id: str,
        severity: AuditSeverity,
        message: str,
        observed: float | None = None,
        threshold: float | None = None,
        **context: str | int | float | bool,
    ) -> AuditAnomaly:
        return AuditAnomaly(
            rule_id=rule_id,
            severity=severity,
            message=message,
            observed=observed,
            threshold=threshold,
            context=context,
        )

    @staticmethod
    def _check_scalar_metric_bounds(risk: RiskMetrics) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []

        if not np.isfinite(risk.sharpe_ratio) or abs(risk.sharpe_ratio) > 6.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="metrics.sharpe_bounds",
                    severity=AuditSeverity.CRITICAL,
                    message="Nonsensical Sharpe ratio detected",
                    observed=float(risk.sharpe_ratio),
                    threshold=6.0,
                )
            )

        if not np.isfinite(risk.max_loss_zscore) or abs(risk.max_loss_zscore) > 10.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="metrics.max_loss_zscore_bounds",
                    severity=AuditSeverity.CRITICAL,
                    message="Extreme max-loss z-score detected",
                    observed=float(risk.max_loss_zscore),
                    threshold=10.0,
                )
            )

        if risk.cvar < risk.var:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="metrics.cvar_vs_var",
                    severity=AuditSeverity.CRITICAL,
                    message="CVaR is lower than VaR",
                    observed=float(risk.cvar),
                    threshold=float(risk.var),
                )
            )

        if risk.loss_stddev <= 0.0 and risk.mean_loss > 0.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="metrics.dispersion_consistency",
                    severity=AuditSeverity.CRITICAL,
                    message="Positive mean loss with zero or negative dispersion",
                    observed=float(risk.loss_stddev),
                    threshold=0.0,
                )
            )

        return anomalies

    @staticmethod
    def _check_loss_vector_integrity(losses: np.ndarray) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []

        if losses.size == 0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.non_empty",
                    severity=AuditSeverity.CRITICAL,
                    message="Loss vector is empty",
                )
            )
            return anomalies

        if np.isnan(losses).any() or np.isinf(losses).any():
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.finite_values",
                    severity=AuditSeverity.CRITICAL,
                    message="NaN or Inf detected in loss distribution",
                )
            )

        finite_values = losses[np.isfinite(losses)]
        if finite_values.size > 0 and float(np.min(finite_values)) < 0.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.non_negative_losses",
                    severity=AuditSeverity.CRITICAL,
                    message="Negative losses detected in long-only loss vector",
                    observed=float(np.min(finite_values)),
                    threshold=0.0,
                )
            )

        return anomalies

    @staticmethod
    def _check_var_monotonicity(risk: RiskMetrics, losses: np.ndarray) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []

        confidence_grid = sorted({0.90, float(risk.confidence_level), 0.99})
        var_values = [float(np.quantile(losses, level)) for level in confidence_grid]
        cvar_values: list[float] = []
        for var_value in var_values:
            tail_losses = losses[losses >= var_value]
            cvar_values.append(float(np.mean(tail_losses)) if tail_losses.size > 0 else var_value)

        for idx in range(len(confidence_grid) - 1):
            left_cl = confidence_grid[idx]
            right_cl = confidence_grid[idx + 1]
            if var_values[idx] > var_values[idx + 1] + 1e-12:
                anomalies.append(
                    LLMAuditor._anomaly(
                        rule_id="risk.var_monotonicity",
                        severity=AuditSeverity.CRITICAL,
                        message="VaR is not monotonic across confidence levels",
                        observed=var_values[idx],
                        threshold=var_values[idx + 1],
                        left_confidence=left_cl,
                        right_confidence=right_cl,
                    )
                )

            if cvar_values[idx] > cvar_values[idx + 1] + 1e-12:
                anomalies.append(
                    LLMAuditor._anomaly(
                        rule_id="risk.cvar_monotonicity",
                        severity=AuditSeverity.CRITICAL,
                        message="CVaR is not monotonic across confidence levels",
                        observed=cvar_values[idx],
                        threshold=cvar_values[idx + 1],
                        left_confidence=left_cl,
                        right_confidence=right_cl,
                    )
                )

        target_idx = confidence_grid.index(float(risk.confidence_level))
        empirical_var = var_values[target_idx]
        empirical_cvar = cvar_values[target_idx]

        var_tolerance = max(0.20 * max(empirical_var, 1e-9), 1e-4)
        cvar_tolerance = max(0.20 * max(empirical_cvar, 1e-9), 1e-4)

        if abs(risk.var - empirical_var) > var_tolerance:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="risk.var_consistency",
                    severity=AuditSeverity.WARNING,
                    message="Reported VaR deviates from empirical quantile",
                    observed=float(risk.var),
                    threshold=float(empirical_var),
                    tolerance=float(var_tolerance),
                )
            )

        if abs(risk.cvar - empirical_cvar) > cvar_tolerance:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="risk.cvar_consistency",
                    severity=AuditSeverity.WARNING,
                    message="Reported CVaR deviates from empirical tail mean",
                    observed=float(risk.cvar),
                    threshold=float(empirical_cvar),
                    tolerance=float(cvar_tolerance),
                )
            )

        return anomalies

    @staticmethod
    def _check_variance_sanity(risk: RiskMetrics, losses: np.ndarray) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []

        ddof = 1 if losses.size > 1 else 0
        empirical_var = float(np.var(losses, ddof=ddof))

        if not np.isfinite(empirical_var) or empirical_var < 0.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="variance.non_negative",
                    severity=AuditSeverity.CRITICAL,
                    message="Loss variance is not finite and non-negative",
                    observed=empirical_var,
                    threshold=0.0,
                )
            )
            return anomalies

        reported_var = float(risk.loss_stddev**2)
        variance_tolerance = max(0.35 * max(empirical_var, 1e-9), 1e-6)
        if abs(reported_var - empirical_var) > variance_tolerance:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="variance.consistency",
                    severity=AuditSeverity.WARNING,
                    message="Reported variance is inconsistent with sampled losses",
                    observed=reported_var,
                    threshold=empirical_var,
                    tolerance=float(variance_tolerance),
                )
            )

        return anomalies

    @staticmethod
    def _check_distribution_sanity(risk: RiskMetrics, losses: np.ndarray) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []

        empirical_mean = float(np.mean(losses))
        empirical_std = float(np.std(losses, ddof=1 if losses.size > 1 else 0))

        if risk.mean_loss > 1.0 or empirical_mean > 1.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.mean_bounds",
                    severity=AuditSeverity.WARNING,
                    message="Mean loss exceeds expected long-only bounds",
                    observed=max(float(risk.mean_loss), empirical_mean),
                    threshold=1.0,
                )
            )

        if risk.loss_stddev > 1.0 or empirical_std > 1.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.volatility_bounds",
                    severity=AuditSeverity.WARNING,
                    message="Loss volatility exceeds expected long-only bounds",
                    observed=max(float(risk.loss_stddev), empirical_std),
                    threshold=1.0,
                )
            )

        if risk.estimated_volatility > 2.0:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.estimated_volatility_bounds",
                    severity=AuditSeverity.WARNING,
                    message="Estimated return volatility exceeds sanity threshold",
                    observed=float(risk.estimated_volatility),
                    threshold=2.0,
                )
            )

        mean_tolerance = max(0.25 * max(empirical_mean, 1e-9), 1e-5)
        if abs(risk.mean_loss - empirical_mean) > mean_tolerance:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.mean_consistency",
                    severity=AuditSeverity.WARNING,
                    message="Reported mean loss is inconsistent with sampled losses",
                    observed=float(risk.mean_loss),
                    threshold=empirical_mean,
                    tolerance=float(mean_tolerance),
                )
            )

        std_tolerance = max(0.25 * max(empirical_std, 1e-9), 1e-5)
        if abs(risk.loss_stddev - empirical_std) > std_tolerance:
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.stddev_consistency",
                    severity=AuditSeverity.WARNING,
                    message="Reported loss stddev is inconsistent with sampled losses",
                    observed=float(risk.loss_stddev),
                    threshold=empirical_std,
                    tolerance=float(std_tolerance),
                )
            )

        return anomalies

    @staticmethod
    def _check_outliers(losses: np.ndarray) -> list[AuditAnomaly]:
        anomalies: list[AuditAnomaly] = []
        if losses.size < 5:
            return anomalies

        median = float(np.median(losses))
        abs_deviation = np.abs(losses - median)
        mad = float(np.median(abs_deviation))

        if mad <= 1e-12:
            spread = float(np.max(losses) - np.min(losses))
            if spread > 0.25:
                anomalies.append(
                    LLMAuditor._anomaly(
                        rule_id="distribution.outlier_spread",
                        severity=AuditSeverity.WARNING,
                        message="Loss spread indicates potential outlier concentration",
                        observed=spread,
                        threshold=0.25,
                    )
                )
            return anomalies

        robust_z_scores = 0.67448975 * (losses - median) / mad
        max_abs_robust_z = float(np.max(np.abs(robust_z_scores)))

        if max_abs_robust_z > 8.0:
            severity = AuditSeverity.CRITICAL if max_abs_robust_z > 12.0 else AuditSeverity.WARNING
            anomalies.append(
                LLMAuditor._anomaly(
                    rule_id="distribution.mad_outlier",
                    severity=severity,
                    message="Robust z-score outlier detected in loss distribution",
                    observed=max_abs_robust_z,
                    threshold=8.0,
                )
            )

        return anomalies

    async def _run_llm_overlay(
        self,
        risk: RiskMetrics,
        black_swan_ratio: float,
        hallucination_reasons: list[str],
        deterministic_anomalies: list[dict[str, str]],
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
            f"deterministic_anomalies={json.dumps(deterministic_anomalies)}\n"
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
