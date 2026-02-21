from __future__ import annotations

import logging
import os
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from resurgence_py.auditor import LLMAuditor
from resurgence_py.engine import Engine
from resurgence_py.inquisitor import Inquisitor
from resurgence_py.models import (
    AuditReport,
    CrashVolProfile,
    DataPullConfig,
    FlowInput,
    FlowOutput,
    MetaOptimizationReport,
    PortfolioSeries,
    RiskMetrics,
)

logger = logging.getLogger(__name__)


class ResurgenceState(TypedDict, total=False):
    request: FlowInput
    pull_config: DataPullConfig
    rerun_count: int
    rayon_threads: int | None
    portfolio: PortfolioSeries
    crash_profile: CrashVolProfile
    risk: RiskMetrics
    losses: np.ndarray
    engine_latency_ms: float
    audit: AuditReport
    meta: MetaOptimizationReport
    db_path: str


class ResurgenceFlow:
    """LangGraph DAG coordinating ingestion, simulation, optimization, and recursive audit."""

    def __init__(self, db_path: str = "resurgence.duckdb") -> None:
        self.db_path = db_path
        self.inquisitor = Inquisitor(db_path=db_path)
        self.engine = Engine()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResurgenceState)
        graph.add_node("inquisitor", self._inquisitor_node)
        graph.add_node("engine", self._engine_node)
        graph.add_node("meta_optimizer", self._meta_optimizer_node)
        graph.add_node("auditor", self._auditor_node)

        graph.add_edge(START, "inquisitor")
        graph.add_edge("inquisitor", "engine")
        graph.add_edge("engine", "meta_optimizer")
        graph.add_edge("meta_optimizer", "auditor")

        graph.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {
                "rerun": "inquisitor",
                "finish": END,
            },
        )

        return graph.compile()

    async def _inquisitor_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        pull_config = state["pull_config"]
        portfolio, crash_profile = await self.inquisitor.run(payload, pull_config)

        logger.info(
            "Node Inquisitor complete: n_obs=%d vol=%.4f window=[%s,%s] timeout=%.1fs rerun_count=%d",
            portfolio.observation_count,
            portfolio.annualized_volatility,
            pull_config.start_date.isoformat(),
            pull_config.end_date.isoformat(),
            pull_config.request_timeout_s,
            state.get("rerun_count", 0),
        )
        return {
            "portfolio": portfolio,
            "crash_profile": crash_profile,
        }

    async def _engine_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        portfolio = state["portfolio"]
        artifacts = await self.engine.run(
            payload=payload,
            portfolio=portfolio,
            rayon_threads=state.get("rayon_threads"),
        )

        logger.info(
            "Node Engine complete: VaR=%.6f CVaR=%.6f latency=%.2fms threads=%s",
            artifacts.risk.var,
            artifacts.risk.cvar,
            artifacts.latency_ms,
            state.get("rayon_threads"),
        )
        return {
            "risk": artifacts.risk,
            "losses": artifacts.losses,
            "engine_latency_ms": artifacts.latency_ms,
        }

    async def _meta_optimizer_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        latency_ms = state["engine_latency_ms"]
        threshold_ms = payload.target_latency_ms
        cpu_count = max(os.cpu_count() or 4, 1)

        current_threads = state.get("rayon_threads")
        exceeds_threshold = latency_ms > threshold_ms

        suggested_threads: int | None = current_threads
        if exceeds_threshold:
            baseline_threads = current_threads if current_threads is not None else max(2, cpu_count // 2)
            suggested_threads = min(baseline_threads + 2, cpu_count)
            rationale = (
                f"Latency {latency_ms:.2f}ms exceeded {threshold_ms:.2f}ms; "
                f"recommend increasing rayon threads to {suggested_threads}."
            )
        else:
            rationale = (
                f"Latency {latency_ms:.2f}ms is within threshold {threshold_ms:.2f}ms; "
                "no thread-pool adjustment required."
            )

        report = MetaOptimizationReport(
            observed_latency_ms=latency_ms,
            threshold_ms=threshold_ms,
            exceeds_threshold=exceeds_threshold,
            suggested_rayon_threads=suggested_threads,
            rationale=rationale,
        )

        logger.info(
            "Node MetaOptimizer complete: exceeds=%s suggested_threads=%s",
            report.exceeds_threshold,
            report.suggested_rayon_threads,
        )

        return {
            "meta": report,
            "rayon_threads": suggested_threads,
        }

    async def _auditor_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        portfolio = state["portfolio"]
        risk = state["risk"]
        crash_profile = state["crash_profile"]
        losses = state["losses"]
        pull_config = state["pull_config"]
        rerun_count = state["rerun_count"]

        auditor = LLMAuditor(model_name=payload.llm_model)
        audit = await auditor.run(
            risk=risk,
            portfolio=portfolio,
            crash_profile=crash_profile,
            losses=losses,
            pull_config=pull_config,
            rerun_count=rerun_count,
            max_reruns=payload.max_audit_reruns,
        )

        logger.info(
            "Node Auditor complete: severity=%s flagged=%s hallucination=%s rerun=%s",
            audit.severity,
            audit.flagged,
            audit.hallucination_detected,
            audit.requires_rerun,
        )

        updates: ResurgenceState = {"audit": audit}
        if audit.requires_rerun:
            updates["pull_config"] = pull_config.with_lookback_extension(
                extension_days=audit.suggested_lookback_extension_days,
                timeout_s=audit.suggested_timeout_s,
            )
            updates["rerun_count"] = rerun_count + 1

            logger.warning(
                "Auditor triggered recursive rerun: new_start=%s timeout=%.1fs rerun_count=%d",
                updates["pull_config"].start_date.isoformat(),
                updates["pull_config"].request_timeout_s,
                updates["rerun_count"],
            )

        return updates

    @staticmethod
    def _route_after_audit(state: ResurgenceState) -> str:
        audit = state["audit"]
        return "rerun" if audit.requires_rerun else "finish"

    async def run(self, request: FlowInput) -> FlowOutput:
        logger.info("ResurgenceFlow started for tickers=%s", request.tickers)
        initial_pull = DataPullConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            request_timeout_s=request.inquisitor_timeout_s,
        )
        state_in: ResurgenceState = {
            "request": request,
            "pull_config": initial_pull,
            "rerun_count": 0,
            "rayon_threads": None,
            "db_path": self.db_path,
        }

        state_out = await self._graph.ainvoke(state_in)

        return FlowOutput(
            request=request,
            pull_config=state_out["pull_config"],
            portfolio=state_out["portfolio"],
            risk=state_out["risk"],
            audit=state_out["audit"],
            meta=state_out["meta"],
            rerun_count=state_out["rerun_count"],
            db_path=self.db_path,
        )
