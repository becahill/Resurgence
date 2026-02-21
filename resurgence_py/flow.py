from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from resurgence_py.auditor import LLMAuditor
from resurgence_py.engine import Engine
from resurgence_py.inquisitor import Inquisitor
from resurgence_py.models import AuditReport, CrashVolProfile, FlowInput, FlowOutput, PortfolioSeries, RiskMetrics

logger = logging.getLogger(__name__)


class ResurgenceState(TypedDict, total=False):
    request: FlowInput
    portfolio: PortfolioSeries
    crash_profile: CrashVolProfile
    risk: RiskMetrics
    audit: AuditReport
    db_path: str


class ResurgenceFlow:
    """LangGraph DAG coordinating data ingestion, simulation, and risk audit."""

    def __init__(self, db_path: str = "resurgence.duckdb") -> None:
        self.db_path = db_path
        self.inquisitor = Inquisitor(db_path=db_path)
        self.engine = Engine()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResurgenceState)
        graph.add_node("inquisitor", self._inquisitor_node)
        graph.add_node("engine", self._engine_node)
        graph.add_node("auditor", self._auditor_node)

        graph.add_edge(START, "inquisitor")
        graph.add_edge("inquisitor", "engine")
        graph.add_edge("engine", "auditor")
        graph.add_edge("auditor", END)

        return graph.compile()

    async def _inquisitor_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        portfolio, crash_profile = await self.inquisitor.run(payload)
        logger.info(
            "Node Inquisitor complete: n_obs=%d vol=%.4f",
            portfolio.observation_count,
            portfolio.annualized_volatility,
        )
        return {"portfolio": portfolio, "crash_profile": crash_profile}

    async def _engine_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        portfolio = state["portfolio"]
        risk = await self.engine.run(payload, portfolio)

        logger.info(
            "Node Engine complete: VaR=%.6f CVaR=%.6f",
            risk.var,
            risk.cvar,
        )
        return {"risk": risk}

    async def _auditor_node(self, state: ResurgenceState) -> ResurgenceState:
        payload = state["request"]
        portfolio = state["portfolio"]
        risk = state["risk"]
        crash_profile = state["crash_profile"]

        auditor = LLMAuditor(model_name=payload.llm_model)
        audit = await auditor.run(
            risk=risk,
            portfolio=portfolio,
            crash_profile=crash_profile,
        )

        logger.info(
            "Node Auditor complete: severity=%s flagged=%s",
            audit.severity,
            audit.flagged,
        )
        return {"audit": audit}

    async def run(self, request: FlowInput) -> FlowOutput:
        logger.info("ResurgenceFlow started for tickers=%s", request.tickers)
        state_in: ResurgenceState = {
            "request": request,
            "db_path": self.db_path,
        }
        state_out = await self._graph.ainvoke(state_in)

        return FlowOutput(
            request=request,
            portfolio=state_out["portfolio"],
            risk=state_out["risk"],
            audit=state_out["audit"],
            db_path=self.db_path,
        )
