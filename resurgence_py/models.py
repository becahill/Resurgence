from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class FlowInput(BaseModel):
    """Entry payload for ResurgenceFlow."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(min_length=1)
    start_date: date
    end_date: date
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    simulations: int = Field(default=10_000, ge=1_000, le=1_000_000)
    horizon_days: int = Field(default=10, ge=1, le=365)
    seed: int | None = Field(default=42, ge=0)
    llm_model: str = Field(default="gpt-4o-mini")

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = []
        seen: set[str] = set()

        for ticker in tickers:
            clean = ticker.strip().upper()
            if not clean:
                continue
            if clean not in seen:
                seen.add(clean)
                normalized.append(clean)

        if not normalized:
            msg = "At least one non-empty ticker is required"
            raise ValueError(msg)

        return normalized

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, end_date: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start is not None and start > end_date:
            msg = "start_date must be on or before end_date"
            raise ValueError(msg)
        return end_date


class MarketDataRecord(BaseModel):
    """Normalized daily OHLCV row."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    date: date
    close: float = Field(gt=0)
    adj_close: float = Field(gt=0)
    volume: int = Field(ge=0)
    daily_return: float | None = None


class PortfolioSeries(BaseModel):
    """Portfolio-level return series statistics from DuckDB."""

    model_config = ConfigDict(extra="forbid")

    returns: list[float] = Field(min_length=2)
    annualized_volatility: float = Field(ge=0)
    observation_count: int = Field(ge=2)


class CrashVolProfile(BaseModel):
    """Historical stress-volatility anchors used by the auditor."""

    model_config = ConfigDict(extra="forbid")

    volatility_2008: float = Field(ge=0)
    volatility_2020: float = Field(ge=0)
    sample_size_2008: int = Field(ge=0)
    sample_size_2020: int = Field(ge=0)


class EngineInput(BaseModel):
    """Validated payload crossing Python -> Rust boundary."""

    model_config = ConfigDict(extra="forbid")

    returns: list[float] = Field(min_length=2)
    confidence_level: float = Field(gt=0.5, lt=0.999)
    simulations: int = Field(ge=1_000, le=1_000_000)
    horizon_days: int = Field(ge=1, le=365)
    seed: int | None = Field(default=42, ge=0)


class RiskMetrics(BaseModel):
    """Risk metrics output from Monte Carlo engine."""

    model_config = ConfigDict(extra="forbid")

    var: float = Field(ge=0)
    cvar: float = Field(ge=0)
    mean_loss: float = Field(ge=0)
    loss_stddev: float = Field(ge=0)
    confidence_level: float = Field(gt=0.5, lt=0.999)
    simulations: int = Field(ge=1)
    horizon_days: int = Field(ge=1)
    estimated_drift: float
    estimated_volatility: float = Field(ge=0)


class AuditSeverity(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditReport(BaseModel):
    """LLM + heuristic anomaly review for risk outputs."""

    model_config = ConfigDict(extra="forbid")

    severity: AuditSeverity
    flagged: bool
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    black_swan_ratio: float = Field(ge=0)
    recommended_actions: list[str] = Field(default_factory=list)


class FlowOutput(BaseModel):
    """Final output contract from ResurgenceFlow."""

    model_config = ConfigDict(extra="forbid")

    request: FlowInput
    portfolio: PortfolioSeries
    risk: RiskMetrics
    audit: AuditReport
    db_path: str = Field(min_length=1)
