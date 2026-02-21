from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def default_transition_matrix() -> list[list[float]]:
    return [
        [0.92, 0.04, 0.04],
        [0.10, 0.80, 0.10],
        [0.18, 0.14, 0.68],
    ]


def default_state_vol_multipliers() -> list[float]:
    return [0.70, 1.85, 1.00]


def default_state_drift_adjustments() -> list[float]:
    return [0.0003, -0.0008, 0.0]


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
    inquisitor_timeout_s: float = Field(default=20.0, ge=1.0, le=120.0)
    max_audit_reruns: int = Field(default=2, ge=0, le=5)
    target_latency_ms: float = Field(default=200.0, ge=50.0, le=5_000.0)
    hmm_transition_matrix: list[list[float]] = Field(default_factory=default_transition_matrix)
    state_vol_multipliers: list[float] = Field(default_factory=default_state_vol_multipliers)
    state_drift_adjustments: list[float] = Field(default_factory=default_state_drift_adjustments)
    initial_state: int = Field(default=2, ge=0, le=2)

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized: list[str] = []
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

    @field_validator("hmm_transition_matrix")
    @classmethod
    def validate_transition_matrix(cls, matrix: list[list[float]]) -> list[list[float]]:
        if len(matrix) != 3:
            msg = "hmm_transition_matrix must be 3x3"
            raise ValueError(msg)

        for row in matrix:
            if len(row) != 3:
                msg = "hmm_transition_matrix must be 3x3"
                raise ValueError(msg)
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 1e-6:
                msg = "Each transition matrix row must sum to 1.0"
                raise ValueError(msg)
            if any(prob < 0.0 for prob in row):
                msg = "Transition probabilities must be non-negative"
                raise ValueError(msg)

        return matrix

    @field_validator("state_vol_multipliers")
    @classmethod
    def validate_state_vol_multipliers(cls, values: list[float]) -> list[float]:
        if len(values) != 3:
            msg = "state_vol_multipliers must contain 3 values"
            raise ValueError(msg)
        if any(value <= 0.0 for value in values):
            msg = "state_vol_multipliers must be positive"
            raise ValueError(msg)
        return values

    @field_validator("state_drift_adjustments")
    @classmethod
    def validate_state_drift_adjustments(cls, values: list[float]) -> list[float]:
        if len(values) != 3:
            msg = "state_drift_adjustments must contain 3 values"
            raise ValueError(msg)
        return values


class DataPullConfig(BaseModel):
    """Inquisitor pull parameters that may be adjusted by the auditor."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date
    request_timeout_s: float = Field(default=20.0, ge=1.0, le=120.0)

    @field_validator("end_date")
    @classmethod
    def validate_pull_dates(cls, end_date: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start is not None and start > end_date:
            msg = "start_date must be on or before end_date"
            raise ValueError(msg)
        return end_date

    def with_lookback_extension(self, extension_days: int, timeout_s: float | None) -> DataPullConfig:
        start = self.start_date - timedelta(days=max(extension_days, 0))
        timeout = self.request_timeout_s if timeout_s is None else timeout_s
        return DataPullConfig(
            start_date=start,
            end_date=self.end_date,
            request_timeout_s=timeout,
        )


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
    transition_matrix: list[list[float]]
    state_vol_multipliers: list[float]
    state_drift_adjustments: list[float]
    initial_state: int = Field(ge=0, le=2)
    seed: int | None = Field(default=42, ge=0)
    rayon_threads: int | None = Field(default=None, ge=1, le=256)


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
    sharpe_ratio: float = 0.0
    max_loss_zscore: float = 0.0
    state_occupancy: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    regime_transition_matrix: list[list[float]] = Field(default_factory=default_transition_matrix)
    state_vol_multipliers: list[float] = Field(default_factory=default_state_vol_multipliers)


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
    hallucination_detected: bool = False
    requires_rerun: bool = False
    suggested_lookback_extension_days: int = Field(default=0, ge=0)
    suggested_timeout_s: float | None = Field(default=None, ge=1.0, le=120.0)
    detected_sharpe: float = 0.0
    detected_zscore: float = 0.0
    recommended_actions: list[str] = Field(default_factory=list)


class MetaOptimizationReport(BaseModel):
    """Latency-aware control plane output for thread tuning."""

    model_config = ConfigDict(extra="forbid")

    observed_latency_ms: float = Field(ge=0)
    threshold_ms: float = Field(ge=0)
    exceeds_threshold: bool
    suggested_rayon_threads: int | None = Field(default=None, ge=1, le=256)
    rationale: str = Field(min_length=1)


class FlowOutput(BaseModel):
    """Final output contract from ResurgenceFlow."""

    model_config = ConfigDict(extra="forbid")

    request: FlowInput
    pull_config: DataPullConfig
    portfolio: PortfolioSeries
    risk: RiskMetrics
    audit: AuditReport
    meta: MetaOptimizationReport
    rerun_count: int = Field(ge=0)
    db_path: str = Field(min_length=1)
