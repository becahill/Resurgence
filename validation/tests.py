from __future__ import annotations

from collections.abc import Sequence
from math import erfc, log, sqrt

from pydantic import BaseModel, ConfigDict, Field


class KupiecPOFResult(BaseModel):
    """Result payload for the Kupiec Proportion of Failures test."""

    model_config = ConfigDict(extra="forbid")

    confidence_level: float = Field(gt=0.5, lt=0.999)
    expected_breach_rate: float = Field(ge=0.0, le=1.0)
    observed_breach_rate: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=1)
    breaches: int = Field(ge=0)
    lr_stat: float = Field(ge=0.0)
    p_value: float = Field(ge=0.0, le=1.0)
    reject_at_95: bool


class ChristoffersenIndependenceResult(BaseModel):
    """Result payload for the Christoffersen independence test."""

    model_config = ConfigDict(extra="forbid")

    observations: int = Field(ge=2)
    n00: int = Field(ge=0)
    n01: int = Field(ge=0)
    n10: int = Field(ge=0)
    n11: int = Field(ge=0)
    applicable: bool
    lr_stat: float | None = Field(default=None, ge=0.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    reject_at_95: bool | None = None


def _clamp_probability(value: float, epsilon: float = 1e-12) -> float:
    return min(max(value, epsilon), 1.0 - epsilon)


def _bernoulli_log_likelihood(successes: int, total: int, probability: float) -> float:
    p = _clamp_probability(probability)
    failures = total - successes
    return (successes * log(p)) + (failures * log(1.0 - p))


def _chi_square_df1_p_value(lr_stat: float) -> float:
    """Survival function for chi-square with 1 degree of freedom."""
    adjusted = max(lr_stat, 0.0)
    return erfc(sqrt(adjusted / 2.0))


def kupiec_proportion_of_failures(
    breaches: Sequence[bool],
    confidence_level: float,
) -> KupiecPOFResult:
    """Compute unconditional coverage via the Kupiec POF likelihood-ratio test."""
    if not 0.5 < confidence_level < 0.999:
        msg = "confidence_level must be between 0.5 and 0.999"
        raise ValueError(msg)

    observations = len(breaches)
    if observations < 1:
        msg = "At least one breach observation is required"
        raise ValueError(msg)

    x = sum(1 for breach in breaches if breach)
    p = 1.0 - confidence_level
    pi = x / observations

    log_likelihood_h0 = _bernoulli_log_likelihood(x, observations, p)
    log_likelihood_h1 = _bernoulli_log_likelihood(x, observations, pi)
    lr_stat = max(0.0, -2.0 * (log_likelihood_h0 - log_likelihood_h1))
    p_value = _chi_square_df1_p_value(lr_stat)

    return KupiecPOFResult(
        confidence_level=confidence_level,
        expected_breach_rate=p,
        observed_breach_rate=pi,
        observations=observations,
        breaches=x,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_at_95=p_value < 0.05,
    )


def christoffersen_independence_test(breaches: Sequence[bool]) -> ChristoffersenIndependenceResult:
    """Test whether VaR breaches are independently distributed over time."""
    observations = len(breaches)
    if observations < 2:
        msg = "At least two breach observations are required"
        raise ValueError(msg)

    encoded = [1 if value else 0 for value in breaches]
    n00 = n01 = n10 = n11 = 0

    for previous, current in zip(encoded, encoded[1:], strict=False):
        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        else:
            n11 += 1

    transitions_total = n00 + n01 + n10 + n11
    if transitions_total == 0:
        return ChristoffersenIndependenceResult(
            observations=observations,
            n00=n00,
            n01=n01,
            n10=n10,
            n11=n11,
            applicable=False,
        )

    pi = (n01 + n11) / transitions_total

    row0_total = n00 + n01
    row1_total = n10 + n11

    pi01 = 0.0 if row0_total == 0 else n01 / row0_total
    pi11 = 0.0 if row1_total == 0 else n11 / row1_total

    ll_null = _bernoulli_log_likelihood(n01 + n11, transitions_total, pi)
    ll_alt = _bernoulli_log_likelihood(n01, row0_total, pi01) + _bernoulli_log_likelihood(
        n11,
        row1_total,
        pi11,
    )

    lr_stat = max(0.0, -2.0 * (ll_null - ll_alt))
    p_value = _chi_square_df1_p_value(lr_stat)

    return ChristoffersenIndependenceResult(
        observations=observations,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        applicable=True,
        lr_stat=lr_stat,
        p_value=p_value,
        reject_at_95=p_value < 0.05,
    )
