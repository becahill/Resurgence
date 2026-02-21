from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from resurgence_py.models import FlowInput


def test_flow_input_normalizes_tickers() -> None:
    payload = FlowInput(
        tickers=[" spy ", "QQQ", "spy"],
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )
    assert payload.tickers == ["SPY", "QQQ"]


def test_flow_input_rejects_invalid_date_range() -> None:
    with pytest.raises(ValidationError):
        FlowInput(
            tickers=["SPY"],
            start_date=date(2021, 1, 1),
            end_date=date(2020, 1, 1),
        )
