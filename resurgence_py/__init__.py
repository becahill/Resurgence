"""Resurgence AI Python orchestration package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resurgence_py.flow import ResurgenceFlow
    from resurgence_py.models import FlowInput, FlowOutput

__all__ = ["ResurgenceFlow", "FlowInput", "FlowOutput"]


def __getattr__(name: str) -> Any:
    if name == "ResurgenceFlow":
        from resurgence_py.flow import ResurgenceFlow as _ResurgenceFlow

        return _ResurgenceFlow
    if name == "FlowInput":
        from resurgence_py.models import FlowInput as _FlowInput

        return _FlowInput
    if name == "FlowOutput":
        from resurgence_py.models import FlowOutput as _FlowOutput

        return _FlowOutput
    msg = f"module 'resurgence_py' has no attribute '{name}'"
    raise AttributeError(msg)
