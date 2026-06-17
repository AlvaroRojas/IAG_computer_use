"""LangGraph state. Deliberately holds only JSON-serializable values (dicts /
lists / strings) so the checkpointer can persist it. Browser handles and the
OpenAI client live in Resources (closures), never in state.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    trades: list[dict]
    results: Annotated[list[dict], operator.add]
    summary: dict


class WorkerPayload(TypedDict):
    trade: dict
    env: str
