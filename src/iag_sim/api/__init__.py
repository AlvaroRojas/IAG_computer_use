"""HTTP API surface for the before/after comparator.

A thin FastAPI layer over the LangGraph engine: a caller POSTs Murex connection
config + a trade list, the server runs `run_graph_async` in the background and
hands back a run id to poll. The engine, orchestration and config are untouched —
the API only builds a per-request `Settings` and threads it through the existing
`run_graph_async(trades, settings, run_dir, *, resume)` entry point.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
