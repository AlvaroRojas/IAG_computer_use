"""Dependency-injection wiring.

The `RunManager` is created once in the app lifespan and stored on `app.state`, so
routes resolve it via this dependency (rather than a module-level global) — which
keeps it injectable/replaceable in tests.
"""

from __future__ import annotations

from fastapi import Request

from .run_manager import RunManager


def get_run_manager(request: Request) -> RunManager:
    return request.app.state.run_manager
