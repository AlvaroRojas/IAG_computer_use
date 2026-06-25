"""FastAPI application factory.

`create_app` wires the routers and creates the single `RunManager` (stored on
`app.state`) inside the lifespan. `output_dir` is injectable for tests; in
production it comes from `OUTPUT_DIR` (default `data/out`). The lifespan fails fast
if `IAG_SIM_API_KEY` is unset so an unauthenticated server never starts.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .routes.health import router as health_router
from .routes.runs import router as runs_router
from .run_manager import RunManager
from .security import API_KEY_ENV

log = logging.getLogger("iag_sim.api")


def create_app(output_dir: Path | None = None) -> FastAPI:
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(os.environ.get("OUTPUT_DIR", "data/out"))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(
                f"{API_KEY_ENV} env var must be set before serving the API"
            )
        app.state.run_manager = RunManager(output_dir=resolved_output_dir)
        log.info(f"iag-sim API ready — output_dir={resolved_output_dir}")
        yield

    app = FastAPI(
        title="IAG Accounting Simulation API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(runs_router)
    app.include_router(health_router)
    return app
