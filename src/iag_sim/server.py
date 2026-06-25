"""uvicorn entrypoint for the run API — the `iag-sim-api` console script.

Run with:  iag-sim-api            (after `pip install -e .`)
       or:  python -m iag_sim.server

Required env: IAG_SIM_API_KEY plus the provider credentials the engine needs
(CUA_PROVIDER + its key). Optional: API_HOST, API_PORT, OUTPUT_DIR.
"""

from __future__ import annotations

import os

import uvicorn

from .api.app import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
