"""uvicorn entrypoint for the run API — the `iag-sim-api` console script.

Run with:  iag-sim-api            (after `pip install -e .`)
       or:  python -m iag_sim.server

Required env: IAG_SIM_API_KEY plus the provider credentials the engine needs
(CUA_PROVIDER + its key). Optional: API_HOST, API_PORT, OUTPUT_DIR.

These four are read straight from `os.environ` (they are NOT `Settings` fields —
see `api/security.py`), and pydantic-settings' `env_file` only populates a
`Settings` object, never the process environment. So this entrypoint loads `.env`
itself; without it `python -m iag_sim.server` ignores a perfectly good `.env`.
Real env vars win (`override=False`), and the load happens here rather than in
`create_app` so tests keep their isolated environment.
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

from .api.app import create_app


def main() -> None:
    load_dotenv()
    app = create_app()
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
