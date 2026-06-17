"""Test isolation.

`Settings` reads `os.environ` directly (pydantic-settings), so a developer shell
that exports the app's vars (e.g. `CUA_PROVIDER=bedrock`, `MUREX_CHANNEL=thick`,
`MUREX_LOGIN_GROUP=...`, `EXPORT_*`) would leak into every `Settings(...)` built in a
test and break otherwise-deterministic assertions. This autouse fixture strips every
env var the model binds — keyed by each field's alias — before each test, so a test
sees only the values it sets explicitly. `monkeypatch` restores the real environment
afterwards.
"""

from __future__ import annotations

import pytest

from iag_sim.config import Settings


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch):
    for field in Settings.model_fields.values():
        if field.alias:
            monkeypatch.delenv(field.alias, raising=False)
    yield
