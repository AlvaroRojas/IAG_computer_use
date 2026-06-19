"""Two-phase simulate_trade flow: export (phase 1) then a separate best-effort
log-off (phase 2) that runs AFTER the reality gate, in every session-established
path, before the session is closed. No browser/Docker/network."""

from __future__ import annotations

from pathlib import Path

from iag_sim.config import Settings
from iag_sim.cua.loop import LoopResult
from iag_sim.models import EnvName, TradeTask
from iag_sim.murex.simulate import _LOGOFF_TASK, simulate_trade

REQUIRED = {
    "OPENAI_API_KEY": "sk-test",
    "MUREX_BEFORE_URL": "https://before",
    "MUREX_AFTER_URL": "https://after",
    "MUREX_USER": "u",
    "MUREX_PASS": "p",
}


def _settings(monkeypatch, **extra):
    for k, v in {**REQUIRED, **extra}.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


class _FakeSession:
    def __init__(self, export: Path | None):
        self.computer = object()
        self.display = (1024, 768)
        self._export = export
        self.closed = False

    async def collect_export(self, timeout: float = 0.0) -> Path | None:
        return self._export

    async def close(self) -> None:
        self.closed = True


class _FakeHarness:
    def __init__(self, env: EnvName, session: _FakeSession):
        self.env = env
        self._session = session

    async def new_session(self, trade: TradeTask) -> _FakeSession:
        return self._session


class _FakeBackend:
    """Records the task of every run() call. `fail_logoff` makes the SECOND
    (log-off) call raise, to prove a failed log-off can't change the result."""

    def __init__(self, fail_logoff: bool = False):
        self.tasks: list[str] = []
        self._fail_logoff = fail_logoff

    async def run(self, *, computer, task, display_width, display_height,
                  environment="browser", max_turns=60, on_safety_check=None,
                  tracer=None) -> LoopResult:
        self.tasks.append(task)
        if self._fail_logoff and task == _LOGOFF_TASK:
            raise RuntimeError("logoff window crashed")
        return LoopResult(final_text="DONE", turns=3, completed=True)


def _write_csv(path: Path, *, rows: bool) -> Path:
    body = "Trade nb;Amount\n"
    if rows:
        body += "594;100.00\n"
    path.write_text(body, encoding="utf-8")
    return path


async def _run(*, monkeypatch, tmp_path, export, fail_logoff=False, **settings_extra):
    settings = _settings(monkeypatch, **settings_extra)
    session = _FakeSession(export)
    harness = _FakeHarness(EnvName.BEFORE, session)
    backend = _FakeBackend(fail_logoff=fail_logoff)
    result = await simulate_trade(
        harness=harness, trade=TradeTask(trade_id="594"),
        settings=settings, backend=backend, run_dir=tmp_path,
    )
    return result, backend, session


async def test_logoff_runs_after_valid_export(monkeypatch, tmp_path):
    csv = _write_csv(tmp_path / "raw.csv", rows=True)
    result, backend, session = await _run(monkeypatch=monkeypatch, tmp_path=tmp_path, export=csv)
    assert result.ok
    # Two phases: export task, then the dedicated log-off task — in that order.
    assert len(backend.tasks) == 2
    assert _LOGOFF_TASK not in backend.tasks[0]
    assert backend.tasks[1] == _LOGOFF_TASK
    assert session.closed


async def test_logoff_runs_after_failed_validation(monkeypatch, tmp_path):
    # Wrong trade id in the export -> reality gate fails -> ok=False, but the
    # session must still be logged off before the retry tears it down.
    bad = (tmp_path / "raw.csv")
    bad.write_text("Trade nb;Amount\n999;100.00\n", encoding="utf-8")
    result, backend, session = await _run(monkeypatch=monkeypatch, tmp_path=tmp_path, export=bad)
    assert not result.ok
    assert backend.tasks[-1] == _LOGOFF_TASK
    assert session.closed


async def test_logoff_runs_after_no_export(monkeypatch, tmp_path):
    result, backend, session = await _run(monkeypatch=monkeypatch, tmp_path=tmp_path, export=None)
    assert not result.ok
    assert result.error == "no CSV was exported"
    assert backend.tasks[-1] == _LOGOFF_TASK
    assert session.closed


async def test_logoff_skipped_when_disabled(monkeypatch, tmp_path):
    csv = _write_csv(tmp_path / "raw.csv", rows=True)
    result, backend, session = await _run(
        monkeypatch=monkeypatch, tmp_path=tmp_path, export=csv, LOGOFF_MAX_TURNS="0"
    )
    assert result.ok
    assert len(backend.tasks) == 1  # only the export phase ran
    assert _LOGOFF_TASK not in backend.tasks
    assert session.closed


async def test_failed_logoff_does_not_change_result(monkeypatch, tmp_path):
    csv = _write_csv(tmp_path / "raw.csv", rows=True)
    result, backend, session = await _run(
        monkeypatch=monkeypatch, tmp_path=tmp_path, export=csv, fail_logoff=True
    )
    assert result.ok  # export was valid; log-off blowing up is swallowed
    assert backend.tasks[-1] == _LOGOFF_TASK
    assert session.closed
