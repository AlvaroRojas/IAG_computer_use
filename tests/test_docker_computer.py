"""Tests for the thick-client Docker harness command construction. No Docker
required — a fake runner records the argv each method produces."""

from __future__ import annotations

import base64

from pathlib import Path

from iag_sim.config import Settings
from iag_sim.cua.actions import dispatch
from iag_sim.harness.docker import DockerComputer, DockerHarness, DockerSession
from iag_sim.models import EnvName, TradeTask


class FakeRunner:
    def __init__(self, screenshot_png: bytes = b"\x89PNG\r\n\x1a\n") -> None:
        self.calls: list[list[str]] = []
        self._png = screenshot_png

    async def __call__(self, argv):
        self.calls.append(argv)
        # The screenshot path pipes PNG bytes back on stdout.
        if any("import -window root" in a for a in argv):
            return (0, self._png, b"")
        return (0, b"", b"")

    @property
    def last_cmd(self) -> str:
        # docker exec <cid> sh -c "<cmd>"  -> return the command string
        return self.calls[-1][-1]


def _computer(runner) -> DockerComputer:
    return DockerComputer("cid123", ":99", (1280, 800), runner)


async def test_screenshot_pipes_png_and_b64_encodes():
    r = FakeRunner(screenshot_png=b"PNGDATA")
    out = await _computer(r).screenshot()
    assert base64.b64decode(out) == b"PNGDATA"
    assert r.calls[-1][:5] == ["docker", "exec", "cid123", "sh", "-c"]
    assert "import -window root png:-" in r.last_cmd
    assert "DISPLAY=:99" in r.last_cmd


async def test_click_maps_to_xdotool():
    r = FakeRunner()
    await _computer(r).click(100, 200, "left")
    assert "xdotool mousemove 100 200 click 1" in r.last_cmd


async def test_right_click_button_3():
    r = FakeRunner()
    await _computer(r).click(5, 6, "right")
    assert "click 3" in r.last_cmd


async def test_click_with_modifier_keydown_keyup():
    r = FakeRunner()
    await _computer(r).click(1, 2, "left", keys=["CTRL"])
    cmd = r.last_cmd
    assert "keydown ctrl" in cmd and "keyup ctrl" in cmd
    assert cmd.index("keydown ctrl") < cmd.index("click 1") < cmd.index("keyup ctrl")


async def test_keypress_combo():
    r = FakeRunner()
    await _computer(r).keypress(["CTRL", "S"])
    assert "xdotool key --clearmodifiers ctrl+S" in r.last_cmd


async def test_type_is_shell_quoted():
    r = FakeRunner()
    await _computer(r).type("a b; rm -rf /")
    # shlex.quote wraps the dangerous string so it can't break out.
    assert "xdotool type --clearmodifiers -- 'a b; rm -rf /'" in r.last_cmd


async def test_scroll_down_uses_button_5():
    r = FakeRunner()
    await _computer(r).scroll(10, 10, scroll_y=3)
    assert "click --repeat 3 5" in r.last_cmd


async def test_dispatch_drives_docker_computer():
    # The same action dispatcher used by the agent loop works on DockerComputer.
    r = FakeRunner()
    c = _computer(r)
    await dispatch(c, {"type": "type", "text": "hello"})
    assert "xdotool type" in r.last_cmd


def _harness(**env) -> DockerHarness:
    s = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test",
        MUREX_BEFORE_URL="https://before",
        MUREX_AFTER_URL="https://after",
        MUREX_USER="u",
        MUREX_PASS="p",
        MUREX_CHANNEL="thick",
        MUREX_DOCKER_IMAGE="murex-thick:latest",
        **env,
    )
    return DockerHarness(EnvName.BEFORE, s, Path("data/out/run-x"))


def test_run_argv_includes_default_resource_caps():
    argv = _harness()._run_argv(TradeTask(trade_id="594"), Path("/exports/594"))
    # adjacent flag/value pairs, defaults 1 CPU / 512MB
    assert ["--cpus", "1"] == argv[argv.index("--cpus"):argv.index("--cpus") + 2]
    assert ["--memory", "512m"] == argv[argv.index("--memory"):argv.index("--memory") + 2]


def test_run_argv_respects_overridden_caps():
    h = _harness(MUREX_DOCKER_CPUS="2", MUREX_DOCKER_MEMORY="1g")
    argv = h._run_argv(TradeTask(trade_id="594"), Path("/exports/594"))
    assert "2" == argv[argv.index("--cpus") + 1]
    assert "1g" == argv[argv.index("--memory") + 1]


def test_run_argv_omits_caps_when_blank():
    h = _harness(MUREX_DOCKER_CPUS="", MUREX_DOCKER_MEMORY="")
    argv = h._run_argv(TradeTask(trade_id="594"), Path("/exports/594"))
    assert "--cpus" not in argv
    assert "--memory" not in argv


# --- collect_export: bind mount + stray-CSV fallback ---------------------------


async def _noop_stop(_cid):  # pragma: no cover - close() not exercised here
    return None


def _session(host_export: Path, runner=None) -> DockerSession:
    return DockerSession(
        "cid123", computer=None, display=(1280, 800), host_export=host_export,
        stop=_noop_stop, runner=runner, container_export_dir="/exports",
    )


async def test_collect_export_prefers_bind_mount(tmp_path):
    (tmp_path / "accounting_594.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    r = FakeRunner()
    got = await _session(tmp_path, runner=r).collect_export()
    assert got == tmp_path / "accounting_594.csv"
    assert r.calls == []  # no container search needed when the mount has it


async def test_collect_export_recovers_stray_csv(tmp_path):
    stray = "/opt/murex/accounting_594.csv"

    class RecoverRunner:
        def __init__(self):
            self.calls = []

        async def __call__(self, argv):
            self.calls.append(argv)
            if argv[:3] == ["docker", "exec", "cid123"]:
                return (0, f"1700000000.5\t{stray}\n".encode(), b"")
            if argv[:2] == ["docker", "cp"]:
                # emulate docker cp landing the file in the export dir
                Path(argv[3]).write_text("a;b\n1;2\n", encoding="utf-8")
                return (0, b"", b"")
            return (0, b"", b"")

    r = RecoverRunner()
    got = await _session(tmp_path, runner=r).collect_export()
    assert got == tmp_path / "accounting_594.csv"
    assert got.exists()
    # searched then copied from the chooser's default dir
    assert any(a[:3] == ["docker", "exec", "cid123"] for a in r.calls)
    assert any(a[:2] == ["docker", "cp"] and stray in a[2] for a in r.calls)


async def test_collect_export_none_when_nothing_anywhere(tmp_path):
    class EmptyRunner:
        async def __call__(self, argv):
            return (0, b"", b"")  # find prints nothing

    assert await _session(tmp_path, runner=EmptyRunner()).collect_export() is None


async def test_collect_export_none_without_runner(tmp_path):
    # No runner -> can't search the container; just report nothing found.
    assert await _session(tmp_path, runner=None).collect_export() is None


def _stable_session(host_export: Path, poll=0.01, stable_polls=2) -> DockerSession:
    return DockerSession(
        "cid123", computer=None, display=(1280, 800), host_export=host_export,
        stop=_noop_stop, runner=FakeRunner(), container_export_dir="/exports",
        poll=poll, stable_polls=stable_polls,
    )


async def test_collect_export_waits_for_appearance(tmp_path, monkeypatch):
    # File is absent at first; it shows up after the first poll. With timeout the
    # session must wait for it instead of immediately reporting nothing.
    import asyncio

    target = tmp_path / "accounting_594.csv"
    state = {"n": 0}

    async def _sleep(_secs):
        state["n"] += 1
        if state["n"] == 1:
            target.write_text("a;b\n1;2\n", encoding="utf-8")

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    got = await _stable_session(tmp_path, stable_polls=1).collect_export(timeout=5)
    assert got == target


async def test_collect_export_waits_for_size_stability(tmp_path, monkeypatch):
    # File present but only returned once its size is stable for stable_polls polls.
    import asyncio

    (tmp_path / "accounting_594.csv").write_text("a;b\n1;2\n", encoding="utf-8")

    async def _sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    got = await _stable_session(tmp_path, stable_polls=2).collect_export(timeout=5)
    assert got == tmp_path / "accounting_594.csv"


async def test_new_session_clears_stale_csv(tmp_path):
    # A bad CSV from a prior attempt for THIS trade must not survive into the next
    # attempt — host_export is per-trade and reused, and collect_export globs it.
    trade_dir = tmp_path / "before" / "594"
    trade_dir.mkdir(parents=True)
    stale = trade_dir / "old.csv"
    stale.write_text("a;b\n1;2\n", encoding="utf-8")
    h = _thick_harness(tmp_path, RunStopRunner("cidAA"))
    await h.new_session(TradeTask(trade_id="594"))
    assert not stale.exists()


# --- container teardown: process death must not orphan containers -------------


class RunStopRunner:
    """Returns a container id for `docker run`; records every argv."""

    def __init__(self, cid: str = "cidNEW") -> None:
        self.calls: list[list[str]] = []
        self.cid = cid

    async def __call__(self, argv):
        self.calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            return (0, (self.cid + "\n").encode(), b"")
        return (0, b"", b"")

    def stopped(self) -> list[str]:
        return [a[2] for a in self.calls if a[:2] == ["docker", "stop"]]


def _thick_harness(run_dir, runner):
    s = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test",
        MUREX_BEFORE_URL="https://before",
        MUREX_AFTER_URL="https://after",
        MUREX_USER="u",
        MUREX_PASS="p",
        MUREX_CHANNEL="thick",
        MUREX_DOCKER_IMAGE="murex-thick:latest",
        MUREX_CONTAINER_READY_SECS="0",  # no boot wait in tests
        MUREX_LLM_LOGIN="true",
    )
    return DockerHarness(EnvName.BEFORE, s, run_dir, runner=runner)


async def test_new_session_tracks_container(tmp_path):
    r = RunStopRunner("cidAA")
    h = _thick_harness(tmp_path, r)
    await h.new_session(TradeTask(trade_id="594"))
    assert "cidAA" in h._containers


async def test_aclose_stops_container_even_without_session_close(tmp_path):
    # Simulates a kill mid-run: the session was created but close() never ran.
    # aclose() must still stop the container (no orphan).
    from iag_sim.harness import docker as dmod

    r = RunStopRunner("cidAA")
    h = _thick_harness(tmp_path, r)
    await h.new_session(TradeTask(trade_id="594"))
    await h.aclose()
    assert r.stopped() == ["cidAA"]
    assert h._containers == set()
    assert "cidAA" not in dmod._LIVE_CONTAINERS


async def test_session_close_untracks_container(tmp_path):
    from iag_sim.harness import docker as dmod

    r = RunStopRunner("cidBB")
    h = _thick_harness(tmp_path, r)
    sess = await h.new_session(TradeTask(trade_id="594"))
    await sess.close()
    assert "cidBB" in r.stopped()
    assert "cidBB" not in h._containers
    assert "cidBB" not in dmod._LIVE_CONTAINERS


# --- readiness probe: window-class gate + screen-stability -------------------


def _ready_harness(run_dir, runner, **env):
    s = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test",
        MUREX_BEFORE_URL="https://before",
        MUREX_AFTER_URL="https://after",
        MUREX_USER="u",
        MUREX_PASS="p",
        MUREX_CHANNEL="thick",
        MUREX_DOCKER_IMAGE="murex-thick:latest",
        MUREX_LLM_LOGIN="true",
        **env,
    )
    return DockerHarness(EnvName.BEFORE, s, run_dir, runner=runner)


class _RecordingRunner:
    def __init__(self, reply=(0, b"", b"")):
        self.calls: list[list[str]] = []
        self._reply = reply

    async def __call__(self, argv):
        self.calls.append(argv)
        return self._reply

    @property
    def last_cmd(self) -> str:
        return self.calls[-1][-1]


async def test_login_window_present_true_when_xdotool_returns_id(tmp_path):
    r = _RecordingRunner((0, b"  8388612\n", b""))
    h = _ready_harness(tmp_path, r)
    assert await h._login_window_present("cid") is True
    # probes by the env-independent WM_CLASS, not the title
    assert "xdotool search --class murex-rmi-loader" in r.last_cmd
    assert r.calls[-1][:3] == ["docker", "exec", "cid"]


async def test_login_window_present_false_when_no_match(tmp_path):
    # xdotool exits 1 with empty stdout when nothing matches.
    r = _RecordingRunner((1, b"", b""))
    assert await _ready_harness(tmp_path, r)._login_window_present("cid") is False


async def test_login_window_present_false_on_blank_stdout(tmp_path):
    r = _RecordingRunner((0, b"\n", b""))
    assert await _ready_harness(tmp_path, r)._login_window_present("cid") is False


async def test_screen_signature_crops_center_and_returns_hash(tmp_path):
    r = _RecordingRunner((0, b"abc123def  -\n", b""))
    h = _ready_harness(tmp_path, r)
    sig = await h._screen_signature("cid")
    assert sig == "abc123def"
    # 1280x800 -> 420x470 box centred (excludes WM toolbar/clock)
    assert "-crop 420x470+430+165" in r.last_cmd
    assert "import -window root" in r.last_cmd and "md5sum" in r.last_cmd


async def test_screen_signature_none_when_capture_fails(tmp_path):
    r = _RecordingRunner((1, b"", b"boom"))
    assert await _ready_harness(tmp_path, r)._screen_signature("cid") is None


async def test_wait_ready_skipped_when_zero(tmp_path):
    r = _RecordingRunner()
    h = _ready_harness(tmp_path, r, MUREX_CONTAINER_READY_SECS="0")
    await h._wait_ready("cid")
    assert r.calls == []  # no probing at all when disabled


async def test_wait_ready_waits_for_window_then_stability(tmp_path, monkeypatch):
    import asyncio as _asyncio

    async def _nosleep(_secs):  # collapse all polling delays
        return None

    monkeypatch.setattr(_asyncio, "sleep", _nosleep)

    state = {"win_polls": 0, "sig_polls": 0}

    class Probe:
        async def __call__(self, argv):
            cmd = argv[-1]
            if "xdotool search --class" in cmd:
                state["win_polls"] += 1
                # absent on the first poll, present afterwards
                return (0, b"42\n", b"") if state["win_polls"] >= 2 else (1, b"", b"")
            if "md5sum" in cmd:
                state["sig_polls"] += 1
                # frame still painting on first signature, constant after
                sig = b"AAAA  -\n" if state["sig_polls"] == 1 else b"BBBB  -\n"
                return (0, sig, b"")
            return (0, b"", b"")

    h = _ready_harness(
        tmp_path, Probe(),
        MUREX_CONTAINER_READY_SECS="90",
        MUREX_CONTAINER_READY_STABLE_POLLS="2",
    )
    await h._wait_ready("cid")
    # gated on the window first, then confirmed paint via stable signatures
    assert state["win_polls"] >= 2
    # first sig differs (painting), then needs 2 more equal -> >=3 polls
    assert state["sig_polls"] >= 3


def test_force_stop_live_containers_uses_subprocess(monkeypatch):
    # The atexit net force-stops anything still tracked, synchronously.
    from iag_sim.harness import docker as dmod

    dmod._LIVE_CONTAINERS.add("cidZZ")
    calls: list[list[str]] = []
    monkeypatch.setattr(dmod.subprocess, "run", lambda argv, **kw: calls.append(argv))
    dmod._force_stop_live_containers()
    assert ["docker", "stop", "cidZZ"] in calls
    assert "cidZZ" not in dmod._LIVE_CONTAINERS
