"""Thick-client channel: drive the Murex Java desktop client running inside a
Linux Docker container (Xvfb + the client), one container per trade.

Parallelism = many containers (each its own X display), bounded by
MAX_CONCURRENCY — so the thick client parallelizes just like the web channel.

The container is driven via `docker exec <c> sh -c ...` (OpenAI's documented
Docker action handler):
  - screenshot: ImageMagick `import -window root png:-` -> PNG bytes on stdout
  - mouse/keyboard: `xdotool`
computer-use `environment` for this channel is "ubuntu".

Container contract (the user-supplied image must honour this):
  - boots an X server on $DISPLAY (default :99) at $SCREEN_GEOMETRY
  - launches the Murex client connecting to $MUREX_ENV_TARGET, then EITHER:
      * MUREX_LLM_LOGIN unset/false: the entrypoint LOGS IN using
        $MUREX_USER / $MUREX_PASS and leaves the app ready (creds handled by the
        image — never typed by the model, so they stay out of LLM context); or
      * MUREX_LLM_LOGIN=true: the entrypoint leaves the client at the LOGIN
        screen (thick login cannot be scripted) and the computer-use model types
        the credentials + selects the group itself. $MUREX_USER/$MUREX_PASS are
        NOT passed into the container in this mode.
  - has ImageMagick (`import`) and `xdotool` installed
  - exports land in the mounted export dir ($MUREX_CONTAINER_EXPORT_DIR)

The exec/run mechanism is injected (`runner`) so the command construction is
unit-testable without Docker.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import shlex
import subprocess
from pathlib import Path
from typing import Awaitable, Callable

from ..config import Settings
from ..models import EnvName, TradeTask
from .base import Harness, TradeSession

ExecResult = tuple[int, bytes, bytes]
Runner = Callable[[list[str]], Awaitable[ExecResult]]


# Last-resort registry of containers we started. The orchestrator stops them
# cleanly (DockerSession.close / DockerHarness.aclose), but if the process exits
# abnormally — Ctrl+C propagating out of asyncio.run, an unhandled exception —
# this atexit hook synchronously force-stops whatever is still tracked so a dead
# Python NEVER leaves Murex containers running. (SIGKILL can't be caught; nothing
# can cover that.) A container is removed from the set only after it is stopped.
_LIVE_CONTAINERS: set[str] = set()


def _force_stop_live_containers() -> None:
    for cid in list(_LIVE_CONTAINERS):
        try:
            subprocess.run(
                ["docker", "stop", cid],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        except Exception:
            pass
        _LIVE_CONTAINERS.discard(cid)


atexit.register(_force_stop_live_containers)

_BUTTON = {"left": "1", "middle": "2", "right": "3"}

# model key name -> X keysym (xdotool)
_XKEY = {
    "ENTER": "Return",
    "RETURN": "Return",
    "TAB": "Tab",
    "SPACE": "space",
    "BACKSPACE": "BackSpace",
    "DELETE": "Delete",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "UP": "Up",
    "DOWN": "Down",
    "LEFT": "Left",
    "RIGHT": "Right",
    "ARROWUP": "Up",
    "ARROWDOWN": "Down",
    "ARROWLEFT": "Left",
    "ARROWRIGHT": "Right",
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "ALT": "alt",
    "SHIFT": "shift",
    "CMD": "super",
    "META": "super",
    "PAGEUP": "Prior",
    "PAGEDOWN": "Next",
    "HOME": "Home",
    "END": "End",
}


def _xkey(k: str) -> str:
    return _XKEY.get(k.upper(), k)


async def subprocess_runner(argv: list[str]) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return (proc.returncode or 0, out, err)


class DockerComputer:
    """Implements cua.base.Computer by exec-ing xdotool/import in a container."""

    def __init__(
        self, container_id: str, display: str, size: tuple[int, int], runner: Runner
    ) -> None:
        self.cid = container_id
        self.display = display
        self.size = size
        self.runner = runner

    async def _exec(self, shell_cmd: str) -> ExecResult:
        # Matches OpenAI's documented Docker handler: `docker exec <c> sh -c ...`
        full = f"export DISPLAY={self.display} && {shell_cmd}"
        return await self.runner(["docker", "exec", self.cid, "sh", "-c", full])

    async def _xdo(self, shell_cmd: str) -> None:
        rc, _out, err = await self._exec(shell_cmd)
        if rc != 0:
            raise RuntimeError(f"docker exec failed ({rc}): {err.decode(errors='replace')}")

    async def screenshot(self) -> str:
        # ImageMagick `import` per OpenAI's reference Dockerfile (no scrot needed).
        rc, out, err = await self._exec("import -window root png:-")
        if rc != 0:
            raise RuntimeError(f"screenshot failed ({rc}): {err.decode(errors='replace')}")
        return base64.b64encode(out).decode("ascii")

    async def click(self, x, y, button="left", keys=None):
        mods = [_xkey(k) for k in keys or []]
        parts = [f"xdotool keydown {m}" for m in mods]
        parts.append(f"xdotool mousemove {int(x)} {int(y)} click {_BUTTON.get(button, '1')}")
        parts += [f"xdotool keyup {m}" for m in reversed(mods)]
        await self._xdo("; ".join(parts))

    async def double_click(self, x, y, button="left", keys=None):
        await self._xdo(
            f"xdotool mousemove {int(x)} {int(y)} click --repeat 2 {_BUTTON.get(button, '1')}"
        )

    async def move(self, x, y):
        await self._xdo(f"xdotool mousemove {int(x)} {int(y)}")

    async def drag(self, path, keys=None):
        if not path:
            return
        sx, sy = path[0]
        parts = [f"xdotool mousemove {int(sx)} {int(sy)} mousedown 1"]
        for px, py in path[1:]:
            parts.append(f"xdotool mousemove {int(px)} {int(py)}")
        parts.append("xdotool mouseup 1")
        await self._xdo("; ".join(parts))

    async def scroll(self, x, y, scroll_x=0, scroll_y=0):
        parts = [f"xdotool mousemove {int(x)} {int(y)}"]
        if scroll_y:
            button = "5" if scroll_y > 0 else "4"  # 5=down, 4=up
            parts.append(f"xdotool click --repeat {abs(int(scroll_y))} {button}")
        if scroll_x:
            button = "7" if scroll_x > 0 else "6"
            parts.append(f"xdotool click --repeat {abs(int(scroll_x))} {button}")
        await self._xdo("; ".join(parts))

    async def type(self, text):
        await self._xdo(f"xdotool type --clearmodifiers -- {shlex.quote(text)}")

    async def keypress(self, keys):
        if not keys:
            return
        combo = "+".join(_xkey(k) for k in keys)
        await self._xdo(f"xdotool key --clearmodifiers {combo}")

    async def wait(self, ms=1000):
        await asyncio.sleep(ms / 1000)


# Dirs to sweep for a stray CSV if the model saved outside the bind mount
# (e.g. accepted the chooser's default `/opt/murex` instead of `/exports`).
_FALLBACK_SEARCH_DIRS = ("/opt/murex", "/root", "/home")


class DockerSession:
    def __init__(self, cid, computer, display, host_export, stop, runner=None,
                 container_export_dir="/exports", poll=0.5, stable_polls=2):
        self.cid = cid
        self.computer = computer
        self.display = display
        self._host_export = host_export
        self._stop = stop
        self._runner = runner
        self._container_export_dir = container_export_dir
        self._poll = poll
        self._stable_polls = stable_polls

    async def collect_export(self, timeout: float = 0.0) -> Path | None:
        """Return the exported CSV. Primary path: the bind-mounted export dir.
        Waits up to `timeout` s for a CSV to APPEAR and its size to SETTLE — the
        bind-mounted file can be globbed mid-write right after the model presses
        Enter on the save chooser. Fallback: the model sometimes saves to the
        chooser's default directory instead of `/exports` — find the newest stray
        CSV in the container and copy it into the export dir so the deterministic
        pipeline still sees it."""
        path = await self._wait_for_stable_csv(timeout)
        if path is not None:
            return path
        return await self._recover_stray_csv()

    async def _wait_for_stable_csv(self, timeout: float) -> Path | None:
        """Newest `*.csv` in the bind mount once its size is stable for
        `stable_polls` polls, capped by `timeout`. With timeout<=0, returns the
        newest immediately (no wait). On timeout, returns the newest as a
        best-effort (content validation downstream still gates it)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        last_size = -1
        stable = 0
        while True:
            csvs = sorted(self._host_export.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            newest = csvs[-1] if csvs else None
            if newest is not None:
                size = newest.stat().st_size
                if size > 0 and size == last_size:
                    stable += 1
                    if stable >= self._stable_polls:
                        return newest
                else:
                    stable = 0
                    last_size = size
            if loop.time() >= deadline:
                return newest
            await asyncio.sleep(self._poll)

    async def _recover_stray_csv(self) -> Path | None:
        if self._runner is None:
            return None
        dirs = " ".join(_FALLBACK_SEARCH_DIRS + (self._container_export_dir,))
        # newest *.csv across the candidate dirs -> "<epoch>\t<path>"
        find = (
            f"find {dirs} -maxdepth 4 -type f -name '*.csv' "
            "-printf '%T@\\t%p\\n' 2>/dev/null | sort -rn | head -1"
        )
        rc, out, _err = await self._runner(["docker", "exec", self.cid, "sh", "-c", find])
        if rc != 0:
            return None
        line = out.decode(errors="replace").strip()
        if not line or "\t" not in line:
            return None
        src = line.split("\t", 1)[1].strip()
        dest = self._host_export / Path(src).name
        rc, _o, _e = await self._runner(["docker", "cp", f"{self.cid}:{src}", str(dest)])
        if rc != 0 or not dest.exists():
            return None
        return dest

    async def close(self) -> None:
        await self._stop(self.cid)


class DockerHarness(Harness):
    supports_parallel = True

    def __init__(
        self, env: EnvName, settings: Settings, run_dir: Path, runner: Runner | None = None
    ) -> None:
        self.env = env
        self.settings = settings
        self.run_dir = run_dir
        self.runner = runner or subprocess_runner
        # Every container this harness starts, so aclose() can stop ALL of them —
        # including one whose session was cancelled mid-boot (before new_session
        # returned), which DockerSession.close() would otherwise never reach.
        self._containers: set[str] = set()

    async def setup(self) -> None:
        if not self.settings.murex_docker_image:
            raise ValueError("MUREX_DOCKER_IMAGE must be set for the thick (docker) channel")
        rc, _out, err = await self.runner(["docker", "version", "--format", "{{.Server.Version}}"])
        if rc != 0:
            raise RuntimeError(f"docker not available: {err.decode(errors='replace')}")

    def _run_argv(self, trade: TradeTask, host_export: Path) -> list[str]:
        s = self.settings
        geometry = f"{s.display_width}x{s.display_height}"
        argv = [
            "docker", "run", "-d", "--rm",
        ]
        # Per-container cgroup caps (empty string disables either flag).
        if s.murex_docker_cpus:
            argv += ["--cpus", s.murex_docker_cpus]
        if s.murex_docker_memory:
            argv += ["--memory", s.murex_docker_memory]
        argv += [
            "-e", f"MUREX_ENV_TARGET={s.url_for(self.env.value)}",
        ]
        # Deterministic-login mode: hand the creds to the entrypoint. LLM-login
        # mode: the container boots to the login screen and the model types them,
        # so the credentials never enter the container environment.
        if not s.murex_llm_login:
            argv += [
                "-e", f"MUREX_USER={s.murex_user}",
                "-e", f"MUREX_PASS={s.murex_pass.get_secret_value()}",
            ]
        argv += [
            "-e", f"MUREX_TRADE_ID={trade.trade_id}",
            "-e", f"DISPLAY={s.murex_display}",
            "-e", f"SCREEN_GEOMETRY={geometry}",
            "-v", f"{host_export}:{s.murex_container_export_dir}",
            *s.docker_run_extra,
            s.murex_docker_image,  # type: ignore[list-item]
        ]
        return argv

    async def _login_window_present(self, cid: str) -> bool:
        """True once a top-level window whose WM_CLASS matches the Murex client
        class exists. This gates the readiness probe: the flat boot desktop is
        already 'stable' before the client paints, so stability alone would fire
        too early — the window must exist first."""
        cls = self.settings.container_login_window_class
        cmd = f"DISPLAY={self.settings.murex_display} xdotool search --class {shlex.quote(cls)}"
        rc, out, _err = await self.runner(["docker", "exec", cid, "sh", "-c", cmd])
        return rc == 0 and out.strip() != b""

    async def _screen_signature(self, cid: str) -> str | None:
        """Cheap hash of the CENTRE of the screen (where the login card / app
        renders), excluding the WM toolbar + clock so a ticking clock never
        defeats stability. Quantised (few colours, low depth) so sub-pixel noise
        doesn't churn the hash. None if the screenshot fails."""
        s = self.settings
        cw = min(420, s.display_width)
        ch = min(470, s.display_height)
        cx = (s.display_width - cw) // 2
        cy = (s.display_height - ch) // 2
        crop = f"{cw}x{ch}+{cx}+{cy}"
        cmd = (
            f"DISPLAY={s.murex_display} import -window root -crop {crop} +repage "
            "-depth 4 -colors 16 txt:- 2>/dev/null | md5sum"
        )
        rc, out, _err = await self.runner(["docker", "exec", cid, "sh", "-c", cmd])
        if rc != 0:
            return None
        token = out.split()[0] if out.split() else b""
        return token.decode(errors="replace") or None

    async def _wait_ready(self, cid: str) -> None:
        """Two-phase readiness probe, hard-capped by container_ready_secs:
          1. wait until the Murex client window exists;
          2. wait until the screen signature is stable for
             container_ready_stable_polls consecutive polls (done painting).
        On timeout it simply returns (proceed anyway) — never slower than the
        old fixed sleep, usually much faster."""
        s = self.settings
        if s.container_ready_secs <= 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + s.container_ready_secs
        poll = s.container_ready_poll_secs

        # Phase 1: window appears.
        while loop.time() < deadline:
            if await self._login_window_present(cid):
                break
            await asyncio.sleep(poll)

        # Phase 2: screen stops changing.
        need = s.container_ready_stable_polls
        prev: str | None = None
        stable = 0
        while loop.time() < deadline:
            await asyncio.sleep(poll)
            sig = await self._screen_signature(cid)
            if sig is not None and sig == prev:
                stable += 1
                if stable >= need:
                    return
            else:
                stable = 0
                prev = sig

    async def new_session(self, trade: TradeTask) -> TradeSession:
        s = self.settings
        # Docker bind mounts (`-v src:dst`) require an ABSOLUTE source path; the
        # run_dir is typically relative (data/out/run-...). Resolve it or Docker
        # Desktop rejects the mount ("source path must be absolute").
        host_export = (self.run_dir / self.env.value / trade.trade_id).resolve()
        host_export.mkdir(parents=True, exist_ok=True)
        # Fresh attempt: drop any *.csv a prior (failed/invalid) attempt for THIS
        # trade left behind. host_export is per-trade and reused across retries, and
        # collect_export globs it by mtime — without this, a stale bad file could be
        # re-collected even if this attempt exported nothing.
        for old in host_export.glob("*.csv"):
            try:
                old.unlink()
            except OSError:
                pass

        rc, out, err = await self.runner(self._run_argv(trade, host_export))
        if rc != 0:
            raise RuntimeError(f"docker run failed ({rc}): {err.decode(errors='replace')}")
        cid = out.decode().strip()
        # Track immediately — BEFORE the ready-sleep — so a cancellation during
        # the (long) boot wait still leaves the container reachable by teardown.
        if cid:
            self._containers.add(cid)
            _LIVE_CONTAINERS.add(cid)

        # Wait for the Murex client to be ready — PROBED, not slept (see
        # _wait_ready): returns as soon as the login/app screen is painted,
        # capped by container_ready_secs.
        if cid:
            await self._wait_ready(cid)

        computer = DockerComputer(
            cid, s.murex_display, (s.display_width, s.display_height), self.runner
        )

        async def _stop(container_id: str) -> None:
            await self.runner(["docker", "stop", container_id])
            # Untrack only after a successful stop, so a failed/cancelled stop
            # stays on the list for aclose()/atexit to retry.
            self._containers.discard(container_id)
            _LIVE_CONTAINERS.discard(container_id)

        return DockerSession(
            cid, computer, (s.display_width, s.display_height), host_export, _stop,
            runner=self.runner, container_export_dir=s.murex_container_export_dir,
            poll=s.export_poll_secs, stable_polls=s.export_stable_polls,
        )

    async def aclose(self) -> None:
        """Stop EVERY container this harness started that is still tracked.
        Runs via open_resources' AsyncExitStack on any exit — normal completion,
        exception, OR cancellation (Ctrl+C) — so no container outlives the run.
        Each stop is shielded so teardown completes even while the task is being
        cancelled, and errors (already-gone container) are ignored."""
        for cid in list(self._containers):
            try:
                await asyncio.shield(self.runner(["docker", "stop", cid]))
            except Exception:
                pass
            finally:
                self._containers.discard(cid)
                _LIVE_CONTAINERS.discard(cid)
