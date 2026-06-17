"""Real-time action tracing for the computer-use loop.

One `Tracer` per (trade, env) session writes a JSONL event stream that is flushed
on EVERY event — so the trace is readable WHILE the run is in progress, not only
after it finishes. Each event is also echoed as a compact one-line string to
stdout (prefixed with the session label) so a single tailable stream
(`data/out/run.log`) interleaves every agent's actions live.

Channel-agnostic: the loop drives a `Computer` (thick Docker xdotool OR browser
Playwright), so tracing here covers BOTH channels with one implementation.

Security: in MUREX_LLM_LOGIN mode the model TYPES the Murex password, so the
`type` action text would otherwise land in the trace + stdout. Any secret passed
in `secrets` is redacted to `***` in both the JSONL and the echo line.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    # Wall-clock UTC ISO-8601 with millis; cheap and monotonic enough for a trace.
    t = time.time()
    lt = time.gmtime(t)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{int((t % 1) * 1000):03d}Z"


# Action arg keys worth showing inline in the echo line (kept short + safe).
# `wait_ms`/`duration_ms`/`ms` surface how long a `wait` action paused.
_ECHO_KEYS = (
    "type", "x", "y", "button", "keys", "scroll_x", "scroll_y", "n",
    "wait_ms", "duration_ms", "ms",
)

# Stdout/run.log echo goes through the shared run logger so each line is
# timestamped by the logging formatter (see runlog.setup_run_logging).
_echo_log = logging.getLogger("iag_sim.trace")


class Tracer:
    """Append-only, flush-per-event JSONL tracer for one CUA session."""

    def __init__(
        self,
        path: Path,
        label: str,
        *,
        secrets: Iterable[str] = (),
        echo: bool = True,
    ) -> None:
        self.label = label
        self._secrets = [s for s in secrets if s]
        self._echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)
        # buffering=1 => line-buffered text mode; plus an explicit flush so the
        # OS hands the bytes over immediately for live tailing.
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    def _redact(self, s: str) -> str:
        for secret in self._secrets:
            s = s.replace(secret, "***")
        return s

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"ts": _now(), "label": self.label, "kind": kind, **fields}
        line = self._redact(json.dumps(rec, ensure_ascii=False, default=str))
        self._fh.write(line + "\n")
        self._fh.flush()
        if self._echo:
            _echo_log.info(self._redact(f"[trace {self.label}] {kind} {self._compact(fields)}"))

    def _compact(self, fields: dict[str, Any]) -> str:
        inner = fields.get("action") if isinstance(fields.get("action"), dict) else {}
        parts = [f"{k}={inner[k]}" for k in _ECHO_KEYS if k in inner]
        # Top-level fields the loop attaches alongside the action (e.g. the
        # resolved wait duration) — shown after the action's own keys.
        parts += [f"{k}={fields[k]}" for k in _ECHO_KEYS if k in fields and k not in inner]
        # The text a `type` action typed — same info the JSONL keeps, so run.log
        # is not poorer than the trace file. (Redacted at event() time, so a typed
        # password still shows as ***.)
        if inner.get("text"):
            parts.append(f"text={str(inner['text'])[:200]!r}")
        # Free-text top-level fields (reasoning summary / final message / error).
        for k in ("text", "error"):
            if fields.get(k):
                parts.append(f"{k}={str(fields[k])[:200]!r}")
        return " ".join(parts)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class NullTracer:
    """No-op tracer (default) — keeps the loop signature clean when tracing off."""

    label = "-"

    def event(self, kind: str, **fields: Any) -> None:  # noqa: D401 - no-op
        return None

    def close(self) -> None:
        return None
