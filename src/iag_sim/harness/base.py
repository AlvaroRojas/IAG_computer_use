"""Harness abstraction: one Murex access channel bound to one environment.

A `Harness` knows how to log in / attach to a Murex environment and mint a
`TradeSession` for each trade. A `TradeSession` exposes a `Computer` (driven by
the computer-use loop), the display size the model should target, and a way to
collect the CSV the simulation exports.

Two implementations:
  - BrowserHarness  -> Murex WEB UI via Playwright (real parallelism)
  - DesktopHarness  -> Murex THICK Java client via OS-level input (sequential)

Both yield the same `Computer` interface, so the agent loop is identical
regardless of channel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from ..cua.base import Computer
from ..models import EnvName, TradeTask


class TradeSession(Protocol):
    computer: Computer
    display: tuple[int, int]  # (width, height) the model should assume

    async def collect_export(self, timeout: float = 0.0) -> Path | None:
        """Return the CSV produced by this session's simulation, or None.

        `timeout` (seconds) bounds how long to wait for the export to APPEAR — the
        model's last action may trigger the download/file write just as the agent
        loop returns. 0 = check once, no wait.
        """
        ...

    async def close(self) -> None: ...


class Harness(ABC):
    """One channel + one environment. `supports_parallel` tells the orchestrator
    whether many sessions can run concurrently (web: yes; thick desktop: no —
    a single desktop has one mouse/keyboard)."""

    env: EnvName
    supports_parallel: bool = True

    @abstractmethod
    async def setup(self) -> None:
        """Log in / launch / attach. Called once before any session."""

    @abstractmethod
    async def new_session(self, trade: TradeTask) -> TradeSession:
        """Prepare a session to simulate one trade in this environment."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release channel-level resources (called once at run end)."""
