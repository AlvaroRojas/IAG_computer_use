"""Computer-use harness interface — no external dependencies.

Kept dependency-free (no Playwright import) so the action-dispatch layer and its
unit tests don't require a browser. `PlaywrightComputer` (computer.py) and any
fake/test double implement this Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# An action is the raw dict the model returns inside computer_call.actions[].
Action = dict[str, Any]


@runtime_checkable
class Computer(Protocol):
    """Minimal surface the agent loop drives. All methods are async so a single
    interface works for both Playwright (async) and test fakes."""

    async def screenshot(self) -> str:
        """Return a base64-encoded PNG of the current viewport."""
        ...

    async def click(
        self, x: int, y: int, button: str = "left", keys: list[str] | None = None
    ) -> None: ...

    async def double_click(
        self, x: int, y: int, button: str = "left", keys: list[str] | None = None
    ) -> None: ...

    async def move(self, x: int, y: int) -> None: ...

    async def drag(self, path: list[list[int]], keys: list[str] | None = None) -> None: ...

    async def scroll(
        self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0
    ) -> None: ...

    async def type(self, text: str) -> None: ...

    async def keypress(self, keys: list[str]) -> None: ...

    async def wait(self, ms: int = 1000) -> None: ...
