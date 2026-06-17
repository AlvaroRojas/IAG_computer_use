"""Playwright-backed Computer harness for OpenAI computer-use.

Wraps an async Playwright `Page`: executes the model's actions and produces
screenshots. Also captures browser downloads so the simulation's CSV export can
be saved to disk.

Coordinates from the model are pixel positions in a viewport sized to
display_width x display_height (set when the context is created).
"""

from __future__ import annotations

import base64
from pathlib import Path

from playwright.async_api import Download, Page

# Model key name -> Playwright key name.
_KEY_MAP = {
    "ENTER": "Enter",
    "RETURN": "Enter",
    "TAB": "Tab",
    "SPACE": " ",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "ARROWUP": "ArrowUp",
    "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
    "OPTION": "Alt",
    "SHIFT": "Shift",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "META": "Meta",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "HOME": "Home",
    "END": "End",
}


def _map_key(key: str) -> str:
    return _KEY_MAP.get(key.upper(), key)


class PlaywrightComputer:
    """Implements iag_sim.cua.base.Computer over a Playwright Page."""

    def __init__(self, page: Page, download_dir: Path) -> None:
        self.page = page
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._pending: list[Download] = []
        self._saved: list[Path] = []
        # The download event fires synchronously; just capture the object.
        # Must be a plain function (Playwright sets attrs on the handler, which
        # fails on bound/builtin methods) — hence the lambda.
        self.page.on("download", lambda d: self._pending.append(d))

    async def flush_downloads(self) -> list[Path]:
        """Persist any downloads captured since the last flush. Call after
        turns that may have triggered an export. Returns all saved paths."""
        while self._pending:
            download = self._pending.pop(0)
            name = download.suggested_filename or "export.csv"
            target = self.download_dir / name
            await download.save_as(str(target))
            self._saved.append(target)
        return list(self._saved)

    @property
    def downloads(self) -> list[Path]:
        return list(self._saved)

    async def screenshot(self) -> str:
        png = await self.page.screenshot(type="png")
        return base64.b64encode(png).decode("ascii")

    async def _apply_modifiers(self, keys: list[str] | None, down: bool) -> None:
        for k in keys or []:
            mapped = _map_key(k)
            if down:
                await self.page.keyboard.down(mapped)
            else:
                await self.page.keyboard.up(mapped)

    async def click(self, x, y, button="left", keys=None):
        await self._apply_modifiers(keys, down=True)
        try:
            await self.page.mouse.click(x, y, button=button)
        finally:
            await self._apply_modifiers(keys, down=False)

    async def double_click(self, x, y, button="left", keys=None):
        await self._apply_modifiers(keys, down=True)
        try:
            await self.page.mouse.dblclick(x, y, button=button)
        finally:
            await self._apply_modifiers(keys, down=False)

    async def move(self, x, y):
        await self.page.mouse.move(x, y)

    async def drag(self, path, keys=None):
        if not path:
            return
        await self._apply_modifiers(keys, down=True)
        try:
            start = path[0]
            await self.page.mouse.move(start[0], start[1])
            await self.page.mouse.down()
            for point in path[1:]:
                await self.page.mouse.move(point[0], point[1])
            await self.page.mouse.up()
        finally:
            await self._apply_modifiers(keys, down=False)

    async def scroll(self, x, y, scroll_x=0, scroll_y=0):
        await self.page.mouse.move(x, y)
        await self.page.mouse.wheel(scroll_x, scroll_y)

    async def type(self, text):
        await self.page.keyboard.type(text)

    async def keypress(self, keys):
        mapped = [_map_key(k) for k in keys]
        if not mapped:
            return
        # Hold all but the last, press the last, release in reverse (combo).
        for k in mapped[:-1]:
            await self.page.keyboard.down(k)
        await self.page.keyboard.press(mapped[-1])
        for k in reversed(mapped[:-1]):
            await self.page.keyboard.up(k)

    async def wait(self, ms=1000):
        await self.page.wait_for_timeout(ms)
