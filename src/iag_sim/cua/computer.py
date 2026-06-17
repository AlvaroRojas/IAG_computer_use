"""Playwright-backed Computer harness for OpenAI computer-use.

Wraps an async Playwright `Page`: executes the model's actions and produces
screenshots. Also captures browser downloads so the simulation's CSV export can
be saved to disk.

Coordinates from the model are pixel positions in a viewport sized to
display_width x display_height (set when the context is created).
"""

from __future__ import annotations

import asyncio
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


# Pixels per wheel notch. The canonical scroll delta is in NOTCHES (what OpenAI
# emits and what the thick channel's `xdotool click --repeat` consumes 1:1); the
# Playwright wheel takes PIXELS, so the browser scales notches -> px here. ~100px
# per notch matches a typical browser wheel step.
_WHEEL_PX_PER_NOTCH = 100


class PlaywrightComputer:
    """Implements iag_sim.cua.base.Computer over a Playwright Page."""

    def __init__(
        self,
        page: Page,
        download_dir: Path,
        *,
        click_delay_ms: int = 0,
        settle_ms: int = 0,
    ) -> None:
        self.page = page
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        # Web-SPA interaction tuning (defaults 0 = no-op, for the thick channel /
        # tests). click_delay_ms is the mousedown->mouseup gap that lets custom
        # widgets register a synthetic click; settle_ms is a post-action repaint
        # pause so the next screenshot reflects settled UI, not a mid-transition
        # frame. Wired from Settings in harness/browser.py.
        self._click_delay_ms = click_delay_ms
        self._settle_ms = settle_ms
        self._pending: list[Download] = []
        self._saved: list[Path] = []
        # Capture downloads. The event fires on the Page the download originates
        # from: a same-page export fires on `page`, but a Murex 'Download as CSV'
        # that opens a popup / new tab fires on the CHILD page. So listen on `page`
        # AND on every page opened later in this context, else popup-originated
        # exports are missed and the reality gate reports 'no CSV was exported'.
        # Handlers must be plain functions (Playwright sets attrs on the handler,
        # which fails on bound/builtin methods) — hence the lambdas.
        self.page.on("download", lambda d: self._pending.append(d))
        self.page.context.on(
            "page", lambda pg: pg.on("download", lambda d: self._pending.append(d))
        )

    async def wait_for_download(self, timeout: float, poll: float = 0.25) -> bool:
        """Wait up to `timeout` s for a browser download to arrive before flushing.

        The `page.on("download")` listener appends to `self._pending` the instant a
        real download starts; the model's last action often triggers it just as the
        agent loop returns, so without this wait `collect_export` could miss it and
        report a false 'no CSV'. Returns True once a download is pending/saved, False
        on timeout. Fast-paths when one is already present or `timeout` <= 0.
        """
        if self._pending or self._saved:
            return True
        if timeout <= 0:
            return False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._pending:
                return True
            await asyncio.sleep(poll)
        return bool(self._pending)

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

    async def _settle(self) -> None:
        """Pause for the page to repaint before the loop screenshots.

        The web MX.3 SPA animates menus/dropdowns, so an action's visual effect
        lands a beat after the call returns; without this the next screenshot
        catches a mid-transition frame and the model mis-clicks (dropdown
        selection appears not to take, typed text appears not to land). 0 = off
        (thick channel repaints synchronously; tests use the no-op default).
        """
        if self._settle_ms > 0:
            await self.page.wait_for_timeout(self._settle_ms)

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
            # delay = mousedown->mouseup gap: custom widgets distinguish a real
            # click from a too-fast synthetic one and otherwise drop it.
            await self.page.mouse.click(x, y, button=button, delay=self._click_delay_ms)
        finally:
            await self._apply_modifiers(keys, down=False)
        await self._settle()

    async def double_click(self, x, y, button="left", keys=None):
        await self._apply_modifiers(keys, down=True)
        try:
            await self.page.mouse.dblclick(
                x, y, button=button, delay=self._click_delay_ms
            )
        finally:
            await self._apply_modifiers(keys, down=False)
        await self._settle()

    async def move(self, x, y):
        await self.page.mouse.move(x, y)
        # Settle after a hover too: Murex submenus expand on hover, so the loop
        # must screenshot the expanded menu, not the pre-hover frame.
        await self._settle()

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
        await self._settle()

    async def scroll(self, x, y, scroll_x=0, scroll_y=0):
        await self.page.mouse.move(x, y)
        # Canonical deltas are wheel notches; mouse.wheel wants pixels.
        await self.page.mouse.wheel(
            scroll_x * _WHEEL_PX_PER_NOTCH, scroll_y * _WHEEL_PX_PER_NOTCH
        )
        await self._settle()

    async def type(self, text):
        await self.page.keyboard.type(text)
        await self._settle()

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
        await self._settle()

    async def wait(self, ms=1000):
        await self.page.wait_for_timeout(ms)
