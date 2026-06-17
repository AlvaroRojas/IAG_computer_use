"""Web-UI channel: drive the Murex web front end with Playwright/Chromium.

Real parallelism — each trade gets its own browser context, all sharing the
per-environment `storage_state` saved at login so we don't re-authenticate N
times.
"""

from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI
from playwright.async_api import Browser

from ..config import Settings
from ..cua.computer import PlaywrightComputer
from ..models import EnvName, TradeTask
from ..murex.login import login_and_save_state
from .base import Harness, TradeSession


class BrowserSession:
    def __init__(self, context, page, computer: PlaywrightComputer, display: tuple[int, int]):
        self._context = context
        self._page = page
        self.computer = computer
        self.display = display

    async def collect_export(self) -> Path | None:
        saved = await self.computer.flush_downloads()
        return saved[-1] if saved else None

    async def close(self) -> None:
        await self._context.close()


class BrowserHarness(Harness):
    supports_parallel = True

    def __init__(
        self,
        env: EnvName,
        settings: Settings,
        browser: Browser,
        run_dir: Path,
    ) -> None:
        self.env = env
        self.settings = settings
        self.browser = browser
        self.run_dir = run_dir
        self._state_path: Path | None = None

    async def setup(self) -> None:
        # LLM-login mode: skip the deterministic pre-auth so each session lands
        # on the login screen and the computer-use model logs in + picks the
        # group itself. Trades no longer share a storage_state (one login each).
        if self.settings.murex_llm_login:
            self._state_path = None
            return
        self._state_path = await login_and_save_state(
            self.browser, self.env, self.settings, self.run_dir / ".sessions"
        )

    async def new_session(self, trade: TradeTask) -> TradeSession:
        s = self.settings
        download_dir = self.run_dir / self.env.value / trade.trade_id
        ctx_kwargs: dict = {
            "viewport": {"width": s.display_width, "height": s.display_height},
            "accept_downloads": True,
        }
        # Deterministic-login mode reuses the saved authenticated session;
        # LLM-login mode starts cold so the model sees the login page.
        if self._state_path is not None:
            ctx_kwargs["storage_state"] = str(self._state_path)
        elif not s.murex_llm_login:
            raise AssertionError("setup() not called")
        context = await self.browser.new_context(**ctx_kwargs)
        page = await context.new_page()
        await page.goto(s.url_for(self.env.value), wait_until="domcontentloaded")
        computer = PlaywrightComputer(page, download_dir)
        return BrowserSession(context, page, computer, (s.display_width, s.display_height))

    async def aclose(self) -> None:
        # The browser is owned by Resources; nothing per-harness to release.
        return None
