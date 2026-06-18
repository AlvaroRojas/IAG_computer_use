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
    def __init__(
        self,
        context,
        page,
        computer: PlaywrightComputer,
        display: tuple[int, int],
        poll: float = 0.25,
    ):
        self._context = context
        self._page = page
        self.computer = computer
        self.display = display
        self._poll = poll

    async def collect_export(self, timeout: float = 0.0) -> Path | None:
        # Wait for the download event to land (the model's last action may fire it
        # just as the loop returns), then persist it. save_as resolves only on a
        # COMPLETED browser download, so a returned path is a real file.
        await self.computer.wait_for_download(timeout, poll=self._poll)
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
            # On-prem Murex serves a self-signed cert; without this, navigation
            # aborts with ERR_CERT_AUTHORITY_INVALID.
            "ignore_https_errors": s.murex_ignore_https_errors,
            # Playwright contexts grant no permissions by default, so the async
            # Clipboard API (navigator.clipboard.*) is rejected. Murex's web UI
            # uses it for copy/paste in some grids; grant it so those paths work.
            # Keyboard Ctrl+C/Ctrl+V works regardless — this is only for the JS API.
            "permissions": ["clipboard-read", "clipboard-write"],
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
        computer = PlaywrightComputer(
            page,
            download_dir,
            click_delay_ms=s.cua_web_click_delay_ms,
            settle_ms=s.cua_web_settle_ms,
        )
        return BrowserSession(
            context, page, computer, (s.display_width, s.display_height),
            poll=s.export_poll_secs,
        )

    async def aclose(self) -> None:
        # The browser is owned by Resources; nothing per-harness to release.
        return None
