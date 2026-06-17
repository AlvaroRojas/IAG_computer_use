"""Web channel must pass `ignore_https_errors` to every Playwright context it
opens — on-prem Murex serves a self-signed cert, so without it navigation aborts
with ERR_CERT_AUTHORITY_INVALID (the failure that sank a live run). Covers BOTH
context-creating paths: the per-trade worker context (BrowserHarness.new_session)
and the deterministic pre-auth context (login_and_save_state).
"""

from __future__ import annotations

import asyncio

from iag_sim.config import Settings
from iag_sim.harness.browser import BrowserHarness
from iag_sim.models import EnvName, TradeTask
from iag_sim.murex.login import login_and_save_state

REQUIRED = {
    "OPENAI_API_KEY": "sk-test",
    "MUREX_BEFORE_URL": "https://10.0.0.1/apps/webclient/",
    "MUREX_AFTER_URL": "https://10.0.0.1/apps/webclient/",
    "MUREX_USER": "u",
    "MUREX_PASS": "p",
}


def _settings(monkeypatch, **extra) -> Settings:
    for k, v in {**REQUIRED, **extra}.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


class _FakePage:
    def __init__(self) -> None:
        self.handlers: dict = {}

    def on(self, event, handler):  # PlaywrightComputer registers a download listener
        self.handlers[event] = handler

    async def goto(self, url, wait_until=None):
        self.goto_url = url

    async def fill(self, selector, value):
        pass

    async def click(self, selector):
        pass

    async def wait_for_selector(self, selector, timeout=None):
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.page = _FakePage()

    def on(self, *_args, **_kwargs):  # context "page" listener registration: no-op
        pass

    async def new_page(self):
        self.page.context = self  # PlaywrightComputer reads page.context.on(...)
        return self.page

    async def storage_state(self, path=None):
        # login_and_save_state persists to this path; write something real.
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}")

    async def close(self):
        pass


class _FakeBrowser:
    def __init__(self) -> None:
        self.context_kwargs: list[dict] = []

    async def new_context(self, **kwargs):
        self.context_kwargs.append(kwargs)
        return _FakeContext()


def test_new_session_passes_ignore_https_errors_true_by_default(monkeypatch, tmp_path):
    s = _settings(monkeypatch, MUREX_CHANNEL="web", MUREX_LLM_LOGIN="true")
    browser = _FakeBrowser()
    harness = BrowserHarness(EnvName.BEFORE, s, browser, tmp_path)

    asyncio.run(harness.setup())  # LLM-login mode: no pre-auth
    asyncio.run(harness.new_session(TradeTask(trade_id="4572")))

    assert browser.context_kwargs[-1]["ignore_https_errors"] is True


def test_new_session_respects_ignore_https_errors_false(monkeypatch, tmp_path):
    s = _settings(
        monkeypatch,
        MUREX_CHANNEL="web",
        MUREX_LLM_LOGIN="true",
        MUREX_IGNORE_HTTPS_ERRORS="false",
    )
    browser = _FakeBrowser()
    harness = BrowserHarness(EnvName.AFTER, s, browser, tmp_path)

    asyncio.run(harness.setup())
    asyncio.run(harness.new_session(TradeTask(trade_id="4572")))

    assert browser.context_kwargs[-1]["ignore_https_errors"] is False


def test_deterministic_login_passes_ignore_https_errors(monkeypatch, tmp_path):
    s = _settings(monkeypatch, MUREX_CHANNEL="web")  # llm_login defaults false
    browser = _FakeBrowser()

    asyncio.run(login_and_save_state(browser, EnvName.BEFORE, s, tmp_path / "sess"))

    assert browser.context_kwargs[-1]["ignore_https_errors"] is True
