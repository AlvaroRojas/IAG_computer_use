"""PlaywrightComputer unit tests with a fake Page (no real browser).

Covers the scroll unit contract: canonical deltas are wheel NOTCHES, and the
browser scales them to pixels for mouse.wheel.
"""

from __future__ import annotations

from iag_sim.cua.computer import PlaywrightComputer, _WHEEL_PX_PER_NOTCH


class _FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def move(self, x, y):
        self.calls.append(("move", x, y))

    async def wheel(self, dx, dy):
        self.calls.append(("wheel", dx, dy))

    async def click(self, x, y, button="left", delay=0):
        self.calls.append(("click", x, y, button, delay))

    async def dblclick(self, x, y, button="left", delay=0):
        self.calls.append(("dblclick", x, y, button, delay))


class _FakeKeyboard:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def type(self, text):
        self.calls.append(("type", text))

    async def press(self, key):
        self.calls.append(("press", key))

    async def down(self, key):
        self.calls.append(("down", key))

    async def up(self, key):
        self.calls.append(("up", key))


class _FakeContext:
    def __init__(self) -> None:
        self.page_handlers: list = []

    def on(self, event, handler):
        if event == "page":
            self.page_handlers.append(handler)

    def emit_page(self, pg):  # simulate a popup/new tab opening in this context
        for h in self.page_handlers:
            h(pg)


class _FakePage:
    def __init__(self, context: _FakeContext | None = None) -> None:
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.context = context or _FakeContext()
        self.download_handlers: list = []
        self.waits: list[float] = []  # wait_for_timeout(ms) calls (the settle pause)

    def on(self, event, handler):
        if event == "download":
            self.download_handlers.append(handler)

    def emit_download(self, d):
        for h in self.download_handlers:
            h(d)

    async def wait_for_timeout(self, ms):
        self.waits.append(ms)


async def test_scroll_scales_notches_to_pixels(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")
    await c.scroll(10, 20, scroll_x=0, scroll_y=3)
    assert ("move", 10, 20) in page.mouse.calls
    assert ("wheel", 0, 3 * _WHEEL_PX_PER_NOTCH) in page.mouse.calls


async def test_scroll_negative_notches_scale(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")
    await c.scroll(5, 6, scroll_x=-2, scroll_y=0)
    assert ("wheel", -2 * _WHEEL_PX_PER_NOTCH, 0) in page.mouse.calls


# --- web-SPA interaction tuning: mousedown->up click delay + post-action settle.
# Both default 0 (thick channel / tests). When set, the web SPA registers clicks
# on custom widgets and the loop screenshots a settled (not mid-transition) frame.


async def test_click_forwards_delay(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl", click_delay_ms=60)
    await c.click(10, 20)
    assert ("click", 10, 20, "left", 60) in page.mouse.calls


async def test_click_delay_defaults_zero(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")
    await c.click(1, 2, button="right")
    assert ("click", 1, 2, "right", 0) in page.mouse.calls


async def test_settle_pauses_after_mutating_actions(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl", settle_ms=400)
    await c.click(5, 5)
    await c.type("hi")
    await c.keypress(["ENTER"])
    await c.move(7, 7)
    await c.scroll(0, 0, scroll_y=1)
    assert page.waits == [400, 400, 400, 400, 400]


async def test_settle_zero_no_pause(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")  # settle_ms defaults 0
    await c.click(5, 5)
    await c.type("hi")
    assert page.waits == []


# --- download capture: same-page AND popup/new-tab (the context listener) -----


async def test_same_page_download_captured(tmp_path):
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")
    page.emit_download(object())  # export fires on the current page
    assert len(c._pending) == 1


async def test_popup_download_captured_via_context_listener(tmp_path):
    # Murex 'Download as CSV' can fire on a popup/child page. The context-level
    # "page" listener must wire a download handler onto every new page, else the
    # export is missed (proven live: page-only capture dropped popup downloads).
    page = _FakePage()
    c = PlaywrightComputer(page, tmp_path / "dl")
    popup = _FakePage(context=page.context)
    page.context.emit_page(popup)  # popup opens -> PlaywrightComputer wires its download
    popup.emit_download(object())  # download fires on the popup, not `page`
    assert len(c._pending) == 1


# --- wait_for_download: bound the race between the model's last action and the
# download event landing in `_pending` ---------------------------------------


async def test_wait_for_download_true_when_already_pending(tmp_path):
    c = PlaywrightComputer(_FakePage(), tmp_path / "dl")
    c._pending.append(object())  # a download already captured by the listener
    assert await c.wait_for_download(timeout=5) is True


async def test_wait_for_download_no_wait_returns_false(tmp_path):
    c = PlaywrightComputer(_FakePage(), tmp_path / "dl")
    assert await c.wait_for_download(timeout=0) is False


async def test_wait_for_download_detects_arrival_during_wait(tmp_path, monkeypatch):
    import asyncio

    c = PlaywrightComputer(_FakePage(), tmp_path / "dl")

    async def _sleep(_secs):
        c._pending.append(object())  # the download arrives mid-wait

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    assert await c.wait_for_download(timeout=5, poll=0.01) is True


async def test_wait_for_download_times_out(tmp_path, monkeypatch):
    import asyncio

    calls = {"n": 0}

    async def _sleep(_secs):
        calls["n"] += 1

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    c = PlaywrightComputer(_FakePage(), tmp_path / "dl")
    assert await c.wait_for_download(timeout=0.02, poll=0.01) is False
    assert calls["n"] >= 1
