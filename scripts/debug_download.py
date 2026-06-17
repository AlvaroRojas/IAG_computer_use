"""Isolated proof of the web-channel download capture, independent of Murex.

Builds the SAME context shape browser.py uses (accept_downloads + ignore_https_errors),
wraps the page in the real PlaywrightComputer, and fires two download styles:
  1. same-page  — <a download> click on the current page
  2. popup      — window.open(...) to a page that auto-clicks an <a download>
Then runs collect-export semantics (wait_for_download -> flush_downloads) and reports
what was captured. This tells us whether the page-scoped page.on("download") listener
misses popup-originated downloads (the realistic Murex 'File -> Download as CSV' case).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

from iag_sim.cua.computer import PlaywrightComputer

CSV = "col1;col2%0A1;2%0A3;4"  # url-encoded 2-row semicolon CSV

SAME_PAGE_HTML = (
    "<html><body><a id='dl' href=\"data:text/csv," + CSV + "\" "
    "download='same.csv'>download</a></body></html>"
)

# Opener page: a button that window.open()s a child which immediately clicks an
# <a download>. Mimics a SPA that exports via a popup/new tab.
CHILD_HTML = (
    "<html><body><a id='c' href=\"data:text/csv," + CSV + "\" download='popup.csv'>"
    "x</a><script>document.getElementById('c').click()</script></body></html>"
)


async def _scenario(label: str, setup) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=True,
            ignore_https_errors=True,
        )
        page = await context.new_page()
        with tempfile.TemporaryDirectory() as d:
            comp = PlaywrightComputer(page, Path(d))
            await setup(context, page)
            got = await comp.wait_for_download(timeout=5.0, poll=0.2)
            saved = await comp.flush_downloads()
            names = [pth.name for pth in saved]
            on_disk = [f.name for f in Path(d).glob("*")]
            print(f"[{label}] wait_for_download={got} saved={names} on_disk={on_disk}")
        await context.close()
        await browser.close()


async def _same_page(context, page):
    await page.set_content(SAME_PAGE_HTML)
    await page.click("#dl")
    await page.wait_for_timeout(500)


async def _popup(context, page):
    await page.set_content("<html><body>opener</body></html>")
    # window.open to a child that auto-downloads. The download fires on the CHILD
    # page, not `page` — page.on('download') will miss it if it's page-scoped.
    await page.evaluate(
        "() => { const w = window.open('about:blank'); "
        "w.document.write(" + repr(CHILD_HTML) + "); }"
    )
    await page.wait_for_timeout(800)


async def main():
    await _scenario("same-page", _same_page)
    await _scenario("popup", _popup)


asyncio.run(main())
