"""Log in to a Murex web-UI environment once, persist the session.

Logging in once per environment and reusing the saved Playwright
`storage_state` across that environment's N trade contexts avoids N logins and
keeps each worker fast.

NOTE: the selectors below are PLACEHOLDERS. Fill them from the manual
walkthrough of the real Murex web UI (pre-build step 2 in the plan). If the
login page can't be reduced to stable selectors, set `use_cua_login=True` to
have the computer-use model perform the login instead.
"""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Browser

from ..config import Settings
from ..models import EnvName

# TODO: confirm against the real Murex login page.
USERNAME_SELECTOR = "input[name='username'], #username, input[type='text']"
PASSWORD_SELECTOR = "input[name='password'], #password, input[type='password']"
SUBMIT_SELECTOR = "button[type='submit'], #login, button:has-text('Log in')"
# A selector that is only present AFTER a successful login (home/dashboard).
LOGGED_IN_SELECTOR = "body"


async def login_and_save_state(
    browser: Browser, env: EnvName, settings: Settings, state_dir: Path
) -> Path:
    """Perform a deterministic form login and persist storage_state to disk.

    Returns the path to the saved storage_state JSON, reusable by worker
    contexts via `browser.new_context(storage_state=path)`.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{env.value}_state.json"

    context = await browser.new_context(
        viewport={"width": settings.display_width, "height": settings.display_height},
        # Match the worker contexts: on-prem Murex's self-signed cert would
        # otherwise abort this pre-auth navigation with ERR_CERT_AUTHORITY_INVALID.
        ignore_https_errors=settings.murex_ignore_https_errors,
    )
    page = await context.new_page()
    try:
        await page.goto(settings.url_for(env.value), wait_until="domcontentloaded")
        await page.fill(USERNAME_SELECTOR, settings.murex_user)
        await page.fill(PASSWORD_SELECTOR, settings.murex_pass.get_secret_value())
        await page.click(SUBMIT_SELECTOR)
        await page.wait_for_selector(LOGGED_IN_SELECTOR, timeout=30_000)
        await context.storage_state(path=str(state_path))
    finally:
        await context.close()

    return state_path
