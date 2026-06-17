"""Non-serializable runtime resources shared by workers (harnesses, model-provider
backend, concurrency gate). Created once per run via `open_resources` and passed
to engines through closures — never placed in checkpointed graph state.

A harness is built per environment from its channel:
  - "web"   -> BrowserHarness  (needs a shared Playwright Chromium)
  - "thick" -> DockerHarness   (one Linux container per trade)
Playwright is only launched if at least one environment uses the web channel.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..cua.backend import AgentBackend, build_backend
from ..harness.base import Harness
from ..harness.browser import BrowserHarness
from ..harness.docker import DockerHarness
from ..models import EnvName


@dataclass
class Resources:
    settings: Settings
    backend: AgentBackend
    run_dir: Path
    # PER-ENVIRONMENT concurrency budgets: "before" and "after" each get their
    # own semaphore sized to MAX_CONCURRENCY, so a slow side never starves the
    # other and the two environments scale independently (total in-flight can be
    # up to 2 x MAX_CONCURRENCY).
    semaphores: dict[str, asyncio.Semaphore]
    harnesses: dict[str, Harness] = field(default_factory=dict)

    def harness_for(self, env: EnvName) -> Harness:
        return self.harnesses[env.value]

    def semaphore_for(self, env: EnvName) -> asyncio.Semaphore:
        return self.semaphores[env.value]


@asynccontextmanager
async def open_resources(settings: Settings, run_dir: Path):
    """Build harnesses for both environments, log in / verify, and yield
    Resources. Everything is torn down on exit."""
    run_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(settings)

    channels = {env: settings.channel_for(env.value) for env in (EnvName.BEFORE, EnvName.AFTER)}
    needs_browser = "web" in channels.values()

    async with AsyncExitStack() as stack:
        browser = None
        if needs_browser:
            from playwright.async_api import async_playwright

            pw = await stack.enter_async_context(async_playwright())
            # Bound each Chromium renderer's V8 heap. Contexts share one browser
            # process, so this is the closest Playwright analogue to the thick
            # container's --memory; CPU has no per-session lever here.
            launch_args: list[str] = []
            if settings.playwright_max_memory_mb:
                launch_args.append(
                    f"--js-flags=--max-old-space-size={settings.playwright_max_memory_mb}"
                )
            browser = await pw.chromium.launch(
                headless=settings.headless, args=launch_args
            )
            stack.push_async_callback(browser.close)

        harnesses: dict[str, Harness] = {}
        for env, channel in channels.items():
            if channel == "web":
                assert browser is not None
                harnesses[env.value] = BrowserHarness(env, settings, browser, run_dir)
            else:
                harnesses[env.value] = DockerHarness(env, settings, run_dir)

        for harness in harnesses.values():
            await harness.setup()
            stack.push_async_callback(harness.aclose)

        yield Resources(
            settings=settings,
            backend=backend,
            run_dir=run_dir,
            semaphores={
                env.value: asyncio.Semaphore(settings.effective_concurrency())
                for env in (EnvName.BEFORE, EnvName.AFTER)
            },
            harnesses=harnesses,
        )
