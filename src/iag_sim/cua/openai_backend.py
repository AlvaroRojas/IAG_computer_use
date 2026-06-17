"""OpenAI computer-use backend: a thin AgentBackend wrapper over run_cua_loop.

Keeps the existing Responses-API loop (loop.py) untouched and exposes it through
the provider-neutral backend seam (backend.py) so simulate.py can drive any
provider the same way.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from .base import Computer
from .loop import LoopResult, SafetyHandler, _deny, run_cua_loop
from .trace import NullTracer, Tracer


class OpenAIBackend:
    """AgentBackend over the OpenAI Responses `computer` tool."""

    def __init__(
        self, *, client: AsyncOpenAI, model: str, reasoning_effort: str | None = None
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def run(
        self,
        *,
        computer: Computer,
        task: str,
        display_width: int,
        display_height: int,
        environment: str = "browser",
        max_turns: int = 60,
        on_safety_check: SafetyHandler = _deny,
        tracer: Tracer | NullTracer | None = None,
    ) -> LoopResult:
        return await run_cua_loop(
            client=self.client,
            computer=computer,
            model=self.model,
            task=task,
            display_width=display_width,
            display_height=display_height,
            environment=environment,
            max_turns=max_turns,
            on_safety_check=on_safety_check,
            reasoning_effort=self.reasoning_effort,
            tracer=tracer,
        )
