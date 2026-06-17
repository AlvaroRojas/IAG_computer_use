"""Provider-neutral agent-backend seam.

An `AgentBackend` owns the model client + model id and runs the computer-use loop
against a `Computer`. `simulate.py` calls `backend.run(...)` so the unit of work is
identical regardless of model provider. `build_backend()` picks the implementation
from `settings.cua_provider`; the per-provider SDK + client are imported LAZILY so
an OpenAI-only run never constructs an Anthropic client (and vice versa).
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from ..config import Settings
from .base import Computer
from .loop import LoopResult
from .trace import NullTracer, Tracer

SafetyHandler = Callable[[dict], Awaitable[bool]]


class AgentBackend(Protocol):
    """One model provider's computer-use loop, behind a provider-neutral call."""

    async def run(
        self,
        *,
        computer: Computer,
        task: str,
        display_width: int,
        display_height: int,
        environment: str = "browser",
        max_turns: int = 60,
        on_safety_check: SafetyHandler | None = None,
        tracer: Tracer | NullTracer | None = None,
    ) -> LoopResult: ...


def build_backend(settings: Settings) -> AgentBackend:
    """Construct the AgentBackend for the configured provider (fail-fast creds were
    already validated by Settings; the checks here are defensive)."""
    provider = settings.cua_provider

    if provider == "openai":
        from openai import AsyncOpenAI

        from .openai_backend import OpenAIBackend

        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=settings.cua_max_retries,
        )
        return OpenAIBackend(
            client=client,
            model=settings.cua_model,
            reasoning_effort=settings.cua_reasoning_effort,
        )

    if provider == "bedrock-openai":
        # GPT-5.x on AWS Bedrock via the OpenAI-compatible "mantle" Responses endpoint.
        # Same bearer token as the Anthropic Bedrock path; the SDK sends it as
        # `Authorization: Bearer` to `cua_openai_base_url`. No native computer tool is
        # available there, so this backend emulates it with a custom function tool.
        from openai import AsyncOpenAI

        from .openai_custom_backend import OpenAICustomToolBackend

        token = settings.aws_bearer_token_bedrock
        if token is None:
            raise ValueError(
                "AWS_BEARER_TOKEN_BEDROCK is required for the bedrock-openai provider"
            )
        if not settings.cua_openai_base_url:
            raise ValueError(
                "CUA_OPENAI_BASE_URL is required for the bedrock-openai provider"
            )
        client = AsyncOpenAI(
            base_url=settings.cua_openai_base_url,
            api_key=token.get_secret_value(),
            max_retries=settings.cua_max_retries,
        )
        return OpenAICustomToolBackend(
            client=client,
            model=settings.cua_model,
            reasoning_effort=settings.cua_reasoning_effort,
            max_output_tokens=settings.cua_max_tokens,
            keep_last_screenshots=settings.cua_keep_last_screenshots,
            transient_retries=settings.cua_max_retries,
            prompt_cache_retention=settings.cua_openai_prompt_cache_retention,
        )

    # "anthropic" and "bedrock" share the Messages-API backend; only the client differs.
    from .anthropic_backend import AnthropicBackend

    if provider == "bedrock":
        from anthropic import AsyncAnthropicBedrock

        token = settings.aws_bearer_token_bedrock
        client = AsyncAnthropicBedrock(
            aws_region=settings.aws_region,
            # Explicit bearer token (Bedrock API key); AsyncAnthropicBedrock sends it
            # as `Authorization: Bearer` and skips SigV4. Falls back to the env var
            # AWS_BEARER_TOKEN_BEDROCK when None.
            api_key=token.get_secret_value() if token else None,
            # Ride out transient 503 capacity dips (common on Opus on-demand) with
            # the SDK's exponential backoff instead of failing the whole session.
            max_retries=settings.cua_max_retries,
        )
    else:  # direct anthropic
        from anthropic import AsyncAnthropic

        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=settings.cua_max_retries,
        )

    return AnthropicBackend(
        client=client,
        model=settings.cua_model,
        tool_version=settings.cua_anthropic_tool_version,
        beta_flag=settings.cua_anthropic_beta,
        max_tokens=settings.cua_max_tokens,
        keep_last_screenshots=settings.cua_keep_last_screenshots,
        reasoning_effort=settings.cua_reasoning_effort,
        prompt_cache=settings.cua_prompt_cache,
    )
