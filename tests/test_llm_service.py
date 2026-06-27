import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.services.llm_client import LLMHTTPClient
from app.services.llm_service import LLMService


def _make_settings(provider: str = "ollama") -> Settings:
    return Settings(
        pg_dsn="postgresql://test:test@localhost/test",
        llm_provider=provider,
        openrouter_api_key="sk-test" if provider == "openrouter" else None,
    )


def _mock_transport(
    *responses: httpx.Response,
) -> httpx.MockTransport:
    """Return a MockTransport that serves the given responses in order."""
    queue: list[httpx.Response] = list(responses)

    async def handler(request: httpx.Request) -> httpx.Response:
        return queue.pop(0)

    return httpx.MockTransport(handler)


def _make_service(
    provider: str = "ollama",
    responses: list[httpx.Response] | None = None,
) -> tuple[LLMService, list[httpx.Response]]:
    if responses is None:
        responses = []
    settings = _make_settings(provider)
    transport = _mock_transport(*responses)
    base_url = (
        settings.ollama_base_url if provider == "ollama"
        else settings.openrouter_base_url
    )
    api_key = settings.openrouter_api_key if provider == "openrouter" else ""
    http_client = LLMHTTPClient(base_url=base_url, api_key=api_key, transport=transport)
    service = LLMService(settings=settings, http_client=http_client)
    return service, responses


# ------------------------------------------------------------------ complete tests

@pytest.mark.asyncio
async def test_complete_ollama_request_shape() -> None:
    resp = httpx.Response(
        200,
        json={"message": {"role": "assistant", "content": "Hello!"}},
    )
    service, _ = _make_service("ollama", [resp])

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="llama3",
        temperature=0.5,
        max_tokens=100,
    )
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_complete_openrouter_request_shape() -> None:
    resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Hi there"}}]},
    )
    service, _ = _make_service("openrouter", [resp])

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4",
        temperature=0.5,
        max_tokens=200,
        response_format="json_object",
    )
    assert result == "Hi there"


# ------------------------------------------------------------------ embed tests

@pytest.mark.asyncio
async def test_embed_ollama_batch_loop() -> None:
    r1 = httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    r2 = httpx.Response(200, json={"embedding": [0.4, 0.5, 0.6]})
    service, _ = _make_service("ollama", [r1, r2])

    result = await service.embed(inputs=["text a", "text b"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_embed_openrouter_batch() -> None:
    resp = httpx.Response(
        200,
        json={"data": [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]},
    )
    service, _ = _make_service("openrouter", [resp])

    result = await service.embed(inputs=["text a", "text b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


# ------------------------------------------------------------------ retry tests

@pytest.mark.asyncio
async def test_retries_on_503_then_succeeds() -> None:
    """3 attempts total: 503, 503, 200."""
    r1 = httpx.Response(503, json={"error": "overloaded"})
    r2 = httpx.Response(503, json={"error": "overloaded"})
    r3 = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    service, _ = _make_service("openrouter", [r1, r2, r3])

    result = await service.complete(messages=[{"role": "user", "content": "hi"}])
    assert result == "ok"


@pytest.mark.asyncio
async def test_raises_llm_provider_error_after_max_retries() -> None:
    r1 = httpx.Response(503, json={"error": "overloaded"})
    r2 = httpx.Response(503, json={"error": "overloaded"})
    r3 = httpx.Response(503, json={"error": "overloaded"})
    service, _ = _make_service("openrouter", [r1, r2, r3, httpx.Response(503)])

    with pytest.raises(LLMProviderError) as exc_info:
        await service.complete(messages=[{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_no_retry_on_401() -> None:
    """4xx errors should NOT be retried."""
    r1 = httpx.Response(401, json={"error": "unauthorized"})
    r2 = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    service, _ = _make_service("openrouter", [r1, r2])

    with pytest.raises(LLMProviderError) as exc_info:
        await service.complete(messages=[{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 401


# --------------------------------------------------------------- _is_transient

def test_is_transient_5xx() -> None:
    assert LLMService._is_transient(500, None) is True
    assert LLMService._is_transient(503, None) is True
    assert LLMService._is_transient(599, None) is True
    assert LLMService._is_transient(400, None) is False
    assert LLMService._is_transient(401, None) is False
    assert LLMService._is_transient(429, None) is False


def test_is_transient_timeout() -> None:
    assert LLMService._is_transient(None, httpx.TimeoutException("timeout")) is True


def test_is_transient_request_error() -> None:
    assert LLMService._is_transient(None, httpx.RequestError("conn reset")) is True
