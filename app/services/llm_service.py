import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_request_id
from app.services.llm_client import LLMHTTPClient

_MAX_RETRIES = 3
_PROMPT_TRUNCATE_LEN = 500


def _truncate_prompt(text: str) -> str:
    if len(text) > _PROMPT_TRUNCATE_LEN:
        return text[:_PROMPT_TRUNCATE_LEN]
    return text


class LLMService:
    def __init__(self, settings: Settings, http_client: LLMHTTPClient) -> None:
        self._settings = settings
        self._http = http_client

    # ------------------------------------------------------------------ public

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
        request_id: str | None = None,
    ) -> str:
        req_id = request_id or get_request_id()
        prompt_repr = _truncate_prompt(repr(messages))
        is_truncated = len(repr(messages)) > _PROMPT_TRUNCATE_LEN

        logger.bind(
            event="llm_complete_start",
            request_id=req_id,
            _truncated=is_truncated,
        ).info(prompt_repr)

        t0 = time.perf_counter()
        try:
            result = await self._complete_provider(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                request_id=req_id,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                provider=self._settings.llm_provider,
                status_code=0,
                payload=str(exc),
                request_id=req_id,
            ) from exc

        elapsed = (time.perf_counter() - t0) * 1000
        logger.bind(
            event="llm_complete_done",
            request_id=req_id,
            duration_ms=round(elapsed, 1),
        ).info("complete finished in {:.1f} ms", elapsed)

        return result

    async def embed(
        self,
        inputs: list[str],
        *,
        model: str | None = None,
        request_id: str | None = None,
    ) -> list[list[float]]:
        req_id = request_id or get_request_id()

        logger.bind(
            event="llm_embed_start",
            request_id=req_id,
            input_count=len(inputs),
        ).info("embedding {} inputs", len(inputs))

        t0 = time.perf_counter()
        try:
            result = await self._embed_provider(
                inputs=inputs,
                model=model,
                request_id=req_id,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                provider=self._settings.llm_provider,
                status_code=0,
                payload=str(exc),
                request_id=req_id,
            ) from exc

        elapsed = (time.perf_counter() - t0) * 1000
        logger.bind(
            event="llm_embed_done",
            request_id=req_id,
            duration_ms=round(elapsed, 1),
        ).info("embed finished in {:.1f} ms", elapsed)

        return result

    # ------------------------------------------------------------ provider dispatch

    async def _complete_provider(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        response_format: str | None,
        request_id: str,
    ) -> str:
        if self._settings.llm_provider == "ollama":
            return await self._complete_ollama(
                messages, model, temperature, max_tokens, request_id
            )
        return await self._complete_openrouter(
            messages, model, temperature, max_tokens, response_format, request_id
        )

    async def _embed_provider(
        self,
        inputs: list[str],
        model: str | None,
        request_id: str,
    ) -> list[list[float]]:
        if self._settings.llm_provider == "ollama":
            return await self._embed_ollama(inputs, model, request_id)
        return await self._embed_openrouter(inputs, model, request_id)

    # ---------------------------------------------------------- Ollama

    async def _complete_ollama(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        request_id: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self._settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        resp = await self._retry_with_backoff(
            lambda: self._http.post("/api/chat", payload),
            request_id,
        )
        return resp["message"]["content"]

    async def _embed_ollama(
        self,
        inputs: list[str],
        model: str | None,
        request_id: str,
    ) -> list[list[float]]:
        model = model or self._settings.embedding_model
        embeddings: list[list[float]] = []
        for idx, text in enumerate(inputs):
            resp = await self._retry_with_backoff(
                lambda t=text: self._http.post(
                    "/api/embeddings", {"model": model, "input": t}
                ),
                request_id,
            )
            embeddings.append(resp["embedding"])
        return embeddings

    # ---------------------------------------------------------- OpenRouter

    async def _complete_openrouter(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        response_format: str | None,
        request_id: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self._settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        resp = await self._retry_with_backoff(
            lambda: self._http.post("/chat/completions", payload),
            request_id,
        )
        return resp["choices"][0]["message"]["content"]

    async def _embed_openrouter(
        self,
        inputs: list[str],
        model: str | None,
        request_id: str,
    ) -> list[list[float]]:
        payload = {
            "model": model or self._settings.embedding_model,
            "input": inputs,
        }
        resp = await self._retry_with_backoff(
            lambda: self._http.post("/embeddings", payload),
            request_id,
        )
        return [d["embedding"] for d in resp["data"]]

    # -------------------------------------------------------- retry

    @staticmethod
    def _is_transient(
        status_code: int | None, exception: Exception | None
    ) -> bool:
        if exception is not None:
            if isinstance(exception, httpx.TimeoutException):
                return True
            if isinstance(exception, httpx.RequestError):
                return True
        if status_code is not None and 500 <= status_code < 600:
            return True
        return False

    async def _retry_with_backoff(
        self,
        fn: Any,
        request_id: str,
    ) -> dict:
        last_status = 0
        last_payload = ""

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await fn()
            except httpx.HTTPStatusError as e:
                last_status = e.response.status_code
                last_payload = e.response.text
                if not self._is_transient(last_status, None):
                    raise LLMProviderError(
                        provider=self._settings.llm_provider,
                        status_code=last_status,
                        payload=last_payload,
                        request_id=request_id,
                    ) from e
                logger.warning(
                    "LLM HTTP {status} (transient), attempt {attempt}/{max}",
                    status=last_status,
                    attempt=attempt,
                    max=_MAX_RETRIES,
                )
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_status = 0
                last_payload = str(e)
                if not self._is_transient(None, e):
                    raise LLMProviderError(
                        provider=self._settings.llm_provider,
                        status_code=0,
                        payload=last_payload,
                        request_id=request_id,
                    ) from e
                logger.warning(
                    "LLM HTTP error (transient), attempt {attempt}/{max}: {err}",
                    attempt=attempt,
                    max=_MAX_RETRIES,
                    err=str(e),
                )

            if attempt < _MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                await asyncio.sleep(wait)

        raise LLMProviderError(
            provider=self._settings.llm_provider,
            status_code=last_status,
            payload=last_payload,
            request_id=request_id,
        )
