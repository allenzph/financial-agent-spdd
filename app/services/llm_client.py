import httpx
from loguru import logger


class LLMHTTPClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
        )

    async def post(self, endpoint: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        redacted = self._redact_headers(headers)
        logger.bind(event="llm_http_request", headers=redacted).debug(
            "POST {}{}", self.base_url, endpoint
        )
        response = await self._client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _redact_headers(headers: dict) -> dict:
        redacted = dict(headers)
        for key in redacted:
            if key.lower() == "authorization":
                redacted[key] = "REDACTED"
        return redacted
