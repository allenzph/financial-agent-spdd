class LLMProviderError(Exception):
    def __init__(
        self,
        provider: str,
        status_code: int,
        payload: str,
        request_id: str,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.payload = payload
        self.request_id = request_id
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"LLMProviderError(provider={self.provider}, "
            f"status_code={self.status_code}, "
            f"payload={self.payload!r}, "
            f"request_id={self.request_id})"
        )


class LLMOutputValidationError(Exception):
    def __init__(
        self,
        raw_output: str,
        request_id: str,
    ) -> None:
        self.raw_output = raw_output
        self.request_id = request_id
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"LLMOutputValidationError(raw_output={self.raw_output!r}, "
            f"request_id={self.request_id})"
        )
