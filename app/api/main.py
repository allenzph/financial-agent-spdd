import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from app.core.config import get_settings
from app.core.logging import bind_request_id, configure_logging
from app.core.services_container import ServicesContainer
from app.services.llm_client import LLMHTTPClient
from app.services.llm_service import LLMService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)

    if settings.llm_provider == "ollama":
        base_url = settings.ollama_base_url
        api_key = ""
    else:
        base_url = settings.openrouter_base_url
        api_key = settings.openrouter_api_key or ""

    http_client = LLMHTTPClient(base_url=base_url, api_key=api_key)
    llm_service = LLMService(settings=settings, http_client=http_client)
    app.state.services = ServicesContainer(
        settings=settings, llm_service=llm_service
    )

    yield

    await http_client.aclose()


app = FastAPI(title="Financial Helpdesk Agent", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    bind_request_id(request_id)
    response: Response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    services: ServicesContainer | None = request.app.state.services
    if services is None:
        raise HTTPException(status_code=503, detail="services not initialized")
    return {"status": "ok", "llm_provider": services.settings.llm_provider}
