import sys
from contextvars import ContextVar

from loguru import logger

from app.core.config import Settings

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def _patch_record(record: dict) -> None:
    record["extra"]["request_id"] = _request_id_var.get()


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.configure(patcher=_patch_record)

    if settings.log_format == "json":
        logger.add(
            sys.stderr,
            serialize=True,
            level="INFO",
        )
    else:
        logger.add(
            sys.stderr,
            format=(
                "timestamp={time:YYYY-MM-DDTHH:mm:ss.SSS} "
                "level={level} "
                "request_id={extra[request_id]} "
                "| {message}"
            ),
            level="INFO",
        )


def bind_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()
