import json
from io import StringIO

from loguru import logger

from app.core.config import Settings
from app.core.logging import (
    bind_request_id,
    configure_logging,
    get_request_id,
)


def test_text_format() -> None:
    s = Settings(pg_dsn="postgresql://test:test@localhost/test")
    stream = StringIO()
    logger.remove()
    configure_logging(s)
    logger.remove()
    logger.add(stream, format="{message}", level="INFO")
    bind_request_id("test-001")
    logger.info("hello")
    assert "hello" in stream.getvalue()


def test_request_id_bind_and_get() -> None:
    bind_request_id("")
    assert get_request_id() == ""
    bind_request_id("req-abc")
    assert get_request_id() == "req-abc"
    bind_request_id("req-xyz")
    assert get_request_id() == "req-xyz"


def test_json_format_output() -> None:
    s = Settings(
        pg_dsn="postgresql://test:test@localhost/test", log_format="json"
    )
    stream = StringIO()
    logger.remove()
    configure_logging(s)
    logger.remove()
    logger.add(stream, serialize=True, level="INFO")
    bind_request_id("json-test")
    logger.info("hello")
    output = stream.getvalue().strip()
    record = json.loads(output)
    assert record["record"]["extra"]["request_id"] == "json-test"


def test_patcher_injects_request_id() -> None:
    s = Settings(pg_dsn="postgresql://test:test@localhost/test")
    stream = StringIO()
    logger.remove()
    configure_logging(s)
    logger.remove()
    logger.add(
        stream,
        format="{extra[request_id]} | {message}",
        level="INFO",
    )
    logger.configure(
        patcher=lambda r: r["extra"].update(request_id=get_request_id())
    )
    bind_request_id("patcher-test")
    logger.info("hi")
    output = stream.getvalue().strip()
    assert output.startswith("patcher-test | hi")
