import pytest

from app.core.config import Settings, get_settings


def test_defaults() -> None:
    s = Settings(pg_dsn="postgresql://test:test@localhost/test")
    assert s.llm_provider == "ollama"
    assert s.log_format == "text"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_model == "gpt-4.1-mini"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_chat_model == "gemma3:27b"
    assert s.ollama_ops_model == "qwen3.5:4b"
    assert s.embedding_model == "nomic-embed-text"
    assert s.embedding_dim == 768


def test_openrouter_without_key_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Settings(
            pg_dsn="postgresql://test:test@localhost/test",
            llm_provider="openrouter",
        )


def test_openrouter_with_key_ok() -> None:
    s = Settings(
        pg_dsn="postgresql://test:test@localhost/test",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
    )
    assert s.llm_provider == "openrouter"
    assert s.openrouter_api_key == "sk-test"


def test_ollama_without_key_ok() -> None:
    s = Settings(pg_dsn="postgresql://test:test@localhost/test")
    assert s.llm_provider == "ollama"
    assert s.openrouter_api_key is None


def test_repr_does_not_leak_api_key() -> None:
    s = Settings(
        pg_dsn="postgresql://test:test@localhost/test",
        llm_provider="openrouter",
        openrouter_api_key="sk-secret-123",
    )
    r = repr(s)
    assert "sk-secret-123" not in r
    assert "openrouter_api_key" not in r


def test_get_settings_lru_cache(monkeypatch) -> None:
    monkeypatch.setenv("PG_DSN", "postgresql://test:test@localhost/test")
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()
