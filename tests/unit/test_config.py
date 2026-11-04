"""Settings: AMG_ environment mapping, fail-closed validation and secret handling."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from agent_memory_graph.config import DEFAULT_EMBEDDING_DIM, Settings


def settings(**overrides: object) -> Settings:
    return Settings.model_validate({"neo4j_password": "pw", **overrides})


def test_defaults() -> None:
    config = settings()
    assert config.neo4j_uri == "neo4j://localhost:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_database is None
    assert config.qdrant_url == "http://localhost:6333"
    assert config.qdrant_api_key is None
    assert config.qdrant_collection == "amg_entities"
    assert config.embedding_dim == DEFAULT_EMBEDDING_DIM == 384
    assert config.default_timeout_seconds == 10.0


def test_password_is_required() -> None:
    with pytest.raises(ValidationError, match="neo4j_password"):
        Settings()


def test_reads_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "AMG_NEO4J_URI": "bolt+s://graph.example:7687",
        "AMG_NEO4J_USER": "memory",
        "AMG_NEO4J_PASSWORD": "from-env",
        "AMG_NEO4J_DATABASE": "agents",
        "AMG_QDRANT_URL": "https://vectors.example",
        "AMG_QDRANT_API_KEY": "key-from-env",
        "AMG_EMBEDDING_DIM": "1536",
        "AMG_DEFAULT_TIMEOUT_SECONDS": "2.5",
        "AMG_NEO4J_MAX_POOL_SIZE": "8",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = Settings()
    assert config.neo4j_uri == "bolt+s://graph.example:7687"
    assert config.neo4j_password.get_secret_value() == "from-env"
    assert config.qdrant_api_key == SecretStr("key-from-env")
    assert (config.embedding_dim, config.default_timeout_seconds) == (1536, 2.5)
    assert config.neo4j_max_pool_size == 8


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "wrong-prefix")
    with pytest.raises(ValidationError):
        Settings()


def test_dotenv_files_are_not_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("AMG_NEO4J_PASSWORD=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_masked() -> None:
    config = settings(neo4j_password="hunter2", qdrant_api_key="key-123", qdrant_url="https://q")
    rendered = repr(config) + str(config.model_dump())
    assert "hunter2" not in rendered
    assert "key-123" not in rendered


def test_settings_are_frozen_and_closed() -> None:
    config = settings()
    with pytest.raises(ValidationError):
        config.embedding_dim = 1  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        settings(unknown_option=True)


@pytest.mark.parametrize(
    "uri",
    ["neo4j://h", "neo4j+s://h:7687", "neo4j+ssc://h", "bolt://h", "bolt+s://h", "bolt+ssc://h"],
)
def test_supported_neo4j_schemes(uri: str) -> None:
    assert settings(neo4j_uri=uri).neo4j_uri == uri


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("neo4j_uri", "http://h:7687", "scheme"),
        ("neo4j_uri", "neo4j://", "host"),
        ("neo4j_uri", "neo4j://user:pw@h:7687", "credentials"),
        ("qdrant_url", "ftp://h", "scheme"),
        ("qdrant_url", "https://key@h", "credentials"),
        ("qdrant_url", "http:///path", "host"),
        ("qdrant_collection", "bad name", "pattern"),
        ("qdrant_collection", "_hidden", "pattern"),
        ("embedding_dim", 0, "greater than or equal"),
        ("embedding_dim", 8193, "less than or equal"),
        ("default_timeout_seconds", 0, "greater than"),
        ("default_timeout_seconds", float("inf"), "finite"),
        ("neo4j_user", "", "at least 1"),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        settings(**{field: value})


@pytest.mark.parametrize(
    "url", ["http://localhost:6333", "http://127.0.0.1:6333", "http://[::1]:6333", "https://q"]
)
def test_api_key_allowed_over_tls_or_loopback(url: str) -> None:
    assert settings(qdrant_url=url, qdrant_api_key="k").qdrant_api_key is not None


@pytest.mark.parametrize("url", ["http://qdrant:6333", "http://10.0.0.5:6333"])
def test_api_key_refused_over_plaintext_to_remote_hosts(url: str) -> None:
    with pytest.raises(ValidationError, match="requires an https"):
        settings(qdrant_url=url, qdrant_api_key="k")


def test_plaintext_without_api_key_is_allowed() -> None:
    assert settings(qdrant_url="http://qdrant:6333").qdrant_url == "http://qdrant:6333"


def test_guardrail_defaults() -> None:
    config = settings()
    assert (config.max_concurrent_extractions, config.max_concurrent_store_operations) == (4, 32)
    assert (config.queue_timeout_seconds, config.breaker_failure_threshold) == (2.0, 5)
    assert config.breaker_reset_seconds == 30.0
    assert config.input_guard_mode == "flag"
    assert (config.retrieval_fallback_steps, config.retrieval_fallback_floor) == (3, 0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_concurrent_extractions", 0),
        ("queue_timeout_seconds", 0),
        ("breaker_failure_threshold", 0),
        ("breaker_reset_seconds", float("nan")),
        ("input_guard_mode", "block"),
        ("retrieval_fallback_floor", 1.5),
    ],
)
def test_invalid_guardrail_settings(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        settings(**{field: value})
