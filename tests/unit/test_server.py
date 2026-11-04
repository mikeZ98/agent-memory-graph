"""Process entrypoint: CLI/env configuration, exit codes, signals, composition."""

import logging
import runpy
import signal
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from agent_memory_graph import server
from agent_memory_graph.config import Settings
from agent_memory_graph.domain import ExtractionFailedError
from agent_memory_graph.services.extractor import GraphExtraction, InstructorStructuredLLM


class FakeServer:
    """Records the uvicorn config; ``run`` optionally delivers a shutdown signal."""

    instances: ClassVar[list["FakeServer"]] = []

    def __init__(
        self, config: uvicorn.Config, *, start: bool = True, deliver: int | None = None
    ) -> None:
        self.config = config
        self.started = False
        self._start = start
        self._deliver = deliver
        FakeServer.instances.append(self)

    def run(self) -> None:
        self.started = self._start
        if self._deliver is not None:
            handler = signal.getsignal(self._deliver)
            assert callable(handler)
            handler(self._deliver, None)


def factory(**kwargs: Any) -> Callable[[uvicorn.Config], FakeServer]:
    return lambda config: FakeServer(config, **kwargs)


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    FakeServer.instances.clear()
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class TestMain:
    def test_defaults_bind_loopback_with_the_production_factory(self) -> None:
        assert server.main([], server_factory=factory()) == 0
        (instance,) = FakeServer.instances
        config = instance.config
        assert (config.app, config.factory) == (server.DEFAULT_FACTORY, True)
        assert (config.host, config.port, config.uds) == ("127.0.0.1", 8000, None)
        assert config.lifespan == "on"
        assert config.access_log is False
        assert config.server_header is False
        assert config.proxy_headers is False
        assert config.timeout_graceful_shutdown == 10.0
        assert config.log_level == "info"

    def test_cli_flags(self) -> None:
        argv = [
            "--host", "0.0.0.0", "--port", "9000", "--factory", "pkg.mod:make",  # noqa: S104
            "--log-level", "warning", "--graceful-timeout", "2.5",
        ]  # fmt: skip
        server.main(argv, server_factory=factory())
        config = FakeServer.instances[0].config
        assert (config.host, config.port, config.app) == ("0.0.0.0", 9000, "pkg.mod:make")  # noqa: S104
        assert (config.log_level, config.timeout_graceful_shutdown) == ("warning", 2.5)

    def test_unix_socket(self) -> None:
        server.main(["--uds", ".cache/e2e/x.sock"], server_factory=factory())
        assert FakeServer.instances[0].config.uds == ".cache/e2e/x.sock"

    def test_environment_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMG_HTTP_HOST", "0.0.0.0")  # noqa: S104
        monkeypatch.setenv("AMG_HTTP_PORT", "8443")
        monkeypatch.setenv("AMG_LOG_LEVEL", "debug")
        monkeypatch.setenv("AMG_GRACEFUL_TIMEOUT_SECONDS", "30")
        server.main([], server_factory=factory())
        config = FakeServer.instances[0].config
        assert (config.host, config.port, config.log_level) == ("0.0.0.0", 8443, "debug")  # noqa: S104
        assert config.timeout_graceful_shutdown == 30.0

    def test_failed_startup_exits_3(self) -> None:
        assert server.main([], server_factory=factory(start=False)) == server.STARTUP_FAILURE

    @pytest.mark.parametrize(
        ("sig", "name"), [(signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")]
    )
    def test_shutdown_signal_after_graceful_stop_exits_0(
        self,
        sig: int,
        name: str,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # uvicorn.Config() re-applies its dictConfig (replacing handlers on uvicorn.error),
        # so route the launcher's messages to a propagating test logger instead.
        monkeypatch.setattr(server, "logger", logging.getLogger("amg.test.server"))
        with caplog.at_level("INFO", logger="amg.test.server"):
            assert server.main([], server_factory=factory(deliver=sig)) == 0
        assert f"graceful shutdown complete (signals: {name})" in caplog.text

    def test_invalid_arguments_exit_2(self) -> None:
        with pytest.raises(SystemExit) as exc:
            server.main(["--log-level", "loud"], server_factory=factory())
        assert exc.value.code == 2

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "main", lambda: 0)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("agent_memory_graph", run_name="__main__")
        assert exc.value.code == 0


class TestComposition:
    async def test_unconfigured_llm_fails_ingestion_with_a_domain_error(self) -> None:
        llm = server.llm_from_settings(Settings.model_validate({"neo4j_password": "pw"}))
        assert isinstance(llm, server.UnconfiguredLLM)
        with pytest.raises(ExtractionFailedError, match="set AMG_LLM_MODEL"):
            await llm.create(response_model=GraphExtraction, messages=[], max_retries=0)

    def test_configured_llm_uses_instructor_from_provider(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        client = object()

        def from_provider(model: str, **kwargs: Any) -> object:
            calls.append((model, kwargs))
            return client

        settings = Settings.model_validate(
            {
                "neo4j_password": "pw",
                "llm_model": "anthropic/claude-opus-5-5",
                "llm_max_tokens": 4096,
            }
        )
        llm = server.llm_from_settings(settings, from_provider=from_provider)
        assert calls == [("anthropic/claude-opus-5-5", {"async_client": True})]
        assert isinstance(llm, InstructorStructuredLLM)
        assert (llm._client, llm._model, llm._max_tokens) == (client, "claude-opus-5-5", 4096)

    def test_default_provider_lookup_is_instructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import instructor  # noqa: PLC0415

        seen: list[str] = []
        monkeypatch.setattr(instructor, "from_provider", lambda model, **_: seen.append(model))
        settings = Settings.model_validate({"neo4j_password": "pw", "llm_model": "openai/m"})
        server.llm_from_settings(settings)
        assert seen == ["openai/m"]

    def test_default_app_is_built_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMG_NEO4J_PASSWORD", "pw")
        monkeypatch.setenv("AMG_EMBEDDING_DIM", "64")
        app = server.create_default_app()
        assert isinstance(app, FastAPI)
        assert {"/health", "/metrics", "/v1/memory/ingest"} <= set(app.openapi()["paths"])

    def test_default_app_fails_fast_on_bad_configuration(self) -> None:
        with pytest.raises(ValidationError, match="neo4j_password"):
            server.create_default_app()


@pytest.mark.parametrize(
    "value", ["anthropic", "/claude", "Anthropic/claude", "anthropic/ claude", "a/" + "m" * 201]
)
def test_llm_model_must_be_provider_slash_model(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"neo4j_password": "pw", "llm_model": value})


@pytest.mark.parametrize("field", ["neo4j_database", "qdrant_api_key", "llm_model"])
def test_empty_optional_settings_mean_unset(field: str) -> None:
    settings = Settings.model_validate(
        {"neo4j_password": "pw", "qdrant_url": "http://qdrant:6333", field: "  "}
    )
    assert getattr(settings, field) is None
