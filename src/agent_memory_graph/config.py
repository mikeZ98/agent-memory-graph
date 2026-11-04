"""Runtime configuration, read from ``AMG_*`` environment variables.

Only the process environment is consulted (no ``.env`` file is loaded implicitly), so
what a test or container sees is exactly what it was given. Secrets are held as
:class:`~pydantic.SecretStr` and never appear in ``repr`` or logs.
"""

import ipaddress
from typing import Annotated, Final, Literal, Self
from urllib.parse import SplitResult, urlsplit

from pydantic import AfterValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_memory_graph.domain.models import MAX_EMBEDDING_DIMENSIONS

__all__ = ["DEFAULT_EMBEDDING_DIM", "Settings"]

DEFAULT_EMBEDDING_DIM: Final = 384
"""Dimension of the small sentence-transformer family (all-MiniLM-L6, bge-small)."""

NEO4J_SCHEMES: Final = frozenset({"neo4j", "neo4j+s", "neo4j+ssc", "bolt", "bolt+s", "bolt+ssc"})
HTTP_SCHEMES: Final = frozenset({"http", "https"})
_COLLECTION_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$"


def _split_endpoint(value: str, schemes: frozenset[str]) -> SplitResult:
    parts = urlsplit(value)
    if parts.scheme not in schemes:
        msg = f"scheme must be one of {sorted(schemes)}"
        raise ValueError(msg)
    if not parts.hostname:
        msg = "URL must include a host"
        raise ValueError(msg)
    if parts.username is not None or parts.password is not None:
        msg = "credentials must not be embedded in the URL; use the dedicated settings"
        raise ValueError(msg)
    return parts


def _validate_neo4j_uri(value: str) -> str:
    _split_endpoint(value, NEO4J_SCHEMES)
    return value


def _validate_http_url(value: str) -> str:
    _split_endpoint(value, HTTP_SCHEMES)
    return value


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """Engine configuration; every field maps to ``AMG_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="AMG_",
        env_file=None,
        extra="forbid",
        frozen=True,
        validate_default=True,
        case_sensitive=False,
    )

    neo4j_uri: Annotated[str, AfterValidator(_validate_neo4j_uri)] = "neo4j://localhost:7687"
    neo4j_user: Annotated[str, Field(min_length=1, max_length=256)] = "neo4j"
    neo4j_password: SecretStr
    neo4j_database: Annotated[str, Field(min_length=1, max_length=63)] | None = None
    """Target database; ``None`` uses the server's home database."""

    qdrant_url: Annotated[str, AfterValidator(_validate_http_url)] = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: Annotated[str, Field(pattern=_COLLECTION_PATTERN)] = "amg_entities"

    embedding_dim: Annotated[int, Field(ge=1, le=MAX_EMBEDDING_DIMENSIONS)] = DEFAULT_EMBEDDING_DIM
    default_timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0, allow_inf_nan=False)] = 10.0
    neo4j_max_pool_size: Annotated[int, Field(ge=1, le=1000)] = 50
    qdrant_upsert_batch_size: Annotated[int, Field(ge=1, le=10_000)] = 256

    # Guardrails
    max_concurrent_extractions: Annotated[int, Field(ge=1, le=256)] = 4
    max_concurrent_store_operations: Annotated[int, Field(ge=1, le=1000)] = 32
    queue_timeout_seconds: Annotated[float, Field(gt=0.0, le=60.0, allow_inf_nan=False)] = 2.0
    breaker_failure_threshold: Annotated[int, Field(ge=1, le=100)] = 5
    breaker_reset_seconds: Annotated[float, Field(gt=0.0, le=600.0, allow_inf_nan=False)] = 30.0
    input_guard_mode: Literal["flag", "reject"] = "flag"
    retrieval_fallback_steps: Annotated[int, Field(ge=0, le=10)] = 3
    retrieval_fallback_floor: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.1

    @model_validator(mode="after")
    def _no_api_key_over_plaintext(self) -> Self:
        # Sending a bearer secret over plain HTTP is only tolerable on the loopback.
        parts = urlsplit(self.qdrant_url)
        if (
            self.qdrant_api_key is not None
            and parts.scheme == "http"
            and not _is_loopback(parts.hostname or "")
        ):
            msg = "qdrant_api_key requires an https qdrant_url (except on loopback hosts)"
            raise ValueError(msg)
        return self
