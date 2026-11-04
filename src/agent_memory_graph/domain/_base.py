"""Shared building blocks for every domain contract.

Strictness policy
-----------------
Scalars are strict *per field* (no ``"0.5"`` → ``0.5``, no ``True`` → ``1``), while
containers and enums keep pydantic's lax rules. Model-wide ``strict=True`` would make
FastAPI (which validates request bodies in Python mode) reject JSON arrays for tuple
fields and plain strings for enum fields. Tuples are used instead of lists so a
frozen model is immutable all the way down; the JSON shape is identical.
"""

import math
import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Final, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    Strict,
)

__all__ = [
    "MAX_JSON_NESTING",
    "FrozenJsonMapping",
    "Probability",
    "StrictModel",
    "freeze_json_mapping",
    "sanitize_line",
    "sanitize_multiline",
    "thaw_json",
]

MAX_JSON_NESTING: Final = 16
_MAX_JSON_KEY_LENGTH: Final = 128
# Raw input is rejected before normalisation once it exceeds this multiple of the
# limit, which bounds the work NFKC expansion can be made to do.
_RAW_LENGTH_FACTOR: Final = 4

_WHITESPACE_RUN: Final = re.compile(r"\s+")
_HORIZONTAL_WHITESPACE_RUN: Final = re.compile(r"[^\S\n]+")
_BLANK_LINE_RUN: Final = re.compile(r"\n{3,}")


class StrictModel(BaseModel):
    """Fail-closed base: unknown keys rejected, immutable, defaults validated."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


# ---------------------------------------------------------------------------
# Text sanitisation
# ---------------------------------------------------------------------------
def _check_raw_length(value: str, max_length: int) -> None:
    if len(value) > max_length * _RAW_LENGTH_FACTOR:
        msg = f"text exceeds {max_length} characters"
        raise ValueError(msg)


def _is_invisible(char: str) -> bool:
    # Unicode "C*" categories: control, format (zero-width, bidi overrides),
    # surrogate, private-use and unassigned code points.
    return unicodedata.category(char).startswith("C")


def sanitize_line(value: str, *, max_length: int) -> str:
    """Return a single-line, NFKC-normalised string without invisible characters.

    Every whitespace run (including newlines) collapses to one space and the result is
    stripped. Raises ``ValueError`` when nothing printable remains or it is too long.
    """
    _check_raw_length(value, max_length)
    normalized = unicodedata.normalize("NFKC", value)
    visible = "".join(
        " " if char.isspace() else char
        for char in normalized
        if char.isspace() or not _is_invisible(char)
    )
    cleaned = _WHITESPACE_RUN.sub(" ", visible).strip()
    if not cleaned:
        msg = "text must contain at least one printable character"
        raise ValueError(msg)
    if len(cleaned) > max_length:
        msg = f"text exceeds {max_length} characters"
        raise ValueError(msg)
    return cleaned


def _normalize_line(line: str) -> str:
    indent_end = len(line) - len(line.lstrip())
    body = _HORIZONTAL_WHITESPACE_RUN.sub(" ", line[indent_end:]).rstrip()
    return f"{line[:indent_end]}{body}" if body else ""


def sanitize_multiline(value: str, *, max_length: int) -> str:
    """Return NFKC-normalised text that keeps line structure (for prose and markdown).

    Line endings become LF, invisible characters are removed, leading indentation is
    kept (markdown lists and code blocks depend on it), other horizontal whitespace
    runs collapse to one space, trailing spaces and runs of more than one
    blank line are dropped. An empty result is allowed.
    """
    _check_raw_length(value, max_length)
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    visible = "".join(char for char in normalized if char in "\n\t " or not _is_invisible(char))
    lines = (_normalize_line(line) for line in visible.split("\n"))
    cleaned = _BLANK_LINE_RUN.sub("\n\n", "\n".join(lines)).strip()
    if len(cleaned) > max_length:
        msg = f"text exceeds {max_length} characters"
        raise ValueError(msg)
    return cleaned


# ---------------------------------------------------------------------------
# Immutable JSON payloads (attributes, properties, problem context)
# ---------------------------------------------------------------------------
def _freeze(value: object, path: str, depth: int) -> object:
    if depth > MAX_JSON_NESTING:
        msg = f"{path}: nesting exceeds {MAX_JSON_NESTING} levels"
        raise ValueError(msg)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{path}: non-finite numbers are not valid JSON"
            raise ValueError(msg)
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > _MAX_JSON_KEY_LENGTH:
                msg = f"{path}: keys must be non-empty strings of at most {_MAX_JSON_KEY_LENGTH}"
                raise ValueError(msg)
            frozen[key] = _freeze(item, f"{path}.{key}", depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item, f"{path}[{i}]", depth + 1) for i, item in enumerate(value))
    msg = f"{path}: unsupported JSON value of type {type(value).__name__}"
    raise ValueError(msg)


def freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-copy a JSON object into read-only mappings and tuples.

    Only JSON-representable values (``None``, ``bool``, ``int``, finite ``float``,
    ``str``, sequences and string-keyed mappings) are accepted.
    """
    return cast("Mapping[str, Any]", _freeze(value, "$", 0))


def thaw_json(value: object) -> Any:  # noqa: ANN401 - mirrors arbitrary JSON
    """Inverse of :func:`freeze_json_mapping`: plain ``dict``/``list`` for serialisation."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


FrozenJsonMapping = Annotated[
    Mapping[str, Any],
    AfterValidator(freeze_json_mapping),
    PlainSerializer(thaw_json, return_type=dict[str, Any]),
    Field(default_factory=lambda: MappingProxyType({})),
]
"""Read-only, JSON-only ``Mapping[str, Any]`` that serialises back to a plain dict."""

Probability = Annotated[float, Strict(), Field(ge=0.0, le=1.0, allow_inf_nan=False)]
"""A finite float in ``[0.0, 1.0]``; ints are accepted, strings and bools are not."""
