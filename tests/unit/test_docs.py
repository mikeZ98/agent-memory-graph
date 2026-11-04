"""README tests: documented facts match the code, and Mermaid/markdown is well-formed.

Catches documentation drift (a new error code, setting, route or counter that the
README does not mention, or a stale status/title) and the structural mistakes that
make GitHub render a diagram or table incorrectly.
"""

import itertools
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest

from agent_memory_graph.api import create_app
from agent_memory_graph.config import Settings
from agent_memory_graph.domain.errors import PROBLEM_TAXONOMY, ErrorCode
from agent_memory_graph.guardrails.telemetry import COUNTERS
from agent_memory_graph.services.retrieval import GraphMemoryService

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
README: Final = (REPO_ROOT / "README.md").read_text()
LINES: Final = README.splitlines()

FLOWCHART_BLOCKS: Final = ("subgraph",)
SEQUENCE_BLOCKS: Final = ("loop", "alt", "opt", "par", "critical", "break", "rect")
_SEQ_ARROW_RE: Final = re.compile(
    r"^\s*([A-Za-z]\w*)\s*(?:-{1,2}>>|-{1,2}x|-{1,2}\)|-{1,2}>)\s*([A-Za-z]\w*)\s*:(.*)$"
)
_PARTICIPANT_RE: Final = re.compile(r"^\s*(?:participant|actor)\s+([A-Za-z]\w*)\b")
_FLOW_ARROW_RE: Final = re.compile(r"-\.\s*\.->|-\.->|-->|==>|---")
_LABEL_RE: Final = re.compile(r"[\[\(\{][^\[\]\(\)\{\}]*[\]\)\}]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fenced_blocks(language: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in LINES:
        if current is None and line.strip() == f"```{language}":
            current = []
        elif current is not None and line.strip() == "```":
            blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def outside_code_fences() -> Iterator[str]:
    fenced = False
    for line in LINES:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            yield line


def github_slug(heading: str) -> str:
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def table_row_cells(row: str) -> int:
    return len(re.split(r"(?<!\\)\|", row.strip().strip("|")))


def taxonomy_rows() -> dict[str, str]:
    return {
        match.group(1): line
        for line in LINES
        if (match := re.match(r"^\| `(AMG_[A-Z_]+)` \| `urn:", line))
    }


# ---------------------------------------------------------------------------
# Facts match the code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", list(ErrorCode), ids=lambda c: c.value)
def test_every_error_code_is_documented_exactly(code: ErrorCode) -> None:
    row = taxonomy_rows().get(code.value)
    assert row is not None, f"{code.value} missing from the RFC 9457 taxonomy table"
    spec = PROBLEM_TAXONOMY[code]
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
    assert cells[1] == f"`{spec.type}`"
    assert cells[2] == spec.title
    assert cells[3] == str(spec.status)


def test_taxonomy_table_has_no_stale_codes() -> None:
    assert set(taxonomy_rows()) == {code.value for code in ErrorCode}


@pytest.mark.parametrize("field", list(Settings.model_fields))
def test_every_setting_is_documented(field: str) -> None:
    assert f"`AMG_{field.upper()}`" in README


@pytest.mark.parametrize(
    "variable", ["AMG_HTTP_HOST", "AMG_HTTP_PORT", "AMG_LOG_LEVEL", "AMG_GRACEFUL_TIMEOUT_SECONDS"]
)
def test_launcher_variables_are_documented(variable: str) -> None:
    assert variable in README


def _routes() -> set[str]:
    app = create_app(lambda: GraphMemoryService.__new__(GraphMemoryService))
    return set(app.openapi()["paths"])


def test_every_route_is_documented() -> None:
    for path in _routes():
        assert f"`GET {path}`" in README or f"`POST {path}`" in README, path


def test_curl_examples_only_target_real_routes() -> None:
    targets = set(re.findall(r"curl [^\n]*?http://127\.0\.0\.1:8000(/[\w/.-]*)", README))
    assert {"/health", "/metrics", "/v1/memory/ingest", "/v1/memory/query"} <= targets
    assert targets <= _routes()


@pytest.mark.parametrize("counter", COUNTERS)
def test_every_metrics_counter_is_documented(counter: str) -> None:
    assert f"`{counter}`" in README


def test_json_examples_are_valid_json() -> None:
    blocks = fenced_blocks("json")
    assert len(blocks) >= 3
    for block in blocks:
        json.loads("\n".join(block))


def test_stride_covers_every_category() -> None:
    for letter, category in zip("STRIDE", ("poofing", "ampering", "epudiation",
                                          "nformation disclosure", "enial of service",
                                          "levation of privilege"), strict=True):  # fmt: skip
        assert f"| **{letter}**{category} |" in README, category


def test_pinned_images_match_the_stack() -> None:
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    for image in re.findall(r"(?:neo4j|qdrant/qdrant):[\w.-]+@sha256:[0-9a-f]{64}", README):
        assert image in compose, image


# ---------------------------------------------------------------------------
# Markdown structure
# ---------------------------------------------------------------------------
def test_code_fences_are_balanced() -> None:
    assert sum(1 for line in LINES if line.lstrip().startswith("```")) % 2 == 0


def test_heading_hierarchy() -> None:
    headings = [line for line in outside_code_fences() if re.match(r"^#{1,6} ", line)]
    levels = [len(line) - len(line.lstrip("#")) for line in headings]
    assert levels.count(1) == 1
    assert levels[0] == 1
    assert all(later - earlier <= 1 for earlier, later in itertools.pairwise(levels))


def test_internal_links_resolve() -> None:
    anchors = {
        github_slug(line.lstrip("#"))
        for line in outside_code_fences()
        if re.match(r"^#{1,6} ", line)
    }
    links = re.findall(r"\]\(#([^)]+)\)", README)
    assert links
    assert [link for link in links if link not in anchors] == []


def test_relative_file_links_exist() -> None:
    for target in re.findall(r"\]\((?!https?://|#)([^)]+)\)", README):
        assert (REPO_ROOT / target).exists(), target


def test_tables_have_consistent_columns() -> None:
    table: list[str] = []
    for line in [*outside_code_fences(), ""]:
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            widths = {table_row_cells(row) for row in table}
            assert len(widths) == 1, table[0]
            assert re.match(r"^\|(?:\s*:?-{3,}:?\s*\|)+$", table[1]), table[1]
            table = []


# ---------------------------------------------------------------------------
# Mermaid structure
# ---------------------------------------------------------------------------
MERMAID: Final = fenced_blocks("mermaid")


def test_expected_diagrams_are_present() -> None:
    kinds = [block[0].split()[0] for block in MERMAID]
    assert kinds.count("flowchart") >= 2
    assert kinds.count("sequenceDiagram") >= 3


@pytest.mark.parametrize("block", MERMAID, ids=lambda b: b[0].strip())
def test_diagram_blocks_are_balanced(block: list[str]) -> None:
    kind = block[0].split()[0]
    openers = FLOWCHART_BLOCKS if kind == "flowchart" else SEQUENCE_BLOCKS
    depth = 0
    for line in block[1:]:
        word = line.strip().split(" ", 1)[0]
        if word in openers:
            depth += 1
        elif word == "end":
            depth -= 1
            assert depth >= 0, line
        elif word == "else":
            assert depth > 0, line
    assert depth == 0


@pytest.mark.parametrize(
    "block", [b for b in MERMAID if b[0].startswith("sequenceDiagram")], ids=lambda b: str(len(b))
)
def test_sequence_diagrams_use_declared_participants(block: list[str]) -> None:
    declared = {m.group(1) for line in block if (m := _PARTICIPANT_RE.match(line))}
    messages = [m for line in block if (m := _SEQ_ARROW_RE.match(line))]
    assert messages
    for message in messages:
        assert {message.group(1), message.group(2)} <= declared, message.group(0)
        text = message.group(3)
        assert ";" not in text, f"';' ends a Mermaid statement: {text}"
        assert "#" not in text, f"'#' starts a Mermaid entity code: {text}"
        assert re.search(r"<(?!br/>)", text) is None, f"'<' is parsed as HTML: {text}"
    for line in block:
        note = re.match(r"^\s*Note over ([\w, ]+):", line)
        if note:
            assert {p.strip() for p in note.group(1).split(",")} <= declared, line


@pytest.mark.parametrize(
    "block", [b for b in MERMAID if b[0].startswith("flowchart")], ids=lambda b: b[0]
)
def test_flowchart_edges_reference_defined_nodes(block: list[str]) -> None:
    defined: set[str] = set()
    referenced: set[str] = set()
    for line in block[1:]:
        stripped = line.strip()
        assert stripped.count('"') % 2 == 0, f"unbalanced quotes: {stripped}"
        if not stripped or stripped.startswith(("direction", "end", "%%")):
            continue
        if stripped.startswith("subgraph"):
            defined.add(stripped.split()[1].split("[")[0])
            continue
        defined.update(re.findall(r"([A-Za-z]\w*)\s*[\[\(\{]", stripped))
        unlabeled = re.sub(r'"[^"]*"', "", stripped)
        previous = None
        while previous != unlabeled:
            previous, unlabeled = unlabeled, _LABEL_RE.sub("", unlabeled)
        assert not re.search(r"[\[\]\(\)\{\}]", unlabeled), f"unbalanced brackets: {stripped}"
        if _FLOW_ARROW_RE.search(unlabeled):
            for part in _FLOW_ARROW_RE.split(unlabeled):
                if part.strip():
                    referenced.add(part.strip())
    assert referenced
    assert referenced <= defined, sorted(referenced - defined)


# ---------------------------------------------------------------------------
# The linters above reject broken input (so a passing run means something)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("check", "block"),
    [
        (test_diagram_blocks_are_balanced, ["sequenceDiagram", "    loop x", "    A->>B: hi"]),
        (test_diagram_blocks_are_balanced, ["sequenceDiagram", "    end"]),
        (test_diagram_blocks_are_balanced, ["sequenceDiagram", "    else nothing"]),
        (test_diagram_blocks_are_balanced, ["flowchart LR", '    subgraph s["S"]', "    a --> b"]),
        (test_sequence_diagrams_use_declared_participants,
         ["sequenceDiagram", "    participant A", "    A->>B: undeclared target"]),
        (test_sequence_diagrams_use_declared_participants,
         ["sequenceDiagram", "    participant A", "    participant B", "    A->>B: a; b"]),
        (test_sequence_diagrams_use_declared_participants,
         ["sequenceDiagram", "    participant A", "    participant B", "    A->>B: issue #42"]),
        (test_sequence_diagrams_use_declared_participants,
         ["sequenceDiagram", "    participant A", "    participant B", "    A->>B: <tag> text"]),
        (test_sequence_diagrams_use_declared_participants,
         ["sequenceDiagram", "    participant A", "    participant B", "    A->>B: ok",
          "    Note over Z: who?"]),
        (test_flowchart_edges_reference_defined_nodes,
         ["flowchart LR", '    a["A"]', "    a --> typo"]),
        (test_flowchart_edges_reference_defined_nodes,
         ["flowchart LR", '    a["A]', "    a --> a"]),
        (test_flowchart_edges_reference_defined_nodes,
         ["flowchart LR", "    a[A --> b[B]"]),
    ],
)  # fmt: skip
def test_mermaid_linters_reject_broken_diagrams(
    check: Callable[[list[str]], None], block: list[str]
) -> None:
    with pytest.raises(AssertionError):
        check(block)


def test_table_and_slug_helpers() -> None:
    assert table_row_cells("| a | `b \\| c` | d |") == 3
    assert table_row_cells("| a | b | c |") == 3
    assert github_slug("Threat model (STRIDE)") == "threat-model-stride"
    assert (
        github_slug("GraphRAG vs. classical vector search") == "graphrag-vs-classical-vector-search"
    )
    assert github_slug("RFC 9457 error taxonomy") == "rfc-9457-error-taxonomy"
