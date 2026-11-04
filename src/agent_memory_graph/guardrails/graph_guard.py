"""Adversarial input screening and structural containment of traversal results.

:class:`InputGuard` looks for prompt-injection markers in conversation text before it
reaches the extraction model. Structural forgeries (fake chat-template tokens, fake
role headers, fake delimiters) are defanged in place; instruction-override phrasing
is flagged but kept, since people legitimately talk about such things. In ``reject``
mode any finding fails the request with ``AMG_INVALID_INPUT``.

:class:`TraversalGuard` bounds what a traversal can contribute to a context before
it is synthesised: per-seed fan-out, per-node out-degree (hubs), and the number of
cycle-closing edges. Anything only reachable through a pruned edge is dropped too, so
every kept path stays connected to a seed.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from agent_memory_graph.domain.errors import InvalidInputError
from agent_memory_graph.domain.query import GraphHop

__all__ = [
    "DEFAULT_MAX_CYCLE_EDGES",
    "DEFAULT_MAX_EDGES_PER_SEED",
    "DEFAULT_MAX_NODE_OUT_DEGREE",
    "FindingKind",
    "InputFinding",
    "InputGuard",
    "InputScan",
    "TraversalGuard",
    "TraversalReport",
]

DEFAULT_MAX_EDGES_PER_SEED: Final = 500
DEFAULT_MAX_NODE_OUT_DEGREE: Final = 100
DEFAULT_MAX_CYCLE_EDGES: Final = 100
_EXCERPT_CHARS: Final = 60


# ---------------------------------------------------------------------------
# Input screening
# ---------------------------------------------------------------------------
class FindingKind(StrEnum):
    """Classes of prompt-injection markers."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_INJECTION = "role_injection"
    DELIMITER_FORGERY = "delimiter_forgery"
    TEMPLATE_MARKER = "chat_template_marker"


# Every pattern uses bounded, single-class repetition: linear-time on any input.
_OVERRIDE_RE: Final = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,60}?"
    r"\b(?:instructions?|prompts?|rules?|guidelines?|directives?)\b"
    r"|\byou\s+are\s+now\b|\bnew\s+(?:system\s+)?instructions?\s*:",
    re.IGNORECASE,
)
_ROLE_RE: Final = re.compile(
    r"^([ \t]*)(system|developer)([ \t]*(?:prompt|message)?[ \t]*):", re.IGNORECASE | re.MULTILINE
)
_DELIMITER_RE: Final = re.compile(
    r"<\s*/?\s*(?:conversation|system|instructions?|developer)\b[^<>\n]{0,80}>"
    r"|<</?SYS>>|\[/?(?:INST|SYS)\]|```[ \t]*(?:system|developer)\b",
    re.IGNORECASE,
)
_TEMPLATE_RE: Final = re.compile(r"<\|[A-Za-z_]{2,32}\|>")


@dataclass(frozen=True, slots=True)
class InputFinding:
    """One suspicious span (``excerpt`` is for server-side logs only)."""

    kind: FindingKind
    excerpt: str


@dataclass(frozen=True, slots=True)
class InputScan:
    """The defanged text and everything that looked like an injection attempt."""

    text: str
    findings: tuple[InputFinding, ...]

    @property
    def suspicious(self) -> bool:
        """Whether anything was found."""
        return bool(self.findings)

    @property
    def kinds(self) -> tuple[FindingKind, ...]:
        """Distinct finding kinds, in declaration order."""
        present = {finding.kind for finding in self.findings}
        return tuple(kind for kind in FindingKind if kind in present)


def _normalise(text: str) -> str:
    # Match against what the model will see: NFKC (fullwidth letters fold to ASCII) with
    # zero-width / bidi characters removed, so fullwidth or zero-width-split spellings
    # of "ignore" still match.
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in normalized if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


# Every bracket kind becomes a parenthesis, so no defanged output can re-form a marker
# (mapping "<" to "[" would turn "<<SYS>>" into "[[SYS]]", which contains "[SYS]").
_DEFANG: Final = str.maketrans({"[": "(", "]": ")", "<": "(", ">": ")", "`": "'"})


def _bracket(match: re.Match[str]) -> str:
    return match.group(0).translate(_DEFANG)


class InputGuard:
    """Screens conversation text for prompt-injection markers."""

    def __init__(self, *, mode: Literal["flag", "reject"] = "flag") -> None:
        self.mode = mode

    def scan(self, text: str) -> InputScan:
        """Return the defanged text and the findings, without raising."""
        normalized = _normalise(text)
        findings = [
            InputFinding(kind, match.group(0)[:_EXCERPT_CHARS])
            for kind, pattern in (
                (FindingKind.INSTRUCTION_OVERRIDE, _OVERRIDE_RE),
                (FindingKind.ROLE_INJECTION, _ROLE_RE),
                (FindingKind.DELIMITER_FORGERY, _DELIMITER_RE),
                (FindingKind.TEMPLATE_MARKER, _TEMPLATE_RE),
            )
            for match in pattern.finditer(normalized)
        ]
        defanged = _TEMPLATE_RE.sub(_bracket, normalized)
        defanged = _DELIMITER_RE.sub(_bracket, defanged)
        defanged = _ROLE_RE.sub(lambda m: f"{m.group(1)}[{m.group(2)}]{m.group(3)}:", defanged)
        return InputScan(text=defanged, findings=tuple(findings))

    def check(self, text: str) -> InputScan:
        """Like :meth:`scan`, but raise :class:`InvalidInputError` in ``reject`` mode.

        The error names the finding kinds only; the offending text is never echoed.
        """
        scan = self.scan(text)
        if self.mode == "reject" and scan.suspicious:
            msg = "The text contains prompt-injection markers and was not stored."
            raise InvalidInputError(
                msg,
                errors=[
                    {"loc": ["text"], "msg": f"possible prompt injection: {kind}",
                     "type": "prompt_injection"}
                    for kind in scan.kinds
                ],
            )  # fmt: skip
        return scan


# ---------------------------------------------------------------------------
# Traversal containment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TraversalReport:
    """How a traversal result was contained."""

    hops_in: int = 0
    hops_kept: int = 0
    hub_pruned: int = 0
    fanout_pruned: int = 0
    cycle_edges: int = 0
    cycle_pruned: int = 0
    orphaned: int = 0

    @property
    def pruned(self) -> int:
        """Total hops removed."""
        return self.hops_in - self.hops_kept


class TraversalGuard:
    """Bounds fan-out, hub degree and cycle density of a traversal result.

    Edges are considered shallowest first, then strongest (highest weight), so the
    budget goes to the most relevant structure. A *cycle edge* points at a node that
    was already reached at the same or a shallower depth than the edge's source.
    """

    def __init__(
        self,
        *,
        max_edges_per_seed: int = DEFAULT_MAX_EDGES_PER_SEED,
        max_node_out_degree: int = DEFAULT_MAX_NODE_OUT_DEGREE,
        max_cycle_edges: int = DEFAULT_MAX_CYCLE_EDGES,
    ) -> None:
        if min(max_edges_per_seed, max_node_out_degree) < 1 or max_cycle_edges < 0:
            msg = "fan-out and degree limits must be >= 1 and the cycle budget >= 0"
            raise ValueError(msg)
        self.max_edges_per_seed = max_edges_per_seed
        self.max_node_out_degree = max_node_out_degree
        self.max_cycle_edges = max_cycle_edges

    def contain(
        self, seed_ids: Sequence[str], hops: Sequence[GraphHop]
    ) -> tuple[tuple[GraphHop, ...], TraversalReport]:
        """Return the hops to keep (original order) and what was pruned."""
        distance: dict[str, int] = dict.fromkeys(seed_ids, 0)
        for hop in hops:
            distance[hop.target_id] = min(distance.get(hop.target_id, hop.depth), hop.depth)

        ranked = sorted(
            range(len(hops)),
            key=lambda i: (hops[i].depth, -hops[i].weight, hops[i].source_id,
                           hops[i].predicate, hops[i].target_id),
        )  # fmt: skip
        owner: dict[str, str] = {seed: seed for seed in seed_ids}
        out_degree: Counter[str] = Counter()
        per_seed: Counter[str] = Counter()
        kept: set[int] = set()
        hub = fanout = cycles = cycle_pruned = orphaned = 0

        for index in ranked:
            hop = hops[index]
            seed = owner.get(hop.source_id)
            if seed is None:
                orphaned += 1  # its source is only reachable through a pruned edge
                continue
            if out_degree[hop.source_id] >= self.max_node_out_degree:
                hub += 1
                continue
            if per_seed[seed] >= self.max_edges_per_seed:
                fanout += 1
                continue
            if distance[hop.target_id] <= hop.depth - 1:
                cycles += 1
                if cycles > self.max_cycle_edges:
                    cycle_pruned += 1
                    continue
            kept.add(index)
            out_degree[hop.source_id] += 1
            per_seed[seed] += 1
            owner.setdefault(hop.target_id, seed)

        report = TraversalReport(
            hops_in=len(hops),
            hops_kept=len(kept),
            hub_pruned=hub,
            fanout_pruned=fanout,
            cycle_edges=cycles,
            cycle_pruned=cycle_pruned,
            orphaned=orphaned,
        )
        return tuple(hop for i, hop in enumerate(hops) if i in kept), report
