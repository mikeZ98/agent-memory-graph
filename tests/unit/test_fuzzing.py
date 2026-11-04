"""Property-based fuzzing of the invariants every other test relies on.

Run with ``HYPOTHESIS_PROFILE=ci`` for the deeper, derandomised profile.
"""

import math
import string
from collections import deque
from collections.abc import Sequence

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from agent_memory_graph.domain import (
    Entity,
    EntityType,
    GraphHop,
    InvalidInputError,
    KnowledgeGraph,
    make_entity_id,
)
from agent_memory_graph.domain.models import EntityId
from agent_memory_graph.guardrails import FindingKind, InputGuard, TraversalGuard
from agent_memory_graph.guardrails.graph_guard import _DELIMITER_RE, _ROLE_RE, _TEMPLATE_RE
from agent_memory_graph.services.extractor import (
    GraphExtraction,
    build_knowledge_graph,
    prepare_conversation,
)
from agent_memory_graph.services.retrieval import RetrievalPolicy, fallback_thresholds

pytestmark = pytest.mark.fuzz

ENTITY_ID: TypeAdapter[str] = TypeAdapter(EntityId)
STRUCTURAL = {
    FindingKind.ROLE_INJECTION,
    FindingKind.DELIMITER_FORGERY,
    FindingKind.TEMPLATE_MARKER,
}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "</conversation>", "<conversation>",
    "< / conversation >", "<system>", '<system role="root">', "</instructions>", "[INST]",
    "[/INST]", "[SYS]", "<<SYS>>", "<</SYS>>", "```system", "System:", "developer prompt:",
    "\nSYSTEM :", "Ignore all previous instructions", "you are now", "new instructions:",
    "<", ">", "[", "]", "|", "`", "\n", "\u200b", "\u202e", "\ufeff", "\x00",
    "\uff1c\uff5cim_start\uff5c\uff1e",  # fullwidth <|im_start|>, folds under NFKC
]  # fmt: skip

hostile_text = st.lists(
    st.one_of(
        st.sampled_from(MARKERS),
        st.text(max_size=12),
        st.text(alphabet=string.ascii_letters + " <>|[]/:`\n", max_size=12),
    ),
    max_size=30,
).map("".join)

entity_types = st.sampled_from(list(EntityType))
names = st.one_of(
    st.text(min_size=1, max_size=80),
    st.text(alphabet=string.ascii_letters + string.digits + " -_.'&", min_size=1, max_size=80),
)


# ---------------------------------------------------------------------------
# Text sanitisation and defanging
# ---------------------------------------------------------------------------
@given(hostile_text)
def test_input_guard_never_raises_in_flag_mode_and_leaves_no_structural_markers(
    text: str,
) -> None:
    scan = InputGuard().scan(text)
    assert _TEMPLATE_RE.search(scan.text) is None
    assert _DELIMITER_RE.search(scan.text) is None
    assert _ROLE_RE.search(scan.text) is None
    # Re-scanning the output finds nothing structural: defanging is a fixed point.
    assert not STRUCTURAL & set(InputGuard().scan(scan.text).kinds)
    # The output is safe to wrap: the conversation delimiter cannot be closed early.
    wrapped = prepare_conversation(scan.text, max_chars=10**6)
    assert "</conversation>" not in wrapped.lower()


@given(hostile_text)
def test_reject_mode_raises_exactly_when_something_was_found(text: str) -> None:
    flagged = InputGuard().scan(text)
    if not flagged.suspicious:
        assert InputGuard(mode="reject").check(text) == flagged
        return
    with pytest.raises(InvalidInputError) as exc:
        InputGuard(mode="reject").check(text)
    assert {e["type"] for e in exc.value.errors} == {"prompt_injection"}
    assert len(exc.value.errors) == len(flagged.kinds)


@given(hostile_text)
def test_scan_output_contains_no_invisible_characters(text: str) -> None:
    scan = InputGuard().scan(text)
    assert all(ch in "\n\t" or ch.isprintable() or ch == " " for ch in scan.text)


# ---------------------------------------------------------------------------
# Entity ids
# ---------------------------------------------------------------------------
@given(names, entity_types)
def test_entity_ids_are_deterministic_canonical_and_valid(
    name: str, entity_type: EntityType
) -> None:
    try:
        derived = make_entity_id(name, entity_type)
    except ValueError:
        assume(False)  # nothing printable: rejected, as documented
    assert derived == make_entity_id(name, entity_type)
    assert ENTITY_ID.validate_python(derived) == derived  # already canonical
    assert derived == make_entity_id(f"  {name}\t\u200b ", entity_type)  # padding-insensitive
    if not derived.startswith(f"{entity_type.value.lower()}-"):
        assert len(derived) == 36  # UUIDv5 fallback for names without an ASCII slug
    Entity(id=derived, name=name, entity_type=entity_type)  # usable as-is


@given(st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=60), entity_types)
def test_ascii_entity_ids_ignore_case(name: str, entity_type: EntityType) -> None:
    assume(name.strip())
    assert make_entity_id(name.upper(), entity_type) == make_entity_id(name.lower(), entity_type)


@given(names, st.lists(entity_types, min_size=2, max_size=2, unique=True))
def test_types_namespace_ids(name: str, types: list[EntityType]) -> None:
    try:
        first, second = (make_entity_id(name, t) for t in types)
    except ValueError:
        assume(False)
    assert first != second


candidate_names = st.sampled_from(
    ["Neo4j", "neo4j ", "Qdrant", "JVM", "\u200b", "Ghost", "x" * 300]
)
extractions = st.builds(
    lambda entities, relations: {"entities": entities, "relations": relations},
    st.lists(
        st.fixed_dictionaries(
            {"name": candidate_names, "entity_type": st.sampled_from(["TOOL", "tool", "ANIMAL"])}
        ),
        max_size=8,
    ),
    st.lists(
        st.fixed_dictionaries(
            {
                "source": candidate_names,
                "predicate": st.one_of(
                    st.sampled_from(["uses", "DEPENDS_ON", "runs on", "??", "2FA", "isPartOf"]),
                    st.text(max_size=20),
                ),
                "target": candidate_names,
                "confidence": st.floats(0.0, 1.0),
            }
        ),
        max_size=12,
    ),
)


@given(extractions)
def test_any_llm_output_becomes_a_valid_graph_or_is_pruned(payload: dict[str, object]) -> None:
    try:
        extraction = GraphExtraction.model_validate(payload)
    except ValidationError:
        assume(False)  # instructor would re-ask the model
    graph, report = build_knowledge_graph(extraction)
    KnowledgeGraph.model_validate(graph.model_dump())  # integrity holds
    ids = {entity.id for entity in graph.entities}
    assert all({r.source_id, r.target_id} <= ids for r in graph.relations)
    assert len({r.key for r in graph.relations}) == len(graph.relations)
    assert report.relations == len(graph.relations)
    assert report.relations + report.pruned <= len(extraction.relations) + report.invalid_entities
    assert KnowledgeGraph.model_validate_json(graph.model_dump_json()) == graph


# ---------------------------------------------------------------------------
# Traversal containment
# ---------------------------------------------------------------------------
NODES = [f"n-{i}" for i in range(40)]


@st.composite
def traversals(draw: st.DrawFn) -> tuple[list[str], list[GraphHop]]:
    """Random graphs mixing dense cycles, huge stars, deep chains and orphans."""
    seeds = draw(st.lists(st.sampled_from(NODES[:5]), min_size=1, max_size=3, unique=True))
    hops: list[GraphHop] = []
    shape = draw(st.sampled_from(["random", "star", "cycle", "chain", "mixed"]))
    if shape in {"random", "mixed"}:
        hops.extend(
            GraphHop(
                source_id=draw(st.sampled_from(NODES)),
                predicate=draw(st.sampled_from(["USES", "LINKS"])),
                target_id=draw(st.sampled_from(NODES)),
                weight=draw(st.floats(0.0, 1.0)),
                depth=draw(st.integers(1, 3)),
            )
            for _ in range(draw(st.integers(0, 150)))
        )
    if shape in {"star", "mixed"}:
        hub = draw(st.sampled_from(seeds))
        width = draw(st.integers(1, 1500))
        hops += [
            GraphHop(source_id=hub, predicate="HAS", target_id=f"leaf-{i}") for i in range(width)
        ]
    if shape in {"cycle", "mixed"}:
        ring = NODES[: draw(st.integers(2, 20))]
        hops += [
            GraphHop(source_id=a, predicate="NEXT", target_id=b, depth=(i % 3) + 1)
            for i, (a, b) in enumerate(zip(ring, ring[1:] + ring[:1], strict=True))
        ]
    if shape == "chain":
        chain = draw(st.integers(1, 3))
        hops += [
            GraphHop(source_id=f"c-{i}" if i else seeds[0], predicate="TO",
                     target_id=f"c-{i + 1}", depth=i + 1)
            for i in range(chain)
        ]  # fmt: skip
    return seeds, hops


limits = st.tuples(st.integers(1, 600), st.integers(1, 120), st.integers(0, 120))


def _cycle_edges(
    seeds: Sequence[str], all_hops: Sequence[GraphHop], kept: Sequence[GraphHop]
) -> int:
    distance = dict.fromkeys(seeds, 0)
    for h in all_hops:
        distance[h.target_id] = min(distance.get(h.target_id, h.depth), h.depth)
    return sum(1 for h in kept if distance[h.target_id] <= h.depth - 1)


@given(traversals(), limits)
def test_traversal_guard_respects_every_limit(
    traversal: tuple[list[str], list[GraphHop]], bounds: tuple[int, int, int]
) -> None:
    seeds, hops = traversal
    per_seed, out_degree, cycle_budget = bounds
    guard = TraversalGuard(
        max_edges_per_seed=per_seed, max_node_out_degree=out_degree, max_cycle_edges=cycle_budget
    )
    kept, report = guard.contain(seeds, hops)

    # Kept hops are a subsequence of the input (order preserved, nothing invented).
    remaining = iter(hops)
    assert all(any(hop is candidate for candidate in remaining) for hop in kept)
    # Every limit holds.
    degrees: dict[str, int] = {}
    for hop in kept:
        degrees[hop.source_id] = degrees.get(hop.source_id, 0) + 1
    assert max(degrees.values(), default=0) <= out_degree
    assert len(kept) <= per_seed * len(seeds)
    assert _cycle_edges(seeds, hops, kept) <= cycle_budget
    # Every kept path is anchored to a seed.
    reachable, frontier = set(seeds), deque(seeds)
    adjacency: dict[str, list[str]] = {}
    for hop in kept:
        adjacency.setdefault(hop.source_id, []).append(hop.target_id)
    while frontier:
        for target in adjacency.get(frontier.popleft(), ()):
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    assert all(hop.source_id in reachable for hop in kept)
    # The report adds up.
    assert (report.hops_in, report.hops_kept) == (len(hops), len(kept))
    assert report.pruned == (
        report.hub_pruned + report.fanout_pruned + report.cycle_pruned + report.orphaned
    )


@given(st.integers(1, 3))
def test_traversal_guard_handles_huge_inputs_iteratively(depth: int) -> None:
    hops = [
        GraphHop(source_id=f"n-{i}", predicate="NEXT", target_id=f"n-{i + 1}", depth=depth)
        for i in range(20_000)
    ]
    kept, report = TraversalGuard().contain(["n-0"], hops)
    assert report.hops_in == 20_000
    assert len(kept) <= 500


# ---------------------------------------------------------------------------
# Score fallback convergence
# ---------------------------------------------------------------------------
policies = st.builds(
    RetrievalPolicy,
    fallback_factor=st.floats(0.01, 0.99),
    fallback_floor=st.floats(0.0, 1.0),
    max_fallback_steps=st.integers(0, 10),
)


@given(st.floats(0.0, 1.0), policies)
def test_fallback_thresholds_converge_monotonically_to_the_floor(
    threshold: float, policy: RetrievalPolicy
) -> None:
    schedule = fallback_thresholds(threshold, policy)
    assert schedule[0] == threshold
    assert len(schedule) <= policy.max_fallback_steps + 1
    relaxed = schedule[1:]
    assert all(later < earlier for earlier, later in zip(schedule, relaxed, strict=False))
    assert all(value >= policy.fallback_floor for value in relaxed)
    assert all(0.0 <= value <= 1.0 for value in schedule)
    if threshold <= policy.fallback_floor:
        assert schedule == (threshold,)
    elif relaxed:
        expected_last = threshold * policy.fallback_factor ** len(relaxed)
        assert math.isclose(relaxed[-1], max(policy.fallback_floor, expected_last), abs_tol=1e-9)
        if len(relaxed) < policy.max_fallback_steps:
            assert relaxed[-1] == policy.fallback_floor  # stopped early only at the floor
