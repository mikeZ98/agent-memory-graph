"""Domain schema contracts: immutability, sanitisation, integrity and wire stability."""

import json
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from agent_memory_graph.domain import (
    PROBLEM_JSON_MEDIA_TYPE,
    PROBLEM_TAXONOMY,
    Entity,
    EntityIntegrityError,
    EntityType,
    ErrorCode,
    ExtractionFailedError,
    ExtractionTimeoutError,
    GraphContext,
    GraphHop,
    InvalidInputError,
    KnowledgeGraph,
    MemoryGraphError,
    MemoryProblemDetails,
    MemoryQuery,
    Relation,
    StoreTimeoutError,
    StoreUnavailableError,
    TraversalDepthExceededError,
    VectorSearchTimeoutError,
    make_entity_id,
)
from agent_memory_graph.domain._base import (
    MAX_JSON_NESTING,
    freeze_json_mapping,
    sanitize_line,
    sanitize_multiline,
    thaw_json,
)
from agent_memory_graph.domain.models import (
    ENTITY_NAMESPACE,
    MAX_ENTITY_ID_LENGTH,
    MAX_PREDICATE_LENGTH,
)
from agent_memory_graph.domain.query import MAX_TOP_K, MAX_TRAVERSAL_DEPTH

UUID_UPPER = "123E4567-E89B-12D3-A456-426614174000"


def _error_types(exc: ValidationError) -> set[str]:
    return {str(error["type"]) for error in exc.errors()}


def entity(eid: str, name: str | None = None, **kwargs: Any) -> Entity:
    return Entity(
        id=eid,
        name=eid if name is None else name,
        entity_type=kwargs.pop("entity_type", EntityType.TOOL),
        **kwargs,
    )


def relation(source: str, predicate: str, target: str, **kwargs: Any) -> Relation:
    return Relation(source_id=source, predicate=predicate, target_id=target, **kwargs)


@pytest.fixture
def graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        entities=(
            entity("tool-neo4j", "Neo4j", description="Graph database.", embedding=[0.1, 0.2]),
            entity("tool-qdrant", "Qdrant", attributes={"license": "Apache-2.0", "tags": ["v"]}),
            entity("concept-graphrag", "GraphRAG", entity_type=EntityType.CONCEPT),
        ),
        relations=(
            relation("concept-graphrag", "USES", "tool-neo4j", weight=0.9),
            relation("concept-graphrag", "uses", "tool-qdrant", properties={"since": 2024}),
        ),
    )


# ---------------------------------------------------------------------------
# Immutability and forbidden extra fields
# ---------------------------------------------------------------------------
class TestImmutability:
    def test_models_are_frozen(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValidationError) as exc:
            graph.entities[0].name = "Other"  # type: ignore[misc]
        assert _error_types(exc.value) == {"frozen_instance"}

    def test_collections_are_tuples(self, graph: KnowledgeGraph) -> None:
        assert isinstance(graph.entities, tuple)
        assert isinstance(graph.relations, tuple)
        assert isinstance(graph.entities[0].embedding, tuple)

    def test_json_payloads_are_deeply_read_only(self, graph: KnowledgeGraph) -> None:
        attributes = graph.entities[1].attributes
        assert isinstance(attributes, MappingProxyType)
        assert attributes["tags"] == ("v",)
        with pytest.raises(TypeError):
            attributes["license"] = "MIT"  # type: ignore[index]

    def test_payload_is_copied_not_aliased(self) -> None:
        source: dict[str, Any] = {"nested": {"k": [1]}}
        node = entity("tool-x", attributes=source)
        source["nested"]["k"].append(2)
        assert node.attributes["nested"]["k"] == (1,)

    def test_default_payload_is_empty_read_only_mapping(self) -> None:
        assert entity("tool-x").attributes == MappingProxyType({})
        assert isinstance(relation("tool-x", "USES", "tool-y").properties, MappingProxyType)

    @pytest.mark.parametrize(
        ("model", "payload"),
        [
            (Entity, {"id": "tool-x", "name": "X", "entity_type": "TOOL", "rank": 1}),
            (Relation, {"source_id": "a", "predicate": "USES", "target_id": "b", "label": "x"}),
            (KnowledgeGraph, {"entities": [], "relations": [], "version": 2}),
            (MemoryQuery, {"text": "q", "limit": 3}),
            (GraphHop, {"source_id": "a", "predicate": "USES", "target_id": "b", "path": 1}),
            (GraphContext, {"seed_entities": [], "extra": True}),
        ],
    )
    def test_extra_fields_are_forbidden(
        self, model: type[BaseModel], payload: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError) as exc:
            model.model_validate(payload)
        assert _error_types(exc.value) == {"extra_forbidden"}

    def test_problem_details_forbid_extra_fields(self) -> None:
        body = MemoryProblemDetails.from_code(ErrorCode.INTERNAL, "boom").to_json_dict()
        with pytest.raises(ValidationError):
            MemoryProblemDetails.model_validate({**body, "stack": "trace"})


# ---------------------------------------------------------------------------
# Strict scalars
# ---------------------------------------------------------------------------
class TestStrictScalars:
    @pytest.mark.parametrize("weight", ["0.5", True, -0.01, 1.01, float("nan"), float("inf")])
    def test_relation_weight_rejects_coercion_and_out_of_range(self, weight: object) -> None:
        with pytest.raises(ValidationError):
            relation("a", "USES", "b", weight=weight)

    def test_relation_weight_accepts_int_bounds(self) -> None:
        assert relation("a", "USES", "b", weight=0).weight == 0.0
        assert relation("a", "USES", "b").weight == 1.0

    @pytest.mark.parametrize("value", [[], ["0.1"], [float("nan")], [0.0] * 8193])
    def test_embedding_rejects_invalid_vectors(self, value: list[Any]) -> None:
        with pytest.raises(ValidationError):
            entity("tool-x", embedding=value)

    @pytest.mark.parametrize("bad", [123, None, b"tool-x"])
    def test_ids_must_be_strings(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            relation(bad, "USES", "b")  # type: ignore[arg-type]

    def test_entity_type_accepts_value_string(self) -> None:
        assert entity("x", entity_type="PERSON").entity_type is EntityType.PERSON
        with pytest.raises(ValidationError):
            entity("x", entity_type="ANIMAL")


# ---------------------------------------------------------------------------
# Identifier, name and predicate sanitisation
# ---------------------------------------------------------------------------
class TestEntityIds:
    @pytest.mark.parametrize("eid", ["tool-neo4j", "a", "x1-2-3", "a" * MAX_ENTITY_ID_LENGTH])
    def test_slugs_are_accepted_verbatim(self, eid: str) -> None:
        assert entity(eid).id == eid

    def test_uuids_are_canonicalised_to_lowercase(self) -> None:
        assert entity(UUID_UPPER).id == UUID_UPPER.lower()

    @pytest.mark.parametrize(
        "eid",
        ["", "Tool-Neo4j", "tool--neo4j", "-tool", "tool-", "tool neo4j", "tool_neo4j",
         "{123e4567-e89b-12d3-a456-426614174000}", "a" * (MAX_ENTITY_ID_LENGTH + 1)],
    )  # fmt: skip
    def test_malformed_ids_are_rejected(self, eid: str) -> None:
        with pytest.raises(ValidationError):
            entity(eid, "Name")


class TestMakeEntityId:
    def test_slug_is_prefixed_with_lowercase_type(self) -> None:
        assert make_entity_id("  Ada   Lovelace ", EntityType.PERSON) == "person-ada-lovelace"

    def test_diacritics_are_transliterated(self) -> None:
        assert make_entity_id("Kraków", EntityType.LOCATION) == "location-krakow"

    def test_type_disambiguates_same_name(self) -> None:
        assert make_entity_id("Python", EntityType.TOOL) != make_entity_id(
            "Python", EntityType.CONCEPT
        )

    def test_non_ascii_names_fall_back_to_uuid5(self) -> None:
        derived = make_entity_id("東京", EntityType.LOCATION)
        assert derived == str(uuid.uuid5(ENTITY_NAMESPACE, "LOCATION:東京"))
        assert entity(derived, "東京").id == derived

    def test_overlong_slugs_fall_back_to_case_insensitive_uuid5(self) -> None:
        long_name = "Very " * 40
        derived = make_entity_id(long_name, EntityType.EVENT)
        assert uuid.UUID(derived).version == 5
        assert derived == make_entity_id(long_name.upper(), EntityType.EVENT)

    def test_derived_ids_are_valid_entity_ids(self) -> None:
        for name in ("GraphRAG v2.0!", "O'Reilly Media", "C++"):
            derived = make_entity_id(name, EntityType.ORGANIZATION)
            assert entity(derived, name).id == derived

    def test_unprintable_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="printable"):
            make_entity_id("\u200b\u202e", EntityType.MISC)


class TestNameSanitisation:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("  Neo4j  ", "Neo4j"),
            ("Graph\n\tDatabase", "Graph Database"),
            ("Neo\u200b4j", "Neo4j"),  # zero-width space
            ("evil\u202egnp.exe", "evilgnp.exe"),  # bidi override (Trojan Source)
            ("\uff2e\uff45\uff4f\uff14\uff4a", "Neo4j"),  # fullwidth → NFKC
            ("a\x00b\x07c", "abc"),  # C0 controls
            ("Ada\u00a0Lovelace", "Ada Lovelace"),  # no-break space
        ],
    )
    def test_names_are_normalised(self, raw: str, clean: str) -> None:
        assert entity("tool-x", raw).name == clean

    @pytest.mark.parametrize("raw", ["", "   ", "\u200b\u200d", "\x00\x01"])
    def test_names_without_printable_characters_are_rejected(self, raw: str) -> None:
        with pytest.raises(ValidationError, match="printable"):
            entity("tool-x", raw)

    def test_overlong_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds 256"):
            entity("tool-x", "n" * 257)

    def test_oversized_raw_input_is_rejected_before_normalisation(self) -> None:
        with pytest.raises(ValidationError, match="exceeds 256"):
            entity("tool-x", " " * 2000 + "n")

    def test_description_keeps_line_structure(self) -> None:
        raw = "Line  one\r\n\r\n\r\n\r\n- item  a  \r    code\tblock\u200b\x07\n \t \nend"
        expected = "Line one\n\n- item a\n    code block\n\nend"
        assert entity("tool-x", description=raw).description == expected

    def test_description_may_be_empty_but_bounded(self) -> None:
        assert entity("tool-x", description="  \n ").description == ""
        with pytest.raises(ValidationError, match="exceeds 4096"):
            entity("tool-x", description="d" * 4097)


class TestPredicates:
    @pytest.mark.parametrize(
        ("raw", "normalised"),
        [
            ("DEPENDS_ON", "DEPENDS_ON"),
            ("depends on", "DEPENDS_ON"),
            ("depends-on", "DEPENDS_ON"),
            ("dependsOn", "DEPENDS_ON"),
            ("  uses  ", "USES"),
            ("__part__of__", "PART_OF"),
            ("HAS_V2", "HAS_V2"),
            ("\uff55\uff53\uff45\uff53", "USES"),  # fullwidth → NFKC
        ],
    )
    def test_predicates_normalise_to_screaming_snake_case(self, raw: str, normalised: str) -> None:
        assert relation("a", raw, "b").predicate == normalised

    @pytest.mark.parametrize("raw", ["", "___", "2FA_ENABLES", "używa", "USES\u200b", "-"])
    def test_invalid_predicates_are_rejected(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            relation("a", raw, "b")

    def test_overlong_predicates_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="SCREAMING_SNAKE_CASE"):
            relation("a", "P" * (MAX_PREDICATE_LENGTH + 1), "b")

    def test_cypher_metacharacters_cannot_survive(self) -> None:
        hostile = "USES]->(x) DETACH DELETE n //`"
        predicate = relation("a", hostile, "b").predicate
        assert set(predicate) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


# ---------------------------------------------------------------------------
# JSON payload freezing
# ---------------------------------------------------------------------------
class TestJsonPayloads:
    def test_round_trip_through_freeze_and_thaw(self) -> None:
        payload = {"s": "x", "i": 1, "f": 0.5, "b": True, "n": None, "l": [1, {"k": [2]}]}
        frozen = freeze_json_mapping(payload)
        assert frozen["l"] == (1, MappingProxyType({"k": (2,)}))
        assert thaw_json(frozen) == payload

    @pytest.mark.parametrize(
        "payload",
        [
            {"f": float("nan")},
            {"f": [float("-inf")]},
            {"s": {1, 2}},
            {"o": object()},
            {"": 1},
            {"k" * 129: 1},
            {"nested": {1: "non-string key"}},
        ],
    )
    def test_non_json_payloads_are_rejected(self, payload: Mapping[Any, Any]) -> None:
        with pytest.raises(ValueError, match=r"\$"):
            freeze_json_mapping(payload)

    def test_nesting_is_bounded(self) -> None:
        payload: dict[str, Any] = {}
        cursor = payload
        for _ in range(MAX_JSON_NESTING + 1):
            cursor["k"] = {}
            cursor = cursor["k"]
        with pytest.raises(ValueError, match="nesting"):
            freeze_json_mapping(payload)

    def test_invalid_payload_surfaces_as_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="unsupported JSON value"):
            entity("tool-x", attributes={"when": object()})


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------
class TestKnowledgeGraphIntegrity:
    def test_clean_triplets_pass(self, graph: KnowledgeGraph) -> None:
        assert len(graph.entities) == 3
        assert {r.key for r in graph.relations} == {
            ("concept-graphrag", "USES", "tool-neo4j"),
            ("concept-graphrag", "USES", "tool-qdrant"),
        }

    def test_empty_graph_is_valid(self) -> None:
        assert KnowledgeGraph() == KnowledgeGraph.model_validate({"entities": [], "relations": []})

    def test_self_loops_on_known_entities_pass(self) -> None:
        KnowledgeGraph(entities=(entity("a"),), relations=(relation("a", "RELATES_TO", "a"),))

    @pytest.mark.parametrize(
        ("source", "target", "missing"),
        [
            ("ghost", "a", ["ghost"]),
            ("a", "ghost", ["ghost"]),
            ("ghost-1", "ghost-2", ["ghost-1", "ghost-2"]),
        ],
    )
    def test_dangling_relations_fail(self, source: str, target: str, missing: list[str]) -> None:
        with pytest.raises(ValidationError) as exc:
            KnowledgeGraph(entities=(entity("a"),), relations=(relation(source, "USES", target),))
        (error,) = exc.value.errors()
        assert error["type"] == "dangling_relation"
        assert error["ctx"] == {"dangling_ids": missing}

    def test_relations_without_entities_fail(self) -> None:
        with pytest.raises(ValidationError, match="dangling_relation"):
            KnowledgeGraph(relations=(relation("a", "USES", "b"),))

    def test_dangling_check_uses_normalised_ids(self) -> None:
        KnowledgeGraph(
            entities=(entity(UUID_UPPER.lower()),),
            relations=(relation(UUID_UPPER, "SAME_AS", UUID_UPPER),),
        )

    def test_duplicate_entity_ids_fail(self) -> None:
        with pytest.raises(ValidationError) as exc:
            KnowledgeGraph(
                entities=(
                    entity("a", "One"),
                    entity("a", "Two"),
                    entity("b"),
                )
            )
        (error,) = exc.value.errors()
        assert error["type"] == "duplicate_entity"
        assert error["ctx"] == {"duplicate_ids": ["a"]}

    def test_duplicate_triplets_fail_even_when_predicate_spelling_differs(self) -> None:
        with pytest.raises(ValidationError) as exc:
            KnowledgeGraph(
                entities=(
                    entity("a"),
                    entity("b"),
                ),
                relations=(
                    relation("a", "DEPENDS_ON", "b"),
                    relation("a", "dependsOn", "b"),
                ),
            )
        (error,) = exc.value.errors()
        assert error["type"] == "duplicate_relation"
        assert error["ctx"] == {"duplicate_triplets": ["a|DEPENDS_ON|b"]}

    def test_reverse_direction_is_a_distinct_triplet(self) -> None:
        KnowledgeGraph(
            entities=(
                entity("a"),
                entity("b"),
            ),
            relations=(
                relation("a", "KNOWS", "b"),
                relation("b", "KNOWS", "a"),
            ),
        )


# ---------------------------------------------------------------------------
# Retrieval contracts
# ---------------------------------------------------------------------------
class TestMemoryQuery:
    def test_defaults(self) -> None:
        query = MemoryQuery(text="  what does GraphRAG use?\n")
        assert query.text == "what does GraphRAG use?"
        assert (query.top_k, query.score_threshold, query.traversal_depth) == (5, 0.0, 2)
        assert query.entity_types is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("top_k", 0),
            ("top_k", MAX_TOP_K + 1),
            ("top_k", "5"),
            ("top_k", 5.0),
            ("score_threshold", 1.5),
            ("score_threshold", -0.1),
            ("traversal_depth", 0),
            ("traversal_depth", MAX_TRAVERSAL_DEPTH + 1),
            ("traversal_depth", True),
            ("entity_types", []),
            ("entity_types", ["ANIMAL"]),
            ("text", "\u200b"),
        ],
    )
    def test_bounds_are_enforced(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            MemoryQuery.model_validate({"text": "q", field: value})

    @pytest.mark.parametrize("depth", range(1, MAX_TRAVERSAL_DEPTH + 1))
    def test_traversal_depth_range(self, depth: int) -> None:
        assert MemoryQuery(text="q", traversal_depth=depth).traversal_depth == depth

    def test_entity_type_filter_is_canonical(self) -> None:
        a = MemoryQuery.model_validate({"text": "q", "entity_types": ["TOOL", "PERSON", "TOOL"]})
        b = MemoryQuery.model_validate({"text": "q", "entity_types": ["PERSON", "TOOL"]})
        assert a.entity_types == (EntityType.PERSON, EntityType.TOOL)
        assert a.model_dump_json() == b.model_dump_json()


class TestGraphHop:
    def test_from_relation_and_rendering(self, graph: KnowledgeGraph) -> None:
        hop = GraphHop.from_relation(graph.relations[0], depth=2)
        assert (hop.weight, hop.depth) == (0.9, 2)
        assert str(hop) == "(concept-graphrag)-[USES]->(tool-neo4j)"

    def test_depth_is_bounded(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValidationError):
            GraphHop.from_relation(graph.relations[0], depth=MAX_TRAVERSAL_DEPTH + 1)

    def test_hop_endpoints_are_sanitised_like_relations(self) -> None:
        hop = GraphHop(source_id=UUID_UPPER, predicate="part of", target_id="b")
        assert (hop.source_id, hop.predicate) == (UUID_UPPER.lower(), "PART_OF")


class TestGraphContext:
    def test_empty_context(self) -> None:
        context = GraphContext.empty()
        assert (context.seed_entities, context.paths, context.summary) == ((), (), "")
        assert context.confidence == 0.0

    def test_populated_context(self, graph: KnowledgeGraph) -> None:
        context = GraphContext(
            seed_entities=graph.entities[:1],
            paths=tuple(GraphHop.from_relation(r, depth=1) for r in graph.relations),
            summary="## GraphRAG\r\n\r\n- uses **Neo4j**  \n- uses Qdrant\u200b",
            confidence=0.8,
        )
        assert context.summary == "## GraphRAG\n\n- uses **Neo4j**\n- uses Qdrant"

    def test_duplicate_seeds_fail(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(ValidationError) as exc:
            GraphContext(
                seed_entities=(
                    graph.entities[0],
                    graph.entities[0],
                ),
                confidence=0.5,
            )
        (error,) = exc.value.errors()
        assert error["type"] == "duplicate_seed"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"paths": [GraphHop(source_id="a", predicate="USES", target_id="b")]},
            {"confidence": 0.1},
        ],
    )
    def test_seedless_context_cannot_claim_results(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="seedless_context"):
            GraphContext(**kwargs)

    def test_confidence_and_summary_bounds(self, graph: KnowledgeGraph) -> None:
        seeds = graph.entities[:1]
        with pytest.raises(ValidationError):
            GraphContext(seed_entities=seeds, confidence=1.1)
        with pytest.raises(ValidationError, match="exceeds 32768"):
            GraphContext(seed_entities=seeds, summary="s" * 32_769)


# ---------------------------------------------------------------------------
# Serialisation stability
# ---------------------------------------------------------------------------
def _round_trip_cases() -> list[BaseModel]:
    neo4j = entity("tool-neo4j", "Neo4j", embedding=[0.25, -1.0], attributes={"v": [5, 18]})
    graphrag = entity("concept-graphrag", "GraphRAG", entity_type=EntityType.CONCEPT)
    uses = relation("concept-graphrag", "uses", "tool-neo4j", weight=0.75, properties={"a": None})
    return [
        neo4j,
        uses,
        KnowledgeGraph(
            entities=(
                neo4j,
                graphrag,
            ),
            relations=(uses,),
        ),
        MemoryQuery(text="q", top_k=3, entity_types=(EntityType.TOOL,), score_threshold=0.2),
        GraphHop.from_relation(uses, depth=3),
        GraphContext(
            seed_entities=(graphrag,),
            paths=(GraphHop.from_relation(uses, depth=1),),
            summary="# Context\n\n- (concept-graphrag)-[USES]->(tool-neo4j)",
            confidence=0.66,
        ),
        MemoryProblemDetails.from_code(
            ErrorCode.ENTITY_INTEGRITY, "dangling", instance="/graphs/1", context={"ids": ["x"]}
        ),
    ]


class TestSerialisation:
    @pytest.mark.parametrize("model", _round_trip_cases(), ids=lambda m: type(m).__name__)
    def test_json_round_trip_is_lossless_and_byte_stable(self, model: BaseModel) -> None:
        dumped = model.model_dump_json()
        restored = type(model).model_validate_json(dumped)
        assert restored == model
        assert restored.model_dump_json() == dumped

    @pytest.mark.parametrize("model", _round_trip_cases(), ids=lambda m: type(m).__name__)
    def test_python_round_trip_is_lossless(self, model: BaseModel) -> None:
        assert type(model).model_validate(model.model_dump()) == model

    def test_wire_shape_uses_plain_json_types(self, graph: KnowledgeGraph) -> None:
        wire = json.loads(graph.model_dump_json())
        neo4j, qdrant, _ = wire["entities"]
        assert neo4j["embedding"] == [0.1, 0.2]
        assert neo4j["entity_type"] == "TOOL"
        assert qdrant["attributes"] == {"license": "Apache-2.0", "tags": ["v"]}
        assert wire["relations"][1]["predicate"] == "USES"

    def test_python_dump_thaws_payloads(self, graph: KnowledgeGraph) -> None:
        attributes = graph.model_dump()["entities"][1]["attributes"]
        assert type(attributes) is dict
        assert type(attributes["tags"]) is list

    def test_traversal_serialises_in_traversal_order(self, graph: KnowledgeGraph) -> None:
        hops = tuple(GraphHop.from_relation(r, depth=i + 1) for i, r in enumerate(graph.relations))
        context = GraphContext(seed_entities=graph.entities[2:], paths=hops, confidence=0.5)
        wire = json.loads(context.model_dump_json())
        assert [(h["target_id"], h["depth"]) for h in wire["paths"]] == [
            ("tool-neo4j", 1),
            ("tool-qdrant", 2),
        ]


# ---------------------------------------------------------------------------
# Error taxonomy and RFC 9457 payloads
# ---------------------------------------------------------------------------
class TestProblemDetails:
    def test_taxonomy_covers_every_code_with_unique_types(self) -> None:
        assert set(PROBLEM_TAXONOMY) == set(ErrorCode)
        types = [spec.type for spec in PROBLEM_TAXONOMY.values()]
        assert len(set(types)) == len(types)
        assert all(t.startswith("urn:agent-memory-graph:problem:") for t in types)

    def test_media_type(self) -> None:
        assert PROBLEM_JSON_MEDIA_TYPE == "application/problem+json"

    def test_from_code_fills_canonical_members(self) -> None:
        problem = MemoryProblemDetails.from_code(ErrorCode.VECTOR_SEARCH_TIMEOUT, "slow")
        assert problem.to_json_dict() == {
            "type": "urn:agent-memory-graph:problem:vector-search-timeout",
            "title": "Vector search exceeded its deadline",
            "status": 504,
            "detail": "slow",
            "code": "AMG_VECTOR_SEARCH_TIMEOUT",
            "context": {},
        }

    @pytest.mark.parametrize(
        "override",
        [
            {"status": 500},
            {"type": "urn:agent-memory-graph:problem:internal-error"},
            {"title": "Something else"},
        ],
    )
    def test_members_must_agree_with_taxonomy(self, override: dict[str, Any]) -> None:
        body = MemoryProblemDetails.from_code(ErrorCode.ENTITY_INTEGRITY, "x").to_json_dict()
        with pytest.raises(ValidationError, match="do not match AMG_ENTITY_INTEGRITY"):
            MemoryProblemDetails.model_validate({**body, **override})

    @pytest.mark.parametrize(
        "override",
        [{"detail": ""}, {"instance": "has space"}, {"instance": ""}, {"status": 200}],
    )
    def test_members_are_validated(self, override: dict[str, Any]) -> None:
        body = MemoryProblemDetails.from_code(ErrorCode.INTERNAL, "x").to_json_dict()
        with pytest.raises(ValidationError):
            MemoryProblemDetails.model_validate({**body, **override})


class TestDomainErrors:
    def test_base_error_maps_to_internal_problem(self) -> None:
        error = MemoryGraphError("unexpected", instance="/requests/42", context={"k": 1})
        assert str(error) == "unexpected"
        assert error.status == 500
        problem = error.to_problem()
        assert (problem.code, problem.instance) == (ErrorCode.INTERNAL, "/requests/42")
        assert problem.context == {"k": 1}

    def test_error_context_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            MemoryGraphError("x").context["k"] = 1  # type: ignore[index]

    def test_entity_integrity_error(self) -> None:
        error = EntityIntegrityError(
            "relations reference missing entities",
            dangling_ids=["b", "a", "b"],
            duplicate_ids=("c",),
        )
        assert isinstance(error, MemoryGraphError)
        assert (error.dangling_ids, error.duplicate_ids) == (("a", "b"), ("c",))
        body = error.to_problem().to_json_dict()
        assert (body["status"], body["code"]) == (422, "AMG_ENTITY_INTEGRITY")
        assert body["context"] == {"dangling_ids": ["a", "b"], "duplicate_ids": ["c"]}

    def test_traversal_depth_exceeded_error(self) -> None:
        error = TraversalDepthExceededError(5, MAX_TRAVERSAL_DEPTH)
        assert (error.requested, error.maximum, error.status) == (5, 3, 422)
        problem = error.to_problem()
        assert problem.detail == "Requested traversal depth 5 exceeds the maximum of 3."
        assert problem.context == {"requested_depth": 5, "max_depth": 3}

    @pytest.mark.parametrize(
        ("collection", "detail", "context"),
        [
            (None, "Vector search did not complete within 2.5s.", {"timeout_seconds": 2.5}),
            (
                "memories",
                "Vector search on collection 'memories' did not complete within 2.5s.",
                {"timeout_seconds": 2.5, "collection": "memories"},
            ),
        ],
    )
    def test_vector_search_timeout_error(
        self, collection: str | None, detail: str, context: dict[str, Any]
    ) -> None:
        error = VectorSearchTimeoutError(2.5, collection=collection)
        assert (error.timeout_seconds, error.collection, error.status) == (2.5, collection, 504)
        problem = error.to_problem()
        assert (problem.detail, problem.context) == (detail, context)

    @pytest.mark.parametrize(
        ("attempts", "context"),
        [(None, {"reason": "bad output"}), (3, {"reason": "bad output", "attempts": 3})],
    )
    def test_extraction_failed_error(self, attempts: int | None, context: dict[str, Any]) -> None:
        error = ExtractionFailedError("bad output", attempts=attempts)
        assert (error.reason, error.attempts, error.status) == ("bad output", attempts, 502)
        problem = error.to_problem()
        assert problem.detail == "Knowledge extraction failed: bad output."
        assert problem.context == context

    def test_invalid_input_error_never_echoes_input(self) -> None:
        error = InvalidInputError(
            "bad request",
            errors=[
                {"loc": ("body", "text", 0), "msg": "too long", "type": "string_too_long",
                 "input": "secret-value", "ctx": {"max_length": 3}},
                {"loc": (), "msg": 42},
                {},
            ],
        )  # fmt: skip
        assert error.status == 422
        assert error.to_problem().to_json_dict()["context"] == {
            "errors": [
                {"loc": ["body", "text", 0], "msg": "too long", "type": "string_too_long"},
                {"loc": [], "msg": "42", "type": "value_error"},
                {"loc": [], "msg": "", "type": "value_error"},
            ]
        }
        assert "secret-value" not in str(error.to_problem().to_json_dict())

    def test_extraction_timeout_error(self) -> None:
        error = ExtractionTimeoutError(30.0)
        assert (error.timeout_seconds, error.status) == (30.0, 504)
        assert error.to_problem().detail == "Knowledge extraction did not complete within 30s."

    def test_store_unavailable_error(self) -> None:
        error = StoreUnavailableError("neo4j")
        assert (error.store, error.status) == ("neo4j", 503)
        problem = error.to_problem()
        assert (problem.detail, problem.context) == (
            "The neo4j store is unavailable.",
            {"store": "neo4j"},
        )

    def test_store_unavailable_error_with_retry_hint(self) -> None:
        error = StoreUnavailableError("qdrant", retry_after_seconds=12)
        assert error.retry_after_seconds == 12
        problem = error.to_problem()
        assert problem.detail == "The qdrant store is unavailable; retry in 12s."
        assert problem.context == {"store": "qdrant", "retry_after_seconds": 12}

    def test_store_timeout_error(self) -> None:
        error = StoreTimeoutError("qdrant", "upsert_entities", 1.5)
        assert (error.store, error.operation, error.timeout_seconds) == (
            "qdrant",
            "upsert_entities",
            1.5,
        )
        problem = error.to_problem()
        assert problem.status == 504
        assert (
            problem.detail == "The qdrant upsert_entities operation did not complete within 1.5s."
        )
        assert problem.context == {
            "store": "qdrant",
            "operation": "upsert_entities",
            "timeout_seconds": 1.5,
        }

    @pytest.mark.parametrize(
        "error_type",
        [
            EntityIntegrityError,
            TraversalDepthExceededError,
            VectorSearchTimeoutError,
            StoreUnavailableError,
            StoreTimeoutError,
            ExtractionFailedError,
            ExtractionTimeoutError,
            InvalidInputError,
        ],
    )
    def test_each_error_has_its_own_code(self, error_type: type[MemoryGraphError]) -> None:
        assert error_type.code is not MemoryGraphError.code
        assert issubclass(error_type, MemoryGraphError)


def test_line_and_multiline_sanitisers_are_exposed() -> None:
    assert sanitize_line(" a \n b ", max_length=10) == "a b"
    assert sanitize_multiline(" a \n b ", max_length=10) == "a\n b"
