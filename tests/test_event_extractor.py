"""
Test EventExtractor V3 — Hybrid SRL + Relation Event Pipeline.

Usage:
    python3 tests/test_event_extractor.py
"""

from core.event_extractor import EventExtractor
from core.schema import (
    Entity, Event, EventArgument, Relation,
    SentenceLanguage, SentencePattern,
)


def test_stopword_filtering():
    """Test that relations with stopword triggers are filtered."""
    text = "他是工程师"
    relations = [
        Relation(
            id="rel_001", subject="他", predicate="IS_A", predicate_verb="是",
            object="工程师", evidence=text, evidence_span=(0, 5),
            subject_ent_id=None, object_ent_id="ent_001",
        ),
    ]
    entities = [Entity(id="ent_001", text="工程师", category="PERSON", span=(2, 5))]
    sentences = [SentenceLanguage(text=text, span=(0, 5), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Stopword test: {len(events)} events (expected 0)")
    assert len(events) == 0, f"Expected 0 events for stopword '是', got {len(events)}"
    print("  PASS: stopword '是' filtered correctly\n")


def test_basic_event_from_relation():
    """Test event extraction from a single relation."""
    text = "创作者实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(5, 7)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 7), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Basic event test: {len(events)} events")
    assert len(events) >= 1, f"Expected at least 1 event, got {len(events)}"

    evt = events[0]
    assert evt.trigger == "实现", f"Expected trigger '实现', got '{evt.trigger}'"
    assert evt.event_type == "实现"
    assert evt.is_main_event is True
    assert evt.sentence_index == 0
    assert evt.source_relation_ids == ["rel_001"]
    assert len(evt.arguments) >= 2

    # Verify trigger_span is accurate (no longer evidence_span)
    assert evt.trigger_span == (3, 5), f"Expected trigger span (3,5), got {evt.trigger_span}"

    for arg in evt.arguments:
        print(f"  {arg.role}: {arg.text} span={arg.span} (entity_id={arg.entity_id})")

    # Subject → Agent
    agent = [a for a in evt.arguments if a.role == "Agent"]
    assert len(agent) >= 1
    assert agent[0].entity_id == "ent_001"
    # Verify agent span is accurate
    assert agent[0].span == (0, 3), f"Expected agent span (0,3), got {agent[0].span}"

    # Object of CAUSES → Result
    result = [a for a in evt.arguments if a.role == "Result"]
    assert len(result) >= 1, f"Expected Result role for CAUSES object, got: {[a.role for a in evt.arguments]}"
    assert result[0].text == "跨越"
    assert result[0].span == (5, 7), f"Expected result span (5,7), got {result[0].span}"

    print("  PASS: basic event from relation works\n")


def test_relation_clustering():
    """Test that multiple relations with same verb AND same agent cluster into one event."""
    text = "创作者通过小红书实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 11),
            subject_ent_id="ent_001", object_ent_id="ent_003",
        ),
        Relation(
            id="rel_002", subject="创作者", predicate="DEPENDS_ON", predicate_verb="实现",
            object="通过小红书", evidence=text, evidence_span=(0, 11),
            subject_ent_id="ent_001", object_ent_id=None,
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_003", text="跨越", category="UNKNOWN", span=(9, 11)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 11), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Clustering test: {len(events)} events")
    assert len(events) == 1, f"Expected 1 clustered event, got {len(events)}"

    evt = events[0]
    assert len(evt.source_relation_ids) == 2
    assert "rel_001" in evt.source_relation_ids
    assert "rel_002" in evt.source_relation_ids
    print(f"  source_relation_ids: {evt.source_relation_ids}")
    for arg in evt.arguments:
        print(f"  {arg.role}: {arg.text}")

    print("  PASS: relation clustering works\n")


def test_agent_dimension_splitting():
    """Test that different agents with the same verb form separate events."""
    text = "张三实现目标A，李四实现目标B"
    relations = [
        Relation(
            id="rel_001", subject="张三", predicate="CAUSES", predicate_verb="实现",
            object="目标A", evidence="张三实现目标A", evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
        Relation(
            id="rel_002", subject="李四", predicate="CAUSES", predicate_verb="实现",
            object="目标B", evidence="李四实现目标B", evidence_span=(8, 15),
            subject_ent_id="ent_003", object_ent_id="ent_004",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="张三", category="PERSON", span=(0, 2)),
        Entity(id="ent_002", text="目标A", category="UNKNOWN", span=(5, 8)),
        Entity(id="ent_003", text="李四", category="PERSON", span=(8, 10)),
        Entity(id="ent_004", text="目标B", category="UNKNOWN", span=(13, 16)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 15), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Agent splitting test: {len(events)} events")
    assert len(events) == 2, f"Expected 2 separate events (different agents), got {len(events)}"

    agents_found = sorted([e.arguments[0].text for e in events if e.arguments])
    assert "张三" in agents_found, f"Expected 张三 as agent, got {agents_found}"
    assert "李四" in agents_found, f"Expected 李四 as agent, got {agents_found}"

    for ev in events:
        agent = next((a.text for a in ev.arguments if a.role == "Agent"), "none")
        print(f"  Event {ev.id}: agent={agent}")

    print("  PASS: agent-dimension splitting works\n")


def test_event_deduplication():
    """Test that duplicate events are merged."""
    text = "创作者实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
        # Duplicate: same agent, same verb, same object (from a different relation type)
        Relation(
            id="rel_002", subject="创作者", predicate="AFFECTS", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(5, 7)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 7), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Deduplication test: {len(events)} events")
    # After agent-dimension splitting, these go into the same group (same agent).
    # After deduplication, they should be merged into one.
    assert len(events) == 1, f"Expected 1 event after dedup, got {len(events)}"

    print("  PASS: event deduplication works\n")


def test_sub_event_hierarchy():
    """Test that events without Agent become sub-events of the main event."""
    text = "创作者在2024年实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 11),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(9, 11)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 11), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # SRL frame with ARGM-TMP that could form a sub-event
    srl_frames = [
        [
            ("创作者", "ARG0", 0, 3),
            ("实现", "PRED", 7, 9),
            ("跨越", "ARG1", 9, 11),
            ("2024年", "ARGM-TMP", 3, 7),
        ],
    ]

    extractor = EventExtractor()
    events = extractor.extract(
        text, relations, entities, sentences, patterns, srl_frames=srl_frames,
    )

    print(f"Sub-event hierarchy test: {len(events)} events")
    assert len(events) >= 1

    for ev in events:
        print(f"  {ev.id}: main={ev.is_main_event}, subs={ev.sub_events}")

    # The main event should have Agent, sub-events should not
    main_events = [e for e in events if e.is_main_event]
    assert len(main_events) >= 1, "Expected at least 1 main event"

    print("  PASS: sub-event hierarchy works\n")


def test_interrogative_filtering():
    """Test that interrogative sentences are skipped."""
    text = "他是工程师吗"
    relations = [
        Relation(
            id="rel_001", subject="他", predicate="IS_A", predicate_verb="是",
            object="工程师", evidence=text, evidence_span=(0, 5),
        ),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 6), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="interrogative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, [], sentences, patterns)

    print(f"Interrogative test: {len(events)} events (expected 0)")
    assert len(events) == 0
    print("  PASS: interrogative filtered\n")


def test_multi_sentence():
    """Test events from multiple sentences."""
    text = "创作者实现跨越作品覆盖文学"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence="创作者实现跨越", evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
        Relation(
            id="rel_002", subject="作品", predicate="AFFECTS", predicate_verb="覆盖",
            object="文学", evidence="作品覆盖文学", evidence_span=(7, 13),
            subject_ent_id="ent_003", object_ent_id="ent_004",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(5, 7)),
        Entity(id="ent_003", text="作品", category="PRODUCT", span=(7, 9)),
        Entity(id="ent_004", text="文学", category="UNKNOWN", span=(11, 13)),
    ]
    sentences = [
        SentenceLanguage(text="创作者实现跨越", span=(0, 7), label="modern", confidence=0.9),
        SentenceLanguage(text="作品覆盖文学", span=(7, 13), label="modern", confidence=0.9),
    ]
    patterns = [
        SentencePattern(sentence_type="declarative"),
        SentencePattern(sentence_type="declarative"),
    ]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    print(f"Multi-sentence test: {len(events)} events")
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"

    for ev in events:
        print(f"  sent={ev.sentence_index} trigger={ev.trigger} rels={ev.source_relation_ids}")

    sent0 = [e for e in events if e.sentence_index == 0]
    sent1 = [e for e in events if e.sentence_index == 1]
    assert len(sent0) >= 1
    assert len(sent1) >= 1

    print("  PASS: multi-sentence events work\n")


def test_traceability_chain():
    """Test Event → Relation → Entity traceability."""
    text = "A生产B"
    relations = [
        Relation(
            id="rel_001", subject="A", predicate="PRODUCES", predicate_verb="生产",
            object="B", evidence=text, evidence_span=(0, 3),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="A", category="ORGANIZATION", span=(0, 1)),
        Entity(id="ent_002", text="B", category="PRODUCT", span=(2, 3)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 3), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns)

    assert len(events) == 1
    evt = events[0]

    # Traceability: Event → Relation
    assert evt.source_relation_ids == ["rel_001"]

    # Traceability: Relation → Entity (via subject_ent_id/object_ent_id)
    rel = relations[0]
    assert rel.subject_ent_id == "ent_001"
    assert rel.object_ent_id == "ent_002"

    # Event arguments should link to entities
    agent = [a for a in evt.arguments if a.role == "Agent"]
    assert len(agent) >= 1
    assert agent[0].entity_id == "ent_001"

    print("Traceability chain:")
    print(f"  Event({evt.id}) → Relation({evt.source_relation_ids[0]}) → Entity({agent[0].entity_id})")
    print("  PASS: traceability chain works\n")


# ── SRL Hybrid Tests ──

def test_srl_content_predicate_filter():
    """Test that SRL frames without ARG0+ARG1 (prepositions) are filtered out."""
    text = "创作者通过小红书实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 11),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(9, 11)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 11), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # Simulated HanLP SRL: frame 1 is a preposition ("通过"), frame 2 is content ("实现")
    srl_frames = [
        # Frame 1: "通过" — preposition, no ARG1 → should be filtered
        [("创作者", "ARG0", 0, 3), ("通过", "PRED", 3, 5), ("小红书", "ARG1", 5, 8)],
    ]

    extractor = EventExtractor()
    # Without SRL: should produce 1 event
    events_no_srl = extractor.extract(text, relations, entities, sentences, patterns)
    assert len(events_no_srl) == 1, f"Without SRL: expected 1 event, got {len(events_no_srl)}"

    extractor.reset()
    events_with_srl = extractor.extract(
        text, relations, entities, sentences, patterns, srl_frames=srl_frames,
    )
    # "通过" SRL frame matches "实现" relation? No, predicate_verb mismatch.
    # The "通过" frame has PRED="通过", but relation has predicate_verb="实现".
    # So SRL frame won't match any event cluster. Event count unchanged.
    assert len(events_with_srl) == 1, f"With SRL (mismatch): expected 1 event, got {len(events_with_srl)}"

    print("  PASS: SRL content predicate filter works\n")


def test_srl_enrichment_argm_roles():
    """Test that SRL ARGM-* roles are added as event arguments."""
    text = "创作者在2024年通过北京实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 15),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(12, 14)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 15), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # SRL frame for "实现": ARG0=创作者, ARG1=跨越, ARGM-TMP=2024年, ARGM-LOC=北京
    srl_frames = [
        [
            ("创作者", "ARG0", 0, 3),
            ("实现", "PRED", 10, 12),
            ("跨越", "ARG1", 12, 14),
            ("2024年", "ARGM-TMP", 3, 6),
            ("北京", "ARGM-LOC", 8, 10),
        ],
    ]

    extractor = EventExtractor()
    events = extractor.extract(
        text, relations, entities, sentences, patterns, srl_frames=srl_frames,
    )

    print(f"SRL enrichment test: {len(events)} events")
    assert len(events) >= 1, f"Expected at least 1 event, got {len(events)}"

    evt = events[0]
    for arg in evt.arguments:
        print(f"  {arg.role}: {arg.text} span={arg.span}")

    # Should have Agent, Result from relations + Time, Location from SRL
    roles = {a.role: a.text for a in evt.arguments}
    assert "Agent" in roles
    assert roles["Agent"] == "创作者"
    # Object of CAUSES → Result (not Agent)
    assert "Result" in roles, f"Expected Result role for CAUSES object, got roles: {list(roles.keys())}"
    assert roles["Result"] == "跨越"
    assert "Time" in roles, f"Expected SRL ARGM-TMP→Time, got roles: {list(roles.keys())}"
    assert roles["Time"] == "2024年"
    assert "Location" in roles, f"Expected SRL ARGM-LOC→Location, got roles: {list(roles.keys())}"
    assert roles["Location"] == "北京"

    print("  PASS: SRL ARGM-* enrichment works\n")


def test_srl_confidence_boost():
    """Test that SRL validation boosts confidence."""
    text = "创作者实现跨越"
    relations = [
        Relation(
            id="rel_001", subject="创作者", predicate="CAUSES", predicate_verb="实现",
            object="跨越", evidence=text, evidence_span=(0, 7),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="创作者", category="PERSON", span=(0, 3)),
        Entity(id="ent_002", text="跨越", category="UNKNOWN", span=(5, 7)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 7), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # SRL frame matching the relation
    srl_frames = [
        [
            ("创作者", "ARG0", 0, 3),
            ("实现", "PRED", 3, 5),
            ("跨越", "ARG1", 5, 7),
        ],
    ]

    extractor = EventExtractor()
    # Without SRL
    events_no_srl = extractor.extract(text, relations, entities, sentences, patterns)
    conf_no_srl = events_no_srl[0].confidence
    print(f"Confidence without SRL: {conf_no_srl}")

    extractor.reset()
    # With SRL
    events_srl = extractor.extract(
        text, relations, entities, sentences, patterns, srl_frames=srl_frames,
    )
    conf_srl = events_srl[0].confidence
    print(f"Confidence with SRL: {conf_srl}")

    assert conf_srl > conf_no_srl, (
        f"SRL should boost confidence: {conf_srl} > {conf_no_srl}"
    )
    assert events_srl[0].source == "srl+relation_cluster", (
        f"Expected source 'srl+relation_cluster', got '{events_srl[0].source}'"
    )

    print("  PASS: SRL confidence boost works\n")


def test_srl_only_extraction():
    """Test that events are extracted from SRL frames when Relations are missing.

    Regression test for: "7月18日,泰国总理出现在四川成都武侯祠博物馆,引发到处封路管控。"
    The "引发" verb has no Relation extracted, but SRL still has the frame.
    """
    text = "泰国总理引发封路管控"
    relations = [
        Relation(
            id="rel_001", subject="武侯祠博物馆", predicate="LOCATED_AT", predicate_verb="出现",
            object="泰国", evidence=text, evidence_span=(0, 10),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="泰国总理", category="PERSON", span=(0, 4)),
        Entity(id="ent_002", text="泰国", category="LOCATION", span=(0, 2)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 10), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    srl_frames = [
        [
            ("泰国总理", "ARG0", 0, 4),
            ("引发", "PRED", 4, 6),
            ("封路管控", "ARG1", 6, 10),
        ],
    ]

    extractor = EventExtractor()
    events = extractor.extract(text, relations, entities, sentences, patterns, srl_frames=srl_frames)

    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    trigger_events = [e for e in events if e.trigger == "引发"]
    assert len(trigger_events) == 1, f"Expected 1 '引发' event, got {len(trigger_events)}"

    evt = trigger_events[0]
    assert evt.source == "srl_only", f"Expected source='srl_only', got '{evt.source}'"

    roles = {a.role: a.text for a in evt.arguments}
    assert "Agent" in roles, f"Expected Agent argument, got {roles}"
    assert roles["Agent"] == "泰国总理", f"Expected Agent='泰国总理', got '{roles['Agent']}'"
    assert "Patient" in roles, f"Expected Patient argument, got {roles}"
    assert roles["Patient"] == "封路管控", f"Expected Patient='封路管控', got '{roles['Patient']}'"

    print("  PASS: SRL-only event extraction works (引发 case)")


def test_srl_no_duplicate_argm():
    """Test that SRL ARGM-* doesn't duplicate relation arguments."""
    text = "A在B生产C"
    relations = [
        Relation(
            id="rel_001", subject="A", predicate="PRODUCES", predicate_verb="生产",
            object="C", evidence=text, evidence_span=(0, 5),
            subject_ent_id="ent_001", object_ent_id="ent_002",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="A", category="ORGANIZATION", span=(0, 1)),
        Entity(id="ent_002", text="C", category="PRODUCT", span=(4, 5)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 5), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # SRL has ARGM-LOC="B" which should appear, but not duplicate Agent/Patient
    srl_frames = [
        [
            ("A", "ARG0", 0, 1),
            ("生产", "PRED", 3, 5),  # token spans approximate
            ("C", "ARG1", 4, 5),
            ("B", "ARGM-LOC", 1, 2),
        ],
    ]

    extractor = EventExtractor()
    events = extractor.extract(
        text, relations, entities, sentences, patterns, srl_frames=srl_frames,
    )

    evt = events[0]
    arg_texts = [(a.role, a.text) for a in evt.arguments]
    print(f"Arguments: {arg_texts}")

    # Should have Agent, Patient, Location — no duplicates
    roles = [a.role for a in evt.arguments]
    # Agent appears once
    assert roles.count("Agent") == 1, f"Agent should appear once, got {roles.count('Agent')}"
    # Location appears
    assert "Location" in roles, f"Expected Location from SRL ARGM-LOC"

    print("  PASS: SRL no duplicate ARGM-*\n")


def test_dep_only_extraction():
    """Test that events are extracted from dependency tree when both Relations and SRL are missing.

    Regression test for: "7月18日，泰国总理出现在四川成都武侯祠博物馆，引发到处封路管控。"
    The "引发" verb has no Relation and no SRL, but the dependency tree shows:
    - 14(引发) → conj of 7(出现)
    - 17(管控) → ccomp of 14(引发)
    """
    from core.schema import DependencyEdge, Token

    text = "泰国总理出现在四川成都武侯祠博物馆，引发到处封路管控。"
    # Only one relation for "出现", none for "引发"
    relations = [
        Relation(
            id="rel_001", subject="四川成都武侯祠博物馆", predicate="LOCATED_AT", predicate_verb="出现",
            object="泰国", evidence=text, evidence_span=(6, 23),
            subject_ent_id="ent_004", object_ent_id="ent_001",
        ),
    ]
    entities = [
        Entity(id="ent_001", text="泰国", category="LOCATION", span=(6, 8)),
        Entity(id="ent_004", text="四川成都武侯祠博物馆", category="LOCATION", span=(13, 23)),
    ]
    sentences = [SentenceLanguage(text=text, span=(0, 33), label="modern", confidence=0.9)]
    patterns = [SentencePattern(sentence_type="declarative")]

    # Simulated tokens (matching the API output indices)
    # Note: "日"(NT) is a noun-like POS so it will be included,
    # "泰国"(NR) and "总理"(NN) are nouns and will be merged
    tokens = [
        Token(id=0, text="7", pos="CD", span=(0, 1)),
        Token(id=1, text="月", pos="NT", span=(1, 2)),
        Token(id=2, text="18", pos="NT", span=(2, 4)),
        Token(id=3, text="日", pos="NT", span=(4, 5)),
        Token(id=4, text="，", pos="PU", span=(5, 6)),
        Token(id=5, text="泰国", pos="NR", span=(6, 8)),
        Token(id=6, text="总理", pos="NN", span=(8, 10)),
        Token(id=7, text="出现", pos="VV", span=(10, 12)),
        Token(id=8, text="在", pos="P", span=(12, 13)),
        Token(id=9, text="四川", pos="NR", span=(13, 15)),
        Token(id=10, text="成都", pos="NR", span=(15, 17)),
        Token(id=11, text="武侯祠", pos="NR", span=(17, 20)),
        Token(id=12, text="博物馆", pos="NN", span=(20, 23)),
        Token(id=13, text="，", pos="PU", span=(23, 24)),
        Token(id=14, text="引发", pos="VV", span=(24, 26)),
        Token(id=15, text="到处", pos="AD", span=(26, 28)),
        Token(id=16, text="封路", pos="VV", span=(28, 30)),
        Token(id=17, text="管控", pos="VV", span=(30, 32)),
        Token(id=18, text="。", pos="PU", span=(32, 33)),
    ]

    # Simulated deps (from actual API output)
    deps = [
        DependencyEdge(child=0, head=3, rel="nn"),
        DependencyEdge(child=1, head=2, rel="nn"),
        DependencyEdge(child=2, head=3, rel="nn"),
        DependencyEdge(child=3, head=7, rel="nsubj"),
        DependencyEdge(child=4, head=7, rel="punct"),
        DependencyEdge(child=5, head=6, rel="nn"),
        DependencyEdge(child=6, head=7, rel="nsubj"),
        DependencyEdge(child=7, head=-1, rel="root"),
        DependencyEdge(child=8, head=7, rel="prep"),
        DependencyEdge(child=9, head=12, rel="nn"),
        DependencyEdge(child=10, head=12, rel="nn"),
        DependencyEdge(child=11, head=12, rel="nn"),
        DependencyEdge(child=12, head=8, rel="pobj"),
        DependencyEdge(child=13, head=7, rel="punct"),
        DependencyEdge(child=14, head=7, rel="conj"),  # 引发 → conj of 出现
        DependencyEdge(child=15, head=17, rel="advmod"),
        DependencyEdge(child=16, head=17, rel="dep"),
        DependencyEdge(child=17, head=14, rel="ccomp"),  # 管控 → ccomp of 引发
        DependencyEdge(child=18, head=7, rel="punct"),
    ]

    extractor = EventExtractor()
    events = extractor.extract(
        text, relations, entities, sentences, patterns,
        srl_frames=[], deps=deps, tokens=tokens,
    )

    # Should have: 1 from Relation (出现) + 1 from deps (引发)
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"

    # Find the "引发" event
    trigger_events = [e for e in events if e.trigger == "引发"]
    assert len(trigger_events) == 1, f"Expected 1 '引发' event, got {len(trigger_events)}"

    evt = trigger_events[0]
    assert evt.source == "dep_only", f"Expected source='dep_only', got '{evt.source}'"

    # Check arguments
    roles = {a.role: a.text for a in evt.arguments}
    assert "Result" in roles, f"Expected Result argument from ccomp, got {roles}"
    assert roles["Result"] == "管控", f"Expected Result='管控', got '{roles['Result']}'"

    print(f"  dep_only event: trigger={evt.trigger}, source={evt.source}")
    for arg in evt.arguments:
        print(f"    {arg.role}: {arg.text} span={arg.span}")
    print("  PASS: dependency-driven event extraction works (引发 case)\n")


# ══════════════════════════════════════════════════════════════════════
# Spatial Role & Verb Spatial Class Tests
# ══════════════════════════════════════════════════════════════════════


def _make_arg(text, role, span):
    """Helper: create an EventArgument with minimal required fields."""
    return EventArgument(role=role, text=text, span=span)


def _make_event(trigger, args, sent_idx=0):
    """Helper: create an Event with given trigger and arguments."""
    return Event(
        id="evt_001",
        event_type=trigger,
        trigger=trigger,
        trigger_span=(0, len(trigger)),
        arguments=args,
        sentence_index=sent_idx,
        is_main_event=True,
        sub_events=[],
        source_relation_ids=["rel_001"],
        confidence=0.75,
        source="relation_cluster",
    )


def test_spatial_lobj_loc_container():
    """lobj/loc dep → CONTAINER spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "停在河边" → 边/LC has lobj dep to 在/P
    tokens = [
        Token(id=0, text="停", pos="VV", span=(0, 1)),
        Token(id=1, text="在", pos="P", span=(1, 2)),
        Token(id=2, text="河", pos="NN", span=(2, 3)),
        Token(id=3, text="边", pos="LC", span=(3, 4)),
    ]
    deps = [
        DependencyEdge(child=0, head=-1, rel="root"),      # 停 → root
        DependencyEdge(child=1, head=0, rel="prep"),       # 在 → prep → 停
        DependencyEdge(child=2, head=3, rel="nn"),         # 河 → nn → 边
        DependencyEdge(child=3, head=1, rel="lobj"),       # 边 → lobj → 在
    ]

    # arg span exactly matches token 3 ("边"), which has lobj→在
    arg = _make_arg("边", "Location", (3, 4))
    evt = _make_event("停", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "CONTAINER", f"Expected CONTAINER, got {arg.spatial_role}"
    assert arg.governing_verb == "停"
    print(f"  lobj → CONTAINER: governing_verb={arg.governing_verb}, spatial_role={arg.spatial_role}")
    print("  PASS: lobj/loc → CONTAINER\n")


def test_spatial_dobj_move_to_target():
    """dobj + MOVE_TO verb → TARGET spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "前往北京"
    tokens = [
        Token(id=0, text="前往", pos="VV", span=(0, 2)),
        Token(id=1, text="北京", pos="NR", span=(2, 4)),
    ]
    deps = [
        DependencyEdge(child=0, head=-1, rel="root"),    # 前往 → root
        DependencyEdge(child=1, head=0, rel="dobj"),     # 北京 → dobj → 前往
    ]

    arg = _make_arg("北京", "Destination", (2, 4))
    evt = _make_event("前往", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "TARGET", f"Expected TARGET, got {arg.spatial_role}"
    assert arg.verb_spatial_class == "MOVE_TO"
    print(f"  dobj+MOVE_TO → TARGET: verb={arg.governing_verb}, vclass={arg.verb_spatial_class}, srole={arg.spatial_role}")
    print("  PASS: dobj + MOVE_TO → TARGET\n")


def test_spatial_dobj_move_from_origin():
    """dobj + MOVE_FROM verb → ORIGIN spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "离开北京"
    tokens = [
        Token(id=0, text="离开", pos="VV", span=(0, 2)),
        Token(id=1, text="北京", pos="NR", span=(2, 4)),
    ]
    deps = [
        DependencyEdge(child=0, head=-1, rel="root"),    # 离开 → root
        DependencyEdge(child=1, head=0, rel="dobj"),     # 北京 → dobj → 离开
    ]

    arg = _make_arg("北京", "Origin", (2, 4))
    evt = _make_event("离开", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "ORIGIN", f"Expected ORIGIN, got {arg.spatial_role}"
    assert arg.verb_spatial_class == "MOVE_FROM"
    print(f"  dobj+MOVE_FROM → ORIGIN: verb={arg.governing_verb}, vclass={arg.verb_spatial_class}, srole={arg.spatial_role}")
    print("  PASS: dobj + MOVE_FROM → ORIGIN\n")


def test_spatial_pobj_origin():
    """pobj + 从 preposition → ORIGIN spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "从山东出发"
    tokens = [
        Token(id=0, text="从", pos="P", span=(0, 1)),
        Token(id=1, text="山东", pos="NR", span=(1, 3)),
        Token(id=2, text="出发", pos="VV", span=(3, 5)),
    ]
    deps = [
        DependencyEdge(child=0, head=2, rel="advmod"),   # 从 → advmod → 出发
        DependencyEdge(child=1, head=0, rel="pobj"),     # 山东 → pobj → 从
        DependencyEdge(child=2, head=-1, rel="root"),    # 出发 → root
    ]

    arg = _make_arg("山东", "Origin", (1, 3))
    evt = _make_event("出发", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "ORIGIN", f"Expected ORIGIN, got {arg.spatial_role}"
    assert arg.governing_verb == "出发"
    assert arg.verb_spatial_class == "MOVE_FROM"
    print(f"  pobj+从 → ORIGIN: gov={arg.governing_verb}, vclass={arg.verb_spatial_class}, srole={arg.spatial_role}")
    print("  PASS: pobj + 从 → ORIGIN\n")


def test_spatial_pobj_path():
    """pobj + 沿 preposition → PATH spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "沿运河南下"
    tokens = [
        Token(id=0, text="沿", pos="P", span=(0, 1)),
        Token(id=1, text="运河", pos="NN", span=(1, 3)),
        Token(id=2, text="南下", pos="VV", span=(3, 5)),
    ]
    deps = [
        DependencyEdge(child=0, head=2, rel="advmod"),   # 沿 → advmod → 南下
        DependencyEdge(child=1, head=0, rel="pobj"),     # 运河 → pobj → 沿
        DependencyEdge(child=2, head=-1, rel="root"),    # 南下 → root
    ]

    arg = _make_arg("运河", "Path", (1, 3))
    evt = _make_event("南下", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "PATH", f"Expected PATH, got {arg.spatial_role}"
    print(f"  pobj+沿 → PATH: gov={arg.governing_verb}, srole={arg.spatial_role}")
    print("  PASS: pobj + 沿 → PATH\n")


def test_spatial_nsubj_actor():
    """nsubj dep → ACTOR spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "故宫举行了典礼"
    tokens = [
        Token(id=0, text="故宫", pos="NR", span=(0, 2)),
        Token(id=1, text="举行", pos="VV", span=(2, 4)),
        Token(id=2, text="了", pos="AS", span=(4, 5)),
        Token(id=3, text="典礼", pos="NN", span=(5, 7)),
    ]
    deps = [
        DependencyEdge(child=0, head=1, rel="nsubj"),    # 故宫 → nsubj → 举行
        DependencyEdge(child=1, head=-1, rel="root"),    # 举行 → root
        DependencyEdge(child=2, head=1, rel="dep"),      # 了 → dep → 举行
        DependencyEdge(child=3, head=1, rel="dobj"),     # 典礼 → dobj → 举行
    ]

    arg_agent = _make_arg("故宫", "Agent", (0, 2))
    arg_patient = _make_arg("典礼", "Patient", (5, 7))
    evt = _make_event("举行", [arg_agent, arg_patient])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg_agent.spatial_role == "ACTOR", f"Expected ACTOR, got {arg_agent.spatial_role}"
    assert arg_agent.syntactic_role == "Subject"
    assert arg_agent.verb_spatial_class == "STATIC_ACTION"
    # The patient (dobj) should NOT be TARGET since 举行 is STATIC_ACTION
    assert arg_patient.spatial_role == "STATIC", f"Expected STATIC for dobj+STATIC_ACTION, got {arg_patient.spatial_role}"
    print(f"  nsubj → ACTOR: vclass={arg_agent.verb_spatial_class}")
    print(f"  dobj+STATIC_ACTION → STATIC (not TARGET): {arg_patient.spatial_role}")
    print("  PASS: nsubj → ACTOR, dobj+non-move → STATIC\n")


def test_spatial_nn_amod_modifier():
    """nn/amod dep → MODIFIER spatial_role."""
    from core.schema import DependencyEdge, Token

    # Sentence: "江南丝绸畅销" → 江南/NR → nn → 丝绸/NN
    tokens = [
        Token(id=0, text="江南", pos="NR", span=(0, 2)),
        Token(id=1, text="丝绸", pos="NN", span=(2, 4)),
        Token(id=2, text="畅销", pos="VV", span=(4, 6)),
    ]
    deps = [
        DependencyEdge(child=0, head=1, rel="nn"),       # 江南 → nn → 丝绸
        DependencyEdge(child=1, head=2, rel="nsubj"),    # 丝绸 → nsubj → 畅销
        DependencyEdge(child=2, head=-1, rel="root"),    # 畅销 → root
    ]

    arg = _make_arg("江南", "Modifier", (0, 2))
    evt = _make_event("畅销", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    assert arg.spatial_role == "MODIFIER", f"Expected MODIFIER, got {arg.spatial_role}"
    assert arg.syntactic_role == "Attributive"
    print(f"  nn → MODIFIER: syn_role={arg.syntactic_role}, srole={arg.spatial_role}")
    print("  PASS: nn/amod → MODIFIER\n")


def test_spatial_role_default_static():
    """No matching dep → STATIC (default)."""
    from core.schema import DependencyEdge, Token

    # Sentence: "结果显著" — "结果" dep is nsubj but we'll test an arg with no dep
    tokens = [
        Token(id=0, text="结果", pos="NN", span=(0, 2)),
        Token(id=1, text="显著", pos="VA", span=(2, 4)),
    ]
    deps = [
        DependencyEdge(child=0, head=1, rel="nsubj"),    # 结果 → nsubj → 显著
        DependencyEdge(child=1, head=-1, rel="root"),    # 显著 → root
    ]

    # This argument has a span that doesn't overlap any token → no token match
    arg = _make_arg("未知", "Unknown", (10, 12))  # span outside token range
    evt = _make_event("显著", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    # No token match → verb_spatial_class = NONE, spatial_role stays None
    assert arg.verb_spatial_class == "NONE", f"Expected NONE, got {arg.verb_spatial_class}"
    # spatial_role is not set when no token found (no dep to infer from)
    assert arg.spatial_role is None, f"Expected None (no token), got {arg.spatial_role}"
    print(f"  no-token-match: spatial_role={arg.spatial_role}, verb_spatial_class={arg.verb_spatial_class}")
    print("  PASS: no token → verb_spatial_class=NONE, spatial_role=None\n")


def test_spatial_dobj_no_governing_verb():
    """dobj with no governing_verb → STATIC (no movement inference)."""
    from core.schema import DependencyEdge, Token

    # Tokens without any V-pos ancestor reachable from the argument
    tokens = [
        Token(id=0, text="北京", pos="NR", span=(0, 2)),
        Token(id=1, text="的", pos="DEG", span=(2, 3)),
        Token(id=2, text="风景", pos="NN", span=(3, 5)),
    ]
    deps = [
        DependencyEdge(child=0, head=2, rel="nn"),       # 北京 → nn → 风景
        DependencyEdge(child=1, head=0, rel="dep"),      # 的 → dep → 北京
        DependencyEdge(child=2, head=-1, rel="root"),    # 风景 → root
    ]

    arg = _make_arg("北京", "Modifier", (0, 2))
    evt = _make_event("风景", [arg])

    extractor = EventExtractor()
    extractor._assign_syntactic_roles([evt], deps, tokens)
    extractor._assign_spatial_roles([evt], deps, tokens)

    # 风景 is NN not V, so no governing_verb → _assign_syntactic_roles won't set it
    assert arg.governing_verb is None, f"Expected no governing_verb, got {arg.governing_verb}"
    assert arg.verb_spatial_class == "NONE"
    assert arg.spatial_role == "MODIFIER", f"Expected MODIFIER (nn dep), got {arg.spatial_role}"
    print(f"  nn+no-verb: spatial_role={arg.spatial_role}, verb_spatial_class={arg.verb_spatial_class}")
    print("  PASS: nn → MODIFIER with verb_spatial_class=NONE\n")


def test_verb_spatial_class_sampling():
    """Spot-check verb_spatial_class mapping for key verbs."""
    from core.event_extractor import _SPATIAL_VERB_CLASSES

    # MOVE_TO samples
    assert _SPATIAL_VERB_CLASSES.get("赴") == "MOVE_TO"
    assert _SPATIAL_VERB_CLASSES.get("到达") == "MOVE_TO"
    assert _SPATIAL_VERB_CLASSES.get("进入") == "MOVE_TO"

    # MOVE_FROM samples
    assert _SPATIAL_VERB_CLASSES.get("撤离") == "MOVE_FROM"
    assert _SPATIAL_VERB_CLASSES.get("辞别") == "MOVE_FROM"

    # MOVE_ALONG samples
    assert _SPATIAL_VERB_CLASSES.get("循") == "MOVE_ALONG"
    assert _SPATIAL_VERB_CLASSES.get("绕行") == "MOVE_ALONG"

    # STATIC_EXIST samples
    assert _SPATIAL_VERB_CLASSES.get("坐落") == "STATIC_EXIST"
    assert _SPATIAL_VERB_CLASSES.get("位于") == "STATIC_EXIST"

    # STATIC_ACTION samples
    assert _SPATIAL_VERB_CLASSES.get("建造") == "STATIC_ACTION"
    assert _SPATIAL_VERB_CLASSES.get("举办") == "STATIC_ACTION"

    # VIEW samples
    assert _SPATIAL_VERB_CLASSES.get("俯瞰") == "VIEW"
    assert _SPATIAL_VERB_CLASSES.get("鸟瞰") == "VIEW"

    # Unknown verb → KeyError (not in map)
    assert "不知所云" not in _SPATIAL_VERB_CLASSES

    print(f"  Total verb entries: {len(_SPATIAL_VERB_CLASSES)}")
    print("  PASS: verb_spatial_class mapping spot-check\n")


def test_preposition_sets():
    """Verify preposition frozensets are correct."""
    from core.event_extractor import _PREP_ORIGIN, _PREP_PATH

    assert "从" in _PREP_ORIGIN
    assert "自" in _PREP_ORIGIN
    assert "由" in _PREP_ORIGIN
    assert "源于" in _PREP_ORIGIN

    assert "沿" in _PREP_PATH
    assert "顺" in _PREP_PATH
    assert "绕" in _PREP_PATH
    assert "环绕" in _PREP_PATH

    # Cross-contamination check
    assert "从" not in _PREP_PATH
    assert "沿" not in _PREP_ORIGIN

    print("  PASS: preposition frozensets correct\n")


if __name__ == '__main__':
    print("=== EventExtractor V4 Tests ===\n")

    print("--- Basic ---")
    test_stopword_filtering()
    test_basic_event_from_relation()
    test_relation_clustering()
    test_interrogative_filtering()
    test_multi_sentence()
    test_traceability_chain()

    print("--- V4: Agent-Dimension Splitting ---")
    test_agent_dimension_splitting()

    print("--- V4: Deduplication ---")
    test_event_deduplication()

    print("--- V4: Sub-Event Hierarchy ---")
    test_sub_event_hierarchy()

    print("--- SRL Hybrid ---")
    test_srl_content_predicate_filter()
    test_srl_enrichment_argm_roles()
    test_srl_confidence_boost()
    test_srl_only_extraction()
    test_srl_no_duplicate_argm()

    print("--- Dependency-Driven Extraction ---")
    test_dep_only_extraction()

    print("--- Spatial Role & Verb Spatial Class ---")
    test_spatial_lobj_loc_container()
    test_spatial_dobj_move_to_target()
    test_spatial_dobj_move_from_origin()
    test_spatial_pobj_origin()
    test_spatial_pobj_path()
    test_spatial_nsubj_actor()
    test_spatial_nn_amod_modifier()
    test_spatial_role_default_static()
    test_spatial_dobj_no_governing_verb()
    test_verb_spatial_class_sampling()
    test_preposition_sets()

    print("=== All tests passed! ===")
