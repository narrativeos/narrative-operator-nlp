"""
gRPC Server Adapter — Internal High-Performance Bus.

Implements the NarrativeService gRPC server defined in
schemas/narrative.proto. Communicates over Unix Domain Socket (UDS)
by default for zero-network-overhead local IPC with the Worker Runtime.

Pre-requisite::

    python -m grpc_tools.protoc \
        -I schemas \
        --python_out=schemas \
        --grpc_python_out=schemas \
        schemas/narrative.proto

Start::

    python adapters/grpc_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root AND schemas/ are on sys.path
# (generated _pb2_grpc.py uses bare 'import narrative_pb2')
_project_root = Path(__file__).resolve().parent.parent
_schemas_dir = _project_root / "schemas"
for p in (str(_project_root), str(_schemas_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import grpc
from concurrent import futures

from core.analyzer import analyze

# ---------------------------------------------------------------------------
# Generated protobuf stubs (must be compiled first)
# ---------------------------------------------------------------------------

try:
    from schemas import narrative_pb2, narrative_pb2_grpc
except ImportError:
    import sys
    print(
        "ERROR: Protobuf stubs not found. Generate them first:\n"
        "  python -m grpc_tools.protoc \\\n"
        "      -I schemas \\\n"
        "      --python_out=schemas \\\n"
        "      --grpc_python_out=schemas \\\n"
        "      schemas/narrative.proto",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Service Implementation
# ---------------------------------------------------------------------------

def _build_span(start: int, end: int):
    """Build a protobuf Span message."""
    return narrative_pb2.Span(start=start, end=end)


def _doc_to_proto(doc) -> "narrative_pb2.AnalyzeResponse":
    """Convert a NarrativeDocument Pydantic model to protobuf AnalyzeResponse."""
    import json

    meta = narrative_pb2.NarrativeMeta(
        source=doc.meta.source,
        version=doc.meta.version,
        timestamp=doc.meta.timestamp,
        text_length=doc.meta.text_length,
    )

    tokens = [
        narrative_pb2.Token(
            id=t.id, text=t.text, pos=t.pos,
            span=_build_span(t.span[0], t.span[1]),
        )
        for t in doc.content.tokens
    ]

    entities = [
        narrative_pb2.Entity(
            id=e.id, text=e.text, category=e.category,
            span=_build_span(e.span[0], e.span[1]),
            normalized=e.normalized, source=e.source, confidence=e.confidence,
            keep=e.keep, filter=e.filter or "", filter_reason=e.filter_reason or "",
        )
        for e in doc.content.entities
    ]

    relations = [
        narrative_pb2.Relation(
            id=r.id, subject=r.subject, subject_ent_id=r.subject_ent_id or "",
            predicate=r.predicate, object=r.object, object_ent_id=r.object_ent_id or "",
            evidence=r.evidence,
            evidence_span=_build_span(r.evidence_span[0], r.evidence_span[1]),
            confidence=r.confidence, source=r.source,
        )
        for r in doc.content.relations
    ]

    events = [
        narrative_pb2.Event(
            id=evt.id,
            event_type=evt.event_type,
            trigger=evt.trigger,
            trigger_span=_build_span(evt.trigger_span[0], evt.trigger_span[1]),
            arguments=[
                narrative_pb2.EventArgument(
                    role=a.role,
                    text=a.text,
                    entity_id=a.entity_id or "",
                    span=_build_span(a.span[0], a.span[1]),
                    syntactic_role=a.syntactic_role or "",
                    governing_verb=a.governing_verb or "",
                    token_span=_build_span(
                        a.token_span[0] if a.token_span else 0,
                        a.token_span[1] if a.token_span else 0,
                    ),
                    spatial_role=a.spatial_role or "",
                    verb_spatial_class=a.verb_spatial_class or "",
                )
                for a in evt.arguments
            ],
            sentence_index=evt.sentence_index,
            is_main_event=evt.is_main_event,
            sub_events=evt.sub_events,
            source_relation_ids=evt.source_relation_ids,
            confidence=evt.confidence,
            source=evt.source,
        )
        for evt in doc.content.events
    ]

    deps = [
        narrative_pb2.DependencyEdge(
            child=d.child, head=d.head, rel=d.rel,
        )
        for d in doc.content.deps
    ]

    noun_signals = [
        narrative_pb2.NounSignal(
            text=ns.text, pos=ns.pos, syntactic_role=ns.syntactic_role,
            score=ns.score, span=_build_span(ns.span[0], ns.span[1]),
            evidence_json=json.dumps(ns.evidence, ensure_ascii=False),
        )
        for ns in doc.content.noun_signals
    ]

    content = narrative_pb2.NarrativeContent(
        tokens=tokens, entities=entities, relations=relations,
        events=events, deps=deps,
        structural_json=json.dumps(doc.content.structural, ensure_ascii=False),
        noun_signals=noun_signals,
    )

    return narrative_pb2.AnalyzeResponse(meta=meta, content=content)


class NarrativeServiceServicer(narrative_pb2_grpc.NarrativeServiceServicer):
    """gRPC service implementation for NarrativeService."""

    def Analyze(self, request, context):
        """Handle Analyze RPC."""
        try:
            import json
            policy = json.loads(request.policy_json) if request.policy_json else None
            noun_signals = (
                json.loads(request.noun_signals_json) if request.noun_signals_json else None
            )
            doc = analyze(
                request.text, source=request.source,
                policy=policy, noun_signals=noun_signals,
            )
            return _doc_to_proto(doc)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return narrative_pb2.AnalyzeResponse()
        except RuntimeError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return narrative_pb2.AnalyzeResponse()


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------

def serve(socket_path: str = "/tmp/narrative-operator-nlp.sock", max_workers: int = 4):
    """
    Start the gRPC server.

    Args:
        socket_path: Unix Domain Socket path.
        max_workers: Thread pool size.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    narrative_pb2_grpc.add_NarrativeServiceServicer_to_server(
        NarrativeServiceServicer(), server
    )

    server.add_insecure_port(f"unix://{socket_path}")
    server.start()

    print(f"gRPC server listening on unix://{socket_path}")
    print("Press Ctrl+C to stop.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down gRPC server...")
        server.stop(grace=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Narrative Operator NLP gRPC Server")
    parser.add_argument(
        "--socket",
        default="/tmp/narrative-operator-nlp.sock",
        help="Unix Domain Socket path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size",
    )
    args = parser.parse_args()

    serve(socket_path=args.socket, max_workers=args.workers)
