"""
gRPC Client Adapter — For Worker Runtime to Call NLP Operator.

Provides a simple client interface for the Worker Runtime (narrative-core)
to call the NLP operator over gRPC + Unix Domain Socket.

Usage::

    from adapters.grpc_client import NlpOperatorClient

    client = NlpOperatorClient()
    response = client.analyze("碳钢是钢的一种。")
    print(response.meta.source)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root AND schemas/ are on sys.path
_project_root = Path(__file__).resolve().parent.parent
_schemas_dir = _project_root / "schemas"
for p in (str(_project_root), str(_schemas_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import grpc

# Generated protobuf stubs
try:
    from schemas import narrative_pb2, narrative_pb2_grpc
except ImportError:
    import sys
    print("ERROR: Protobuf stubs not found. Run: python -m grpc_tools.protoc ...", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# NlpOperatorClient
# ---------------------------------------------------------------------------

class NlpOperatorClient:
    """
    gRPC client for the NLP operator.

    Connects to the gRPC server via Unix Domain Socket by default.
    """

    def __init__(self, socket_path: str = "/tmp/narrative-operator-nlp.sock"):
        self._channel = grpc.insecure_channel(f"unix://{socket_path}")
        self._stub = narrative_pb2_grpc.NarrativeServiceStub(self._channel)

    def analyze(self, text: str, source: str = "hanlp_v2"):
        request = narrative_pb2.AnalyzeRequest(text=text, source=source)
        return self._stub.Analyze(request)

    def close(self):
        """Close the gRPC channel."""
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_via_grpc(text: str, socket_path: str = "/tmp/narrative-operator-nlp.sock"):
    """One-shot convenience: analyze text via gRPC."""
    with NlpOperatorClient(socket_path=socket_path) as client:
        return client.analyze(text)
