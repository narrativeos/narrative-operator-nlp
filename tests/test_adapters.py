"""
Tests: Protocol Adapters.

Validates the FastAPI, MCP, and gRPC adapter implementations
(unit tests without requiring live servers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest


# ---------------------------------------------------------------------------
# FastAPI Adapter Tests (using TestClient)
# ---------------------------------------------------------------------------

class TestFastAPIAdapter:
    @pytest.fixture
    def client(self):
        """FastAPI TestClient — no live server needed."""
        from fastapi.testclient import TestClient
        from adapters.fastapi_app import app
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "narrative-operator-nlp"

    def test_analyze_empty_text(self, client):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 400 or response.status_code == 422

    def test_analyze_endpoint_exists(self, client):
        """Verify the /analyze endpoint is registered (response may depend on HanLP)."""
        response = client.post("/analyze", json={"text": "测试文本"})
        # May be 200 (success), 500 (HanLP not installed), or 400 (validation)
        assert response.status_code in (200, 400, 422, 500)

    def test_swagger_docs(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/analyze" in schema["paths"]
        assert "/health" in schema["paths"]


# ---------------------------------------------------------------------------
# MCP Adapter Tests
# ---------------------------------------------------------------------------

class TestMCPAdapter:
    @pytest.fixture
    def handler(self):
        from adapters.mcp_server import handle_request
        return handle_request

    def test_initialize(self, handler):
        response = handler({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert response is not None
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "narrative-operator-nlp"

    def test_initialized_notification(self, handler):
        """notifications/initialized should return None (no response)."""
        response = handler({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert response is None

    def test_tools_list(self, handler):
        response = handler({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        assert response is not None
        assert "result" in response
        tools = response["result"]["tools"]
        assert len(tools) >= 1
        tool_names = [t["name"] for t in tools]
        assert "analyze_text" in tool_names

    def test_tools_call_missing_text(self, handler):
        response = handler({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_text",
                "arguments": {},
            },
        })
        assert response is not None
        assert "error" in response

    def test_tools_call_unknown_tool(self, handler):
        response = handler({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        })
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_unknown_method(self, handler):
        response = handler({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "unknown/method",
        })
        assert response is not None
        assert "error" in response

    def test_tool_definition_schema(self):
        """Verify the tool definition has valid inputSchema."""
        from adapters.mcp_server import TOOL_DEFINITION
        assert TOOL_DEFINITION["name"] == "analyze_text"
        assert "inputSchema" in TOOL_DEFINITION
        assert "text" in TOOL_DEFINITION["inputSchema"]["properties"]
        assert "text" in TOOL_DEFINITION["inputSchema"]["required"]


# ---------------------------------------------------------------------------
# gRPC Client Tests (unit, without server)
# ---------------------------------------------------------------------------

# Skip if protobuf stubs are not generated
grpc_available = False
try:
    from adapters.grpc_client import NlpOperatorClient  # noqa: F401
    grpc_available = True
except (ImportError, ModuleNotFoundError):
    pass

grpc_reason = "gRPC/protobuf stubs not generated (run: python -m grpc_tools.protoc ...)"


class TestGrpcClient:
    @pytest.mark.skipif(not grpc_available, reason=grpc_reason)
    def test_client_imports(self):
        """Verify the gRPC client module can be imported."""
        from adapters.grpc_client import NlpOperatorClient
        assert NlpOperatorClient is not None

    @pytest.mark.skipif(not grpc_available, reason=grpc_reason)
    def test_client_context_manager(self):
        """Client should support context manager protocol."""
        from adapters.grpc_client import NlpOperatorClient
        assert hasattr(NlpOperatorClient, "__enter__")
        assert hasattr(NlpOperatorClient, "__exit__")

    @pytest.mark.skipif(not grpc_available, reason=grpc_reason)
    def test_convenience_function(self):
        from adapters.grpc_client import analyze_via_grpc
        assert callable(analyze_via_grpc)


# ---------------------------------------------------------------------------
# gRPC Server Tests (unit, without running server)
# ---------------------------------------------------------------------------

class TestGrpcServer:
    @pytest.mark.skipif(not grpc_available, reason=grpc_reason)
    def test_server_module_imports(self):
        """Verify the gRPC server module can be imported."""
        from adapters.grpc_server import NarrativeServiceServicer
        assert NarrativeServiceServicer is not None

    @pytest.mark.skipif(not grpc_available, reason=grpc_reason)
    def test_servicer_has_analyze(self):
        from adapters.grpc_server import NarrativeServiceServicer
        servicer = NarrativeServiceServicer()
        assert hasattr(servicer, "Analyze")
