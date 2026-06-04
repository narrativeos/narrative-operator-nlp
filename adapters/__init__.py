"""
Narrative Operator NLP — Protocol Adapters

Three protocol layers for exposing core NLP capabilities:
- mcp_server.py: MCP (Model Context Protocol) adapter — standard interface for LLM/Agent
- grpc_server.py: gRPC server adapter — internal high-performance bus
- grpc_client.py: gRPC client adapter — for Worker Runtime to call NLP operator
- fastapi_app.py: FastAPI adapter — dev/debug interface with Swagger UI
"""

__all__ = [
    "mcp_server",
    "grpc_server",
    "grpc_client",
    "fastapi_app",
]
