"""
MCP (Model Context Protocol) Adapter — Standard Interface for LLM/Agent.

Exposes the core analyze() function as an MCP Tool so that LLMs,
Agents, Dify, and LangGraph can call the NLP operator through the
standard MCP protocol.

Start::

    python adapters/mcp_server.py

The MCP server communicates over stdio (standard MCP transport).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.analyzer import analyze

# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "analyze_text",
    "description": (
        "Analyze raw Chinese text using HanLP NLP engine and return "
        "Narrative Schema Protocol (NSP) standardized results including "
        "tokens, named entities, and semantic relations."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Raw text to analyze (Chinese).",
            },
            "source": {
                "type": "string",
                "description": "NLP engine identifier.",
                "default": "hanlp_v2",
            },
        },
        "required": ["text"],
    },
}


# ---------------------------------------------------------------------------
# MCP JSON-RPC Handler
# ---------------------------------------------------------------------------

def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """
    Handle a single MCP JSON-RPC request.

    Supports: initialize, tools/list, tools/call.
    Returns None for notifications (no response needed).
    """
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return _response(req_id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "narrative-operator-nlp",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {},
            },
        })

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "tools/list":
        return _response(req_id, {
            "tools": [TOOL_DEFINITION],
        })

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name != "analyze_text":
            return _error(req_id, -32601, f"Unknown tool: {tool_name}")

        return _handle_analyze(req_id, arguments)

    return _error(req_id, -32601, f"Unknown method: {method}")


def _handle_analyze(req_id, arguments: dict) -> dict[str, Any]:
    """Execute the analyze_text tool."""
    text = arguments.get("text", "")
    source = arguments.get("source", "hanlp_v2")

    if not text:
        return _error(req_id, -32602, "Missing required parameter: text")

    try:
        doc = analyze(text, source=source)
        result = doc.model_dump()
        return _response(req_id, {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ],
        })
    except ValueError as exc:
        return _error(req_id, -32000, str(exc))
    except RuntimeError as exc:
        return _error(req_id, -32000, str(exc))
    except Exception as exc:
        return _error(req_id, -32603, f"Internal error: {exc}")


# ---------------------------------------------------------------------------
# JSON-RPC Helpers
# ---------------------------------------------------------------------------

def _response(req_id, result: dict) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


# ---------------------------------------------------------------------------
# Main — stdio Transport
# ---------------------------------------------------------------------------

def main():
    """Run MCP server over stdio (standard MCP transport)."""
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [MCP] %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("mcp_server")

    logger.info("Narrative Operator NLP MCP server started (stdio).")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON: %s", exc)
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
