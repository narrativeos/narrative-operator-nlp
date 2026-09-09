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
            "policy": {
                "type": "object",
                "description": "Optional entity quality policy (F1 shape + F2 confidence/evidence). "
                               "When omitted, F1/F2 are skipped (backward compatible).",
            },
            "noun_signals": {
                "type": "object",
                "description": "Optional noun-signal extraction config (Step A POS gating + "
                               "Step B syntactic role). When omitted or enabled=false, no noun signals.",
            },
        },
        "required": ["text"],
    },
}

SUMMARIZE_TOOL_DEFINITION = {
    "name": "summarize_text",
    "description": (
        "Generate an extractive summary of raw text using TextRank + position prior. "
        "No full NLP analysis required. Supports 'chars' (fixed length) or "
        "'ratio' (percentage of text) budget modes."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Raw text to summarize.",
            },
            "mode": {
                "type": "string",
                "description": "Budget mode: 'chars' (fixed length) or 'ratio' (percentage).",
                "default": "chars",
            },
            "target_chars": {
                "type": "integer",
                "description": "Target character count (used when mode='chars').",
                "default": 30,
            },
            "target_ratio": {
                "type": "number",
                "description": "Target ratio of original text (used when mode='ratio').",
                "default": 0.2,
            },
            "language": {
                "type": "string",
                "description": "Language: auto, modern, classical, english.",
                "default": "auto",
            },
        },
        "required": ["text"],
    },
}

TITLE_TOOL_DEFINITION = {
    "name": "generate_title",
    "description": (
        "Generate a concise title from raw text using TextRank + position prior. "
        "No full NLP analysis required. Supports 'chars' (fixed length) or "
        "'ratio' (percentage of text) budget modes. For entity-composition titles, "
        "use analyze_text with generate_title=True."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Raw text to generate title for.",
            },
            "mode": {
                "type": "string",
                "description": "Budget mode: 'chars' (fixed length) or 'ratio' (percentage).",
                "default": "chars",
            },
            "target_chars": {
                "type": "integer",
                "description": "Target character count (used when mode='chars').",
                "default": 20,
            },
            "target_ratio": {
                "type": "number",
                "description": "Target ratio of original text (used when mode='ratio').",
                "default": 0.05,
            },
            "language": {
                "type": "string",
                "description": "Language: auto, modern, classical, english.",
                "default": "auto",
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
            "tools": [TOOL_DEFINITION, SUMMARIZE_TOOL_DEFINITION, TITLE_TOOL_DEFINITION],
        })

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "analyze_text":
            return _handle_analyze(req_id, arguments)
        if tool_name == "summarize_text":
            return _handle_summarize(req_id, arguments)
        if tool_name == "generate_title":
            return _handle_title(req_id, arguments)

        return _error(req_id, -32601, f"Unknown tool: {tool_name}")

    return _error(req_id, -32601, f"Unknown method: {method}")


def _handle_analyze(req_id, arguments: dict) -> dict[str, Any]:
    """Execute the analyze_text tool."""
    text = arguments.get("text", "")
    source = arguments.get("source", "hanlp_v2")
    policy = arguments.get("policy")
    noun_signals = arguments.get("noun_signals")

    if not text:
        return _error(req_id, -32602, "Missing required parameter: text")

    try:
        doc = analyze(text, source=source, policy=policy, noun_signals=noun_signals)
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


def _handle_summarize(req_id, arguments: dict) -> dict[str, Any]:
    """Execute the summarize_text tool."""
    text = arguments.get("text", "")
    mode = arguments.get("mode", "chars")
    target_chars = arguments.get("target_chars", 30)
    target_ratio = arguments.get("target_ratio", 0.2)
    language = arguments.get("language", "auto")

    if not text:
        return _error(req_id, -32602, "Missing required parameter: text")

    try:
        from core.summarizer import summarize_text
        summary = summarize_text(
            text, mode=mode, target_chars=target_chars,
            target_ratio=target_ratio, language=language,
        )
        result = summary.model_dump()
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
    except Exception as exc:
        return _error(req_id, -32603, f"Internal error: {exc}")


def _handle_title(req_id, arguments: dict) -> dict[str, Any]:
    """Execute the generate_title tool."""
    text = arguments.get("text", "")
    mode = arguments.get("mode", "chars")
    target_chars = arguments.get("target_chars", 14)
    target_ratio = arguments.get("target_ratio", 0.05)
    language = arguments.get("language", "auto")

    if not text:
        return _error(req_id, -32602, "Missing required parameter: text")

    try:
        from core.titler import generate_title_text
        title = generate_title_text(
            text, mode=mode, target_chars=target_chars,
            target_ratio=target_ratio, language=language,
        )
        result = title.model_dump()
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
