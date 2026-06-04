"""
Example: Call NLP Operator via MCP (Model Context Protocol).

Demonstrates how an LLM/Agent calls the NLP operator through the
standard MCP JSON-RPC interface over stdio.

Prerequisites:
    1. Start the MCP server as a subprocess (this script handles it).
    2. Run this example:
       python examples/call_via_mcp.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def send_request(process: subprocess.Popen, request: dict) -> dict | None:
    """Send a JSON-RPC request and read the response."""
    line = json.dumps(request, ensure_ascii=False) + "\n"
    process.stdin.write(line)
    process.stdin.flush()

    # Read response
    response_line = process.stdout.readline()
    if not response_line:
        return None
    return json.loads(response_line)


def main():
    print("=== Narrative Operator NLP — MCP Call Example ===\n")

    # Start MCP server as subprocess
    server_script = Path(__file__).resolve().parent.parent / "adapters" / "mcp_server.py"

    print(f"Starting MCP server: {server_script}")
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # 1. Initialize
        print("\n1. Sending initialize...")
        init_response = send_request(process, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "example-client", "version": "1.0"},
            },
        })
        if init_response and "result" in init_response:
            print(f"   ✅ Server: {init_response['result']['serverInfo']['name']} "
                  f"v{init_response['result']['serverInfo']['version']}")

        # Send initialized notification
        send_request(process, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        # 2. List tools
        print("\n2. Listing available tools...")
        tools_response = send_request(process, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        if tools_response and "result" in tools_response:
            for tool in tools_response["result"]["tools"]:
                print(f"   ✅ Tool: {tool['name']} — {tool['description'][:60]}...")

        # 3. Call analyze_text tool
        sample_text = "碳钢是钢的一种，具有高强度和高韧性。"
        print(f"\n3. Calling analyze_text with: '{sample_text}'")
        analyze_response = send_request(process, {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_text",
                "arguments": {"text": sample_text},
            },
        })

        if analyze_response and "result" in analyze_response:
            content = analyze_response["result"]["content"]
            if content:
                result_text = content[0]["text"]
                result = json.loads(result_text)
                print(f"   ✅ Success — {len(result['content']['tokens'])} tokens, "
                      f"{len(result['content']['entities'])} entities, "
                      f"{len(result['content']['relations'])} relations")

                if result["content"]["entities"]:
                    print("   Entities:")
                    for ent in result["content"]["entities"]:
                        print(f"     - [{ent['category']}] {ent['text']}")
                if result["content"]["relations"]:
                    print("   Relations:")
                    for rel in result["content"]["relations"]:
                        print(f"     - {rel['subject']} --[{rel['predicate']}]--> "
                              f"{rel['object']}")
        elif analyze_response and "error" in analyze_response:
            print(f"   ❌ Error: {analyze_response['error']['message']}")

        print("\n=== Full NSP Output ===")
        if analyze_response and "result" in analyze_response:
            print(analyze_response["result"]["content"][0]["text"])

    finally:
        print("\nShutting down MCP server...")
        process.stdin.close()
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
