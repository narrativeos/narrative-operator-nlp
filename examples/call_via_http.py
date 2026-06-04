"""
Example: Call NLP Operator via FastAPI (HTTP).

Demonstrates how an external client (Studio UI, curl, etc.) calls the
NLP operator through the FastAPI dev/debug interface.

Prerequisites:
    1. Start the FastAPI server:
       python adapters/fastapi_app.py
    2. Run this example:
       python examples/call_via_http.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import requests


def call_analyze(text: str, base_url: str = "http://127.0.0.1:8000") -> dict:
    """Call the /analyze endpoint via HTTP POST."""
    response = requests.post(
        f"{base_url}/analyze",
        json={"text": text, "source": "hanlp_v2"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def health_check(base_url: str = "http://127.0.0.1:8000") -> bool:
    """Check if the server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def main():
    base_url = "http://127.0.0.1:8000"

    print("=== Narrative Operator NLP — HTTP Call Example ===\n")

    # 1. Health check
    print("1. Checking server health...")
    if not health_check(base_url):
        print(f"   ❌ Server not reachable at {base_url}")
        print("   Start it with: python adapters/fastapi_app.py")
        sys.exit(1)
    print(f"   ✅ Server is healthy at {base_url}\n")

    # 2. Analyze text
    sample_texts = [
        "碳钢是钢的一种，具有高强度和高韧性。",
        "北京立方庭位于海淀区，是一家科技公司的总部。",
    ]

    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. Analyzing: {text}")
        try:
            result = call_analyze(text, base_url)
            print(f"   ✅ Success — {len(result['content']['tokens'])} tokens, "
                  f"{len(result['content']['entities'])} entities, "
                  f"{len(result['content']['relations'])} relations")
            # Print summary
            if result["content"]["entities"]:
                print("   Entities:")
                for ent in result["content"]["entities"]:
                    print(f"     - [{ent['category']}] {ent['text']}")
            if result["content"]["relations"]:
                print("   Relations:")
                for rel in result["content"]["relations"]:
                    print(f"     - {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
        except requests.RequestException as exc:
            print(f"   ❌ Request failed: {exc}")
        print()

    # 3. Pretty-print full output for first text
    print("=== Full NSP Output (first text) ===")
    result = call_analyze(sample_texts[0], base_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
