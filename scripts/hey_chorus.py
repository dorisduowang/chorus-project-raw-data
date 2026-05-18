#!/usr/bin/env python3
"""
Command-line chat with Claude using RAG.

Usage:
    python hey_chorus.py "What research projects are happening?"
    python hey_chorus.py  # Interactive mode

Requires:
    - ANTHROPIC_API_KEY in .env file or environment
    - RAG HTTP server running on localhost:8765
"""

import json
import os
import sys
import urllib.request
import urllib.parse

# Load environment from .env file
from config import load_dotenv
load_dotenv()

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)


RAG_SERVER_URL = "http://localhost:8765"


def call_rag(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API."""
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


# Tool definitions for Claude
TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the CHORUS knowledge base for relevant documents about research, grants, workshops, and team members at Knowledge Lab.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_context_for_question",
        "description": "Get relevant context from the knowledge base to help answer a question. Use this before answering questions about research, grants, workshops, or team members.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to find context for"
                }
            },
            "required": ["question"]
        }
    }
]


def handle_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return result."""
    if name == "search_knowledge_base":
        result = call_rag("/search", {
            "q": inputs["query"],
            "top_k": inputs.get("num_results", 5)
        })
        return json.dumps(result, indent=2)

    elif name == "get_context_for_question":
        result = call_rag("/context", {"q": inputs["question"]})
        return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


def chat(user_message: str, client: anthropic.Anthropic) -> str:
    """Send a message to Claude with RAG tools."""

    system = """You are a helpful assistant with access to the CHORUS knowledge base,
which contains documents about research projects, grants, workshops, and team members
at the Knowledge Lab (University of Chicago).

When answering questions:
1. Use get_context_for_question to retrieve relevant information first
2. Base your answers on the retrieved context
3. Cite your sources
4. If the context doesn't have enough info, say so

Always search the knowledge base before answering questions about the lab."""

    messages = [{"role": "user", "content": user_message}]

    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        tools=TOOLS,
        messages=messages
    )

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Using tool: {block.name}]")
                result = handle_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # Continue with tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages
        )

    # Extract text response
    result = ""
    for block in response.content:
        if hasattr(block, 'text'):
            result += block.text

    return result


def main():
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it with: export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    # Check RAG server
    health = call_rag("/health")
    if "error" in health:
        print(f"Error: RAG server not reachable at {RAG_SERVER_URL}")
        print(f"  {health['error']}")
        print("\nStart the server with:")
        print("  docker exec chorus-jupyter-1 python /app/rag_http_server.py")
        sys.exit(1)

    print(f"Connected to RAG server ({health.get('chunks', 0)} chunks indexed)")

    client = anthropic.Anthropic(api_key=api_key)

    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuestion: {query}\n")
        response = chat(query, client)
        print(response)
        return

    # Interactive mode
    print("\nCHORUS RAG Chat (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print()
            response = chat(user_input, client)
            print(f"Claude: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
