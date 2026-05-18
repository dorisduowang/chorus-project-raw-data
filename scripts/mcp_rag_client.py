#!/usr/bin/env python3
"""CHORUS RAG MCP Client

This runs on the host (Mac) and connects to the RAG HTTP server running in Docker.
"""

import json
import urllib.request
import urllib.parse
from fastmcp import FastMCP

RAG_SERVER_URL = "http://localhost:8765"


def call_rag_api(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API."""
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


# Create MCP server
mcp = FastMCP("CHORUS RAG")


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search the CHORUS knowledge base for relevant documents.

    Args:
        query: The search query (natural language)
        top_k: Number of results to return (default 5)

    Returns:
        JSON with search results including text, citations, and scores
    """
    result = call_rag_api("/search", {"q": query, "top_k": top_k})
    return json.dumps(result, indent=2)


@mcp.tool()
def get_context_for_question(question: str, max_sources: int = 5) -> str:
    """
    Get relevant context from the knowledge base to answer a question.

    Use this to retrieve information before answering questions about:
    - Research projects and collaborations
    - Grants and funding (NSF, NIH, etc.)
    - Workshops and presentations
    - Team members and their work

    Args:
        question: The question to find context for
        max_sources: Maximum number of sources to include

    Returns:
        Formatted context with citations
    """
    result = call_rag_api("/context", {"q": question, "max_sources": max_sources})
    return json.dumps(result, indent=2)


@mcp.tool()
def list_available_sources() -> str:
    """
    List all source documents available in the knowledge base.

    Returns:
        JSON list of sources with file names and chunk counts
    """
    result = call_rag_api("/sources")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    # Check if server is reachable
    health = call_rag_api("/health")
    if "error" in health:
        print(f"WARNING: RAG server not reachable at {RAG_SERVER_URL}")
        print(f"Error: {health['error']}")
        print("\nMake sure the Docker container is running:")
        print("  docker-compose up jupyter")
        print("  # Then in the container: python /app/rag_http_server.py")
    else:
        print(f"Connected to RAG server: {health}")

    print("\nStarting MCP server...")
    mcp.run()
