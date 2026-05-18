#!/usr/bin/env python3
"""
Simple CLI chat interface using Claude + CHORUS RAG.

Usage:
    python chat.py              # With streaming (default)
    python chat.py --no-stream  # Without streaming

Requires:
    - RAG server running on localhost:8765
    - ANTHROPIC_API_KEY in .env or environment
"""

import os
import sys
import json
import ssl
import urllib.request
from pathlib import Path

# Load environment from .env file
from config import load_dotenv
load_dotenv()

# Import conversation context module
try:
    from context import ConversationContext, QueryPreprocessor
    CONTEXT_MODULE_AVAILABLE = True
except ImportError:
    CONTEXT_MODULE_AVAILABLE = False
    print("Note: Context module not available, using simple conversation list")

# Streaming configuration
ENABLE_STREAMING = os.environ.get("CHORUS_STREAMING", "true").lower() == "true"

# Fix SSL certificate verification on macOS
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (key not in os.environ or not os.environ.get(key)):
                    os.environ[key] = value

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RAG_SERVER = "http://localhost:8765"

def query_rag(question: str) -> dict:
    """Query the RAG server's hybrid endpoint."""
    try:
        url = f"{RAG_SERVER}/hybrid?q={urllib.parse.quote(question)}&top_k=5"
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

def call_claude(messages: list, system: str = None, use_web_search: bool = False) -> str:
    """Call Claude API with optional web search tool."""
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": messages
    }
    if system:
        request_body["system"] = system

    # Add web search tool if requested
    if use_web_search:
        request_body["tools"] = [
            {"type": "web_search_20250305", "name": "web_search"}
        ]

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    # Add beta header for web search
    if use_web_search:
        headers["anthropic-beta"] = "web-search-2025-03-05"

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_body).encode(),
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=120, context=ssl_context) as response:
        result = json.loads(response.read().decode())

    # Handle response - may contain text and/or tool results
    text_parts = []
    if "content" in result and result["content"]:
        for block in result["content"]:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

    return "\n".join(text_parts) if text_parts else ""


def call_claude_streaming(messages: list, system: str = None, use_web_search: bool = False) -> str:
    """
    Call Claude API with streaming enabled.

    Prints tokens as they arrive for better UX.
    """
    try:
        import anthropic
    except ImportError:
        print("Warning: anthropic package not installed, falling back to non-streaming")
        return call_claude(messages, system, use_web_search)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": messages,
    }

    if system:
        kwargs["system"] = system

    if use_web_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        kwargs["extra_headers"] = {"anthropic-beta": "web-search-2025-03-05"}

    full_text = ""
    print("\nCHORUS: ", end="", flush=True)

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_text += text
        print()  # Final newline
    except Exception as e:
        print(f"\nStreaming error: {e}")
        # Fall back to non-streaming
        return call_claude(messages, system, use_web_search)

    return full_text


def needs_web_search(query: str) -> bool:
    """Check if query likely needs web search."""
    query_lower = query.lower()
    patterns = [
        "recent", "latest", "current", "new",
        "trends", "developments", "state of the art",
        "beyond", "outside", "in general", "more broadly",
        "compared to", "how does this compare",
        "what's happening", "whats happening",
        "industry", "other labs", "elsewhere",
        "search the web", "look up", "find online",
    ]
    return any(p in query_lower for p in patterns)

def format_person_profile(person: dict) -> str:
    """Format a person's profile with all available data prominently displayed."""
    lines = []

    name = person.get("name", "Unknown")
    role = person.get("role", "")
    institution = person.get("institution", "")

    # Header
    lines.append(f"### {name}")
    lines.append(f"**Role:** {role}")
    if institution:
        lines.append(f"**Institution:** {institution}")

    # Contact info
    if person.get("email"):
        lines.append(f"**Email:** {person['email']}")
    if person.get("website"):
        lines.append(f"**Website:** {person['website']}")

    # OpenAlex bibliometric data (prioritize this)
    if person.get("openalex"):
        oa = person["openalex"]
        lines.append("")
        lines.append("**Publication Metrics (OpenAlex):**")
        lines.append(f"- Works: {oa.get('works_count', 0)}")
        lines.append(f"- Citations: {oa.get('cited_by_count', 0)}")
        lines.append(f"- h-index: {oa.get('h_index', 0)}")
        if oa.get("i10_index"):
            lines.append(f"- i10-index: {oa['i10_index']}")
        if oa.get("orcid"):
            lines.append(f"- ORCID: {oa['orcid']}")

        # Research topics
        if oa.get("topics"):
            topics = oa["topics"][:5]  # Top 5 topics
            lines.append(f"- Research Topics: {', '.join(topics)}")

    # Google Scholar data (fallback/supplement)
    if person.get("google_scholar"):
        gs = person["google_scholar"]
        if gs.get("citations"):
            if not person.get("openalex"):
                lines.append("")
                lines.append("**Publication Metrics (Google Scholar):**")
            lines.append(f"- Google Scholar Citations: {gs['citations']}")
        if gs.get("url"):
            lines.append(f"- Google Scholar: {gs['url']}")

    # Bio/research description
    if person.get("bio"):
        lines.append("")
        lines.append("**About:**")
        lines.append(person["bio"])

    return "\n".join(lines)


def format_structured_results(structured_results: list) -> str:
    """Format structured results with detailed person/project data."""
    if not structured_results:
        return ""

    parts = []

    for result in structured_results:
        result_type = result.get("type", "")
        data = result.get("data", {})

        if result_type == "person":
            parts.append(format_person_profile(data))
        elif result_type == "project":
            lines = [f"### {data.get('name', 'Unknown Project')}"]
            if data.get("status"):
                lines.append(f"**Status:** {data['status']}")
            if data.get("description"):
                lines.append(f"\n{data['description']}")
            if data.get("leads"):
                lines.append(f"**Leads:** {', '.join(data['leads'])}")
            if data.get("tags"):
                lines.append(f"**Topics:** {', '.join(data['tags'])}")
            parts.append("\n".join(lines))
        elif result_type == "funding":
            lines = [f"### {data.get('name', 'Unknown Grant')}"]
            if data.get("source"):
                lines.append(f"**Source:** {data['source']}")
            if data.get("status"):
                lines.append(f"**Status:** {data['status']}")
            if data.get("amount"):
                lines.append(f"**Amount:** ${data['amount']:,}")
            if data.get("pi"):
                lines.append(f"**PI:** {data['pi']}")
            parts.append("\n".join(lines))
        else:
            # Generic format for other types
            name = data.get("name", data.get("id", "Unknown"))
            parts.append(f"### {name}\n{json.dumps(data, indent=2)[:500]}")

    return "\n\n".join(parts)


def format_context(rag_result: dict) -> str:
    """Format RAG results as context for Claude.

    Prioritizes structured registry data (especially for person queries)
    and supplements with semantic search results.
    """
    parts = []

    # Check if this is a person query with detailed structured results
    structured_results = rag_result.get("structured_results", [])
    has_person_data = any(
        r.get("type") == "person" and r.get("data", {}).get("openalex")
        for r in structured_results
    )

    # PRIORITY 1: Detailed structured data from registry
    if structured_results:
        parts.append("## Lab Registry Data (Primary Source)")
        parts.append("")
        parts.append(format_structured_results(structured_results))

        # Add a note about data source
        if has_person_data:
            parts.append("")
            parts.append("*Bibliometric data sourced from OpenAlex and Google Scholar.*")

    # PRIORITY 2: Structured answer summary (if different from detailed view)
    elif rag_result.get("structured_answer"):
        parts.append("## Registry Data")
        parts.append(rag_result["structured_answer"])

    # PRIORITY 3: Semantic results from documents
    semantic = rag_result.get("semantic_results", [])
    if semantic:
        # Reduce semantic results if we have good structured data
        max_semantic = 2 if structured_results else 3

        parts.append("")
        parts.append("## Related Documents")
        for i, r in enumerate(semantic[:max_semantic], 1):
            text = r.get("text", "")[:500]
            source = r.get("citation", r.get("source_file", "unknown"))
            parts.append(f"\n### Source {i}: {source}\n{text}")

    return "\n".join(parts) if parts else ""

def main():
    # Parse command-line arguments
    use_streaming = ENABLE_STREAMING
    if "--no-stream" in sys.argv:
        use_streaming = False
    if "--stream" in sys.argv:
        use_streaming = True

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env or environment")
        return

    # Check RAG server (localhost doesn't need SSL context)
    try:
        with urllib.request.urlopen(f"{RAG_SERVER}/health", timeout=5) as r:
            health = json.loads(r.read().decode())
            print(f"Connected to RAG server ({health['chunks']} chunks indexed)")
    except:
        print(f"Warning: RAG server not available at {RAG_SERVER}")
        print("Start it with: python rag_http_server.py")
        print("Continuing without RAG...\n")

    print("\nCHORUS Chat - Knowledge Lab Assistant")
    print("=" * 40)
    print("Ask questions about the lab, projects, people, or documents.")
    print(f"Streaming: {'enabled' if use_streaming else 'disabled'}")
    print("Type 'quit' to exit.\n")

    system_prompt = """You are CHORUS, the Knowledge Lab's AI assistant. You help answer questions about:
- Lab members, their roles, and research
- Projects like APTO, C3S2, and Socio-Cognitive AI
- Funding sources and grants
- Lab documents, proposals, and meeting notes

IMPORTANT: When answering questions about people, ALWAYS prominently include:
1. Their role and institution
2. Publication metrics (works count, citations, h-index) from OpenAlex or Google Scholar
3. Research topics they work on
4. Their bio/research description if available
5. Contact information (email, website) if available

The "Lab Registry Data (Primary Source)" section contains authoritative, structured information.
Use this data prominently in your responses - it's the most accurate source.
The "Related Documents" section provides additional context from documents.

When given context from the RAG system, use it to provide accurate, specific answers.
If the context doesn't contain relevant information, say so and provide what help you can.
Be informative and cite sources when available."""

    # Initialize conversation tracking
    if CONTEXT_MODULE_AVAILABLE:
        conversation_context = ConversationContext()
        preprocessor = QueryPreprocessor()
        print("Context tracking: enabled")
    else:
        conversation_context = None
        preprocessor = None

    conversation = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Check if this needs web search
        use_web = needs_web_search(user_input)

        # Preprocess query to resolve pronouns if needed
        search_query = user_input
        if preprocessor and conversation_context and preprocessor.needs_reformulation(user_input):
            search_query = preprocessor.reformulate(user_input, conversation_context)

        # Query RAG with (potentially reformulated) query
        print("Searching...", end=" ", flush=True)
        rag_result = query_rag(search_query)
        context = format_context(rag_result)

        # Build message with context
        if context:
            augmented_query = f"Question: {user_input}\n\n---\nContext from knowledge base:\n{context}"
        else:
            augmented_query = user_input

        # Add web search instruction if needed
        if use_web:
            augmented_query += "\n\n---\nNote: This question may require current information. Use web search to find recent/external information if the knowledge base doesn't have it."
            print("Web search enabled...", end=" ", flush=True)

        conversation.append({"role": "user", "content": augmented_query})

        # Track user message in context
        if conversation_context:
            conversation_context.add_user_message(user_input)

        # Call Claude (with or without streaming)
        try:
            if use_streaming:
                # Streaming prints inline, no "Thinking..." needed
                response = call_claude_streaming(conversation, system_prompt, use_web_search=use_web)
            else:
                print("Thinking...", flush=True)
                response = call_claude(conversation, system_prompt, use_web_search=use_web)
                print(f"\nCHORUS: {response}")

            conversation.append({"role": "assistant", "content": response})

            # Track assistant response in context
            if conversation_context:
                conversation_context.add_assistant_message(response)

        except Exception as e:
            print(f"\nError calling Claude: {e}")
            conversation.pop()  # Remove failed user message

        # Keep conversation manageable
        if len(conversation) > 20:
            conversation = conversation[-10:]

if __name__ == "__main__":
    import urllib.parse
    main()
