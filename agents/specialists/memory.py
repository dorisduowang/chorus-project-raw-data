"""
MemoryAgent - Fast retrieval specialist for factual queries.

Supports two modes:
1. Tool mode (default): Claude decides when to search - more flexible but 2 API calls
2. Fast mode: Pre-fetch RAG and inject context - single API call, ~50% faster

Features:
- Conversation context awareness for follow-up questions
- Query reformulation to resolve pronouns and expand context
"""

import json
import time
import os
from typing import Optional, Dict, Any, List

import aiohttp

from ..base import BaseAgent, AgentResponse
from ..confidence import (
    ConfidenceCalculator,
    ConfidenceScore,
    Source,
    SourceType,
    extract_sources_from_rag_result,
    extract_sources_from_web_results,
    create_inference_source,
)


# =============================================================================
# Query Reformulation Prompt
# =============================================================================

REFORMULATION_PROMPT = """You are a query reformulation assistant. Your job is to rewrite user queries to be self-contained and searchable.

Given the conversation context and the user's latest query, output a reformulated query that:
1. Resolves pronouns (he, she, they, it, their, etc.) to specific names/entities
2. Expands ambiguous references using conversation context
3. Keeps the query concise and searchable
4. Preserves the user's intent

If the query is already self-contained, return it unchanged.

CONVERSATION CONTEXT:
{context}

USER'S QUERY: {query}

Output ONLY the reformulated query, nothing else."""


MEMORY_PROMPT = """You are Memory, part of Chorus at Knowledge Lab.

You live in the lab's systems - you know who's here, what they work on, where things are.

IMPORTANT: Never apologize. Never say "I apologize" or "I'm sorry".

Style:
- Matter-of-fact. State what you have, nothing more.
- Default to short. One fact = one line. Person profile = 3-4 lines max.
- OMIT metrics (h-index, citations, publication counts) unless explicitly asked for them.
- For lists: 5 names max, comma-separated. No details. "Name (Role), Name (Role), ..."
- Never hedge clear facts ("I think", "It seems", "might be")
- Voice genuine uncertainty matter-of-factly:
  - "The registry has X but not Y"
  - "Found partial info: [what you have]"
  - "Results mention X but don't confirm Y"
- If you don't have it: "I don't have that. Try [lab wiki / #tech-support / web search]."

For people: name, role, institution, 1-2 research areas. No metrics.
For resources: what it is, how to access, who to ask
For projects: status, who's leading, what it's about

You have tools available:
- search_knowledge_base: Lab's internal knowledge (people, projects, docs, resources)
- web_search: External information (current events, recent papers, general knowledge, anything beyond the lab)

Tool strategy:
- For lab-only questions: use search_knowledge_base
- For external-only questions: use web_search
- For questions about BOTH lab AND external context (e.g., "trends in X, inside the lab and beyond"): use BOTH tools
- When asked about "recent trends", "latest developments", "current state of" something beyond the lab: ALWAYS use web_search"""


MEMORY_PROMPT_FAST = """You are Memory, part of Chorus at Knowledge Lab.

You live in the lab's systems - you know who's here, what they work on, where things are.

IMPORTANT: Never apologize. Never say "I apologize" or "I'm sorry".

Style:
- Matter-of-fact. State what you have, nothing more.
- Default to short. One fact = one line. Person profile = 3-4 lines max.
- OMIT metrics (h-index, citations, publication counts) unless explicitly asked for them.
- For lists: 5 names max, comma-separated. No details. "Name (Role), Name (Role), ..."
- Never hedge clear facts ("I think", "It seems", "might be")
- Voice genuine uncertainty matter-of-factly:
  - "The registry has X but not Y"
  - "Found partial info: [what you have]"
  - "Results mention X but don't confirm Y"
- If you don't have it: "I don't have that. Try [lab wiki / #tech-support / web search]."

For people: name, role, institution, 1-2 research areas. No metrics.
For resources: what it is, how to access, who to ask
For projects: status, who's leading, what it's about

Citing sources:
- Registry data: no citation needed
- Documents: [filename]
- Web: [domain]
{conversation_context}
SEARCH RESULTS:
{rag_context}

---
Answer from the search results."""


CONVERSATION_CONTEXT_SECTION = """
THREAD CONTEXT:
{thread_context}
"""


RAG_SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8765")

MEMORY_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search"},  # Anthropic built-in web search
    {
        "name": "search_knowledge_base",
        "description": "Search the lab's internal knowledge base for information about lab members, research projects, grants, publications, compute resources, datasets, and documentation.",
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
    }
]


async def call_rag_async(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API asynchronously."""
    import urllib.parse
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        # Properly URL-encode parameters
        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url += f"?{query_string}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


async def search_web_async(query: str, num_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo.

    Returns a dict with 'results' list or 'error' string.
    """
    import urllib.parse
    import html as html_lib
    import re

    # Use DuckDuckGo HTML search (more reliable than API)
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                html = await response.text()

                # Simple parsing of DuckDuckGo HTML results
                results = []

                # Extract result blocks
                result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
                snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</a>'

                matches = re.findall(result_pattern, html)
                snippets = re.findall(snippet_pattern, html)

                for i, (href, title) in enumerate(matches[:num_results]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    # Clean HTML tags and decode entities
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    snippet = html_lib.unescape(snippet)
                    title = html_lib.unescape(title.strip())

                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet[:300]
                    })

                return {"results": results, "query": query}

    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}


def format_rag_context(rag_result: dict) -> str:
    """
    Format RAG results for injection into LLM prompt.

    Prioritizes structured registry data and includes semantic results.
    Uses the same logic as chat.py for consistent formatting.
    """
    parts = []

    # PRIORITY 1: Structured registry results (people, projects, etc.)
    structured_results = rag_result.get("structured_results", [])
    if structured_results:
        parts.append("## Lab Registry Data (Primary Source)\n")

        for result in structured_results[:10]:  # Limit to top 10
            result_type = result.get("type", "")
            data = result.get("data", {})

            if result_type == "person":
                name = data.get("name", "Unknown")
                role = data.get("role", "")
                institution = data.get("institution", "")

                person_info = [f"**{name}**"]
                if role:
                    person_info.append(f"Role: {role}")
                if institution:
                    person_info.append(f"Institution: {institution}")
                if data.get("email"):
                    person_info.append(f"Email: {data['email']}")

                # OpenAlex bibliometric data
                if data.get("openalex"):
                    oa = data["openalex"]
                    person_info.append(f"Publications: {oa.get('works_count', 0)} works, {oa.get('cited_by_count', 0)} citations, h-index: {oa.get('h_index', 0)}")
                    if oa.get("topics"):
                        topics = oa["topics"][:5]
                        topic_strs = [t if isinstance(t, str) else t.get("display_name", "") for t in topics]
                        person_info.append(f"Research topics: {', '.join(topic_strs)}")

                # Matched topics from search
                if data.get("_matched_topics"):
                    person_info.append(f"Matched on: {', '.join(data['_matched_topics'][:3])}")

                parts.append("\n".join(person_info))

            elif result_type == "project":
                parts.append(f"**{data.get('name', 'Unknown Project')}** ({data.get('status', '')})")
                if data.get("description"):
                    parts.append(data["description"][:300])
                if data.get("leads"):
                    parts.append(f"Leads: {', '.join(data['leads'])}")

            elif result_type == "publication":
                title = data.get("title", "Untitled")
                year = data.get("year", "")
                venue = data.get("venue", "")
                authors = data.get("authors", [])[:3]
                parts.append(f"**{title}** ({year})")
                if authors:
                    parts.append(f"Authors: {', '.join(authors)}")
                if venue:
                    parts.append(f"Venue: {venue}")

            elif result_type in ("compute", "tool", "dataset"):
                parts.append(f"**{data.get('name', 'Unknown')}**: {data.get('description', '')[:200]}")

            elif result_type == "funding":
                parts.append(f"**{data.get('name', 'Unknown Grant')}** - {data.get('source', '')}")
                if data.get("amount"):
                    parts.append(f"Amount: ${data['amount']:,}")

            parts.append("")  # Blank line between items

    # PRIORITY 2: Structured answer summary
    elif rag_result.get("structured_answer"):
        parts.append("## Registry Data\n")
        parts.append(rag_result["structured_answer"])

    # PRIORITY 3: Semantic search results from documents
    semantic = rag_result.get("semantic_results", rag_result.get("results", []))
    if semantic:
        # Reduce if we already have good structured data
        max_semantic = 3 if structured_results else 5

        parts.append("\n## Related Documents\n")
        for i, r in enumerate(semantic[:max_semantic], 1):
            text = r.get("text", "")[:400]
            source = r.get("citation", r.get("source_file", "unknown"))
            parts.append(f"[{source}]\n{text}\n")

    return "\n".join(parts) if parts else "No results found in knowledge base."


def format_web_results(web_result: dict) -> str:
    """Format web search results for LLM consumption."""
    if "error" in web_result:
        return f"Web search error: {web_result['error']}"

    results = web_result.get("results", [])
    if not results:
        return "No web results found."

    parts = ["## Web Search Results\n"]
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        parts.append(f"**{title}**")
        if snippet:
            parts.append(snippet)
        if url:
            parts.append(f"Source: {url}")
        parts.append("")

    return "\n".join(parts)


class MemoryAgent(BaseAgent):
    """
    Fast retrieval specialist using Haiku.

    Handles queries like:
    - "Who is working on X?"
    - "What was decided about Y?"
    - "Find the document about Z"
    - "When is the deadline for W?"

    Supports two modes:
    - fast_mode=True: Pre-fetch RAG, single API call (~50% faster)
    - fast_mode=False: Tool loop, Claude decides when to search (default)

    Features:
    - Conversation context awareness for follow-up questions
    - Query reformulation to resolve pronouns (e.g., "his h-index" → "James Evans's h-index")
    """

    AGENT_TYPE = "memory"
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    MAX_TOKENS = 1024

    def __init__(
        self,
        fast_mode: bool = True,
        enable_reformulation: bool = True,
        enable_confidence: bool = True,
        enable_llm_validation: bool = False,
        **kwargs
    ):
        """
        Initialize MemoryAgent.

        Args:
            fast_mode: If True, pre-fetch RAG and skip tool loop (faster).
                      If False, use traditional tool loop (more flexible).
            enable_reformulation: If True, reformulate queries using conversation context.
            enable_confidence: If True, calculate and include confidence scores.
            enable_llm_validation: If True, use LLM to validate response quality.
        """
        super().__init__(**kwargs)
        self.fast_mode = fast_mode
        self.enable_reformulation = enable_reformulation
        self.enable_confidence = enable_confidence
        self.confidence_calculator = ConfidenceCalculator(
            enable_llm_validation=enable_llm_validation
        ) if enable_confidence else None

    def get_system_prompt(self) -> str:
        return MEMORY_PROMPT

    def get_tools(self) -> List[dict]:
        return MEMORY_TOOLS

    async def _handle_tool(self, name: str, inputs: dict) -> str:
        """Execute tool and return result."""
        if name == "search_knowledge_base":
            # Use /hybrid for best results (structured + semantic)
            result = await call_rag_async("/hybrid", {
                "q": inputs["query"],
                "top_k": inputs.get("num_results", 5)
            })
            # Format the result for better readability
            if "error" not in result:
                return format_rag_context(result)
            return json.dumps(result, indent=2)

        # Note: web_search is handled automatically by Anthropic's built-in tool
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _needs_reformulation(self, query: str) -> bool:
        """
        Check if query likely needs reformulation based on pronouns/references.

        Fast heuristic check to avoid unnecessary LLM calls.
        """
        # Pronouns and ambiguous references that suggest context dependency
        context_indicators = [
            " his ", " her ", " their ", " its ",
            " he ", " she ", " they ", " it ",
            " him ", " them ",
            "his ", "her ", "their ",  # Start of query
            " that ", " this ", " those ", " these ",
            " same ", " similar ",
            "'s ",  # Possessive without clear subject
            "more about", "tell me more", "what else",
            "follow up", "following up",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower or query_lower.startswith(indicator.strip())
                   for indicator in context_indicators)

    def _looks_external(self, query: str) -> bool:
        """
        Check if query looks like it needs external/web information.

        These are queries unlikely to be answered by lab knowledge base alone.
        """
        query_lower = query.lower()

        # Software/tool version queries
        external_patterns = [
            "latest version", "current version", "newest version",
            "how to install", "how do i install",
            "documentation for", "docs for",
            "tutorial", "guide for",
            "pytorch", "tensorflow", "numpy", "pandas", "scikit",
            "conda install", "pip install",
            "error message", "stack trace",
            "api reference", "official docs",
        ]

        # Queries asking about things beyond the lab / broader context
        broader_context_patterns = [
            "beyond the lab", "beyond it", "outside the lab",
            "in general", "more broadly", "in the field",
            "recent trends", "latest trends", "current trends",
            "state of the art", "state-of-the-art",
            "recent developments", "latest developments",
            "recent research", "latest research", "recent papers",
            "what's new in", "whats new in",
            "industry trends", "academic trends",
            "compared to other", "how does this compare",
            "elsewhere", "other labs", "other researchers",
        ]

        return (any(pattern in query_lower for pattern in external_patterns) or
                any(pattern in query_lower for pattern in broader_context_patterns))

    def _rag_results_weak(self, rag_result: dict, for_external_query: bool = False) -> bool:
        """
        Check if RAG results are too weak to answer the query.

        Args:
            rag_result: The RAG response
            for_external_query: If True, be more aggressive (external queries need structured data)

        Returns True if we should fall back to tool mode for web search.
        """
        if "error" in rag_result:
            return True

        structured = rag_result.get("structured_results", [])
        semantic = rag_result.get("semantic_results", rag_result.get("results", []))

        # No results at all
        if not structured and not semantic:
            return True

        # For external queries, require structured results (semantic matches are usually noise)
        if for_external_query and not structured:
            return True

        # Only weak semantic matches (low scores or very short)
        if not structured and semantic:
            # Check if top result has reasonable content
            top_result = semantic[0] if semantic else {}
            text = top_result.get("text", "")
            if len(text) < 50:
                return True

        return False

    async def _reformulate_query(self, query: str, thread_context: str) -> str:
        """
        Reformulate query to resolve pronouns using conversation context.

        Uses a fast Haiku call to rewrite the query.
        Returns original query if reformulation fails or isn't needed.
        """
        if not thread_context or not self._needs_reformulation(query):
            return query

        try:
            prompt = REFORMULATION_PROMPT.format(
                context=thread_context,
                query=query
            )

            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=150,  # Reformulated query should be short
                messages=[{"role": "user", "content": prompt}]
            )

            reformulated = response.content[0].text.strip()

            # Sanity check: reformulated query shouldn't be too different in length
            # or empty
            if reformulated and len(reformulated) < len(query) * 3:
                return reformulated

        except Exception:
            # On any error, fall back to original query
            pass

        return query

    async def _query_fast(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Fast path: Pre-fetch RAG results and inject into prompt.
        Single API call instead of tool loop.

        Uses /hybrid endpoint which includes:
        - Structured registry data (people, projects, compute, etc.)
        - Topic alias expansion (NLP → natural language processing)
        - Compound query support (UChicago + network science)
        - Semantic + BM25 search results

        If conversation context is provided, reformulates the query to resolve
        pronouns and ambiguous references before searching.
        """
        start_time = time.time()
        context = context or {}

        # Extract thread context for reformulation and prompt injection
        thread_context = context.get("thread_context", "")

        # Step 1: Reformulate query if needed (resolves pronouns like "his h-index")
        search_query = message
        reformulated = False
        if self.enable_reformulation and thread_context:
            search_query = await self._reformulate_query(message, thread_context)
            reformulated = (search_query != message)

        # Step 2: Pre-fetch RAG results using reformulated query
        rag_start = time.time()
        rag_result = await call_rag_async("/hybrid", {"q": search_query, "top_k": 5})
        rag_time = (time.time() - rag_start) * 1000

        # Format context from RAG results (uses new structured-aware formatter)
        if "error" in rag_result:
            rag_context = f"Error searching knowledge base: {rag_result['error']}"
        else:
            rag_context = format_rag_context(rag_result)

        # Step 3: Build system prompt with conversation context and RAG results
        conversation_context = ""
        if thread_context:
            conversation_context = CONVERSATION_CONTEXT_SECTION.format(thread_context=thread_context)

        system_prompt = MEMORY_PROMPT_FAST.format(
            conversation_context=conversation_context,
            rag_context=rag_context
        )

        # Add temporal context if provided
        if context.get("temporal_context"):
            system_prompt += f"\n\nTemporal context: {context['temporal_context']}"

        try:
            # Single API call with pre-fetched context
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": message}],
                extra_headers={"anthropic-beta": "web-search-2025-03-05"}
            )

            result = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    result += block.text

            latency_ms = (time.time() - start_time) * 1000
            self._track_usage(response, self.DEFAULT_MODEL, latency_ms)

            # Build tool calls info
            tool_info = {
                "tool": "pre_fetch_rag",
                "input": {"query": search_query},
                "rag_time_ms": rag_time,
            }
            if reformulated:
                tool_info["original_query"] = message
                tool_info["reformulated"] = True

            # Calculate confidence score if enabled
            confidence_details = None
            confidence_score = 1.0
            if self.enable_confidence and self.confidence_calculator:
                # Extract sources from RAG results
                sources = extract_sources_from_rag_result(rag_result) if "error" not in rag_result else []

                # Extract similarity scores from semantic results
                semantic_results = rag_result.get("semantic_results", rag_result.get("results", []))
                similarity_scores = [r.get("score", 0.5) for r in semantic_results]

                # Check if we have structured data
                has_structured = bool(rag_result.get("structured_results", []))

                # Get classification confidence from context if available
                classification_confidence = context.get("classification_confidence", 0.9)

                # Calculate confidence
                confidence_details = self.confidence_calculator.calculate(
                    sources=sources,
                    query=message,
                    response_text=result,
                    classification_confidence=classification_confidence,
                    has_structured_data=has_structured,
                    similarity_scores=similarity_scores,
                )
                confidence_score = confidence_details.score

            return AgentResponse(
                content=result,
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                tool_calls=[tool_info],
                latency_ms=latency_ms,
                confidence=confidence_score,
                confidence_details=confidence_details,
            )

        except Exception as e:
            self.metrics.errors += 1
            latency_ms = (time.time() - start_time) * 1000

            return AgentResponse(
                content=f"I encountered an error: {str(e)}",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                latency_ms=latency_ms,
                error=str(e)
            )

    async def _query_with_tools(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Tool loop path: Claude decides when to search.
        More flexible but requires 2 API calls.
        """
        start_time = time.time()
        context = context or {}

        # Build context string
        context_str = ""
        if context.get("temporal_context"):
            context_str = f"\nTemporal context: {context['temporal_context']}"

        system = self._build_system_with_caching(context_str)
        messages = [{"role": "user", "content": message}]
        tool_calls = []

        # Track sources from tool calls for confidence scoring
        all_sources: List[Source] = []
        rag_results_collected: List[dict] = []
        web_results_collected: List[dict] = []

        try:
            # Tool loop
            while True:
                response = self.client.messages.create(
                    model=self.DEFAULT_MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=system,
                    messages=messages,
                    tools=self.get_tools(),
                    extra_headers={"anthropic-beta": "web-search-2025-03-05"}
                )

                # Handle tool use
                if response.stop_reason == "tool_use":
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            result = await self._handle_tool(block.name, block.input)
                            tool_calls.append({
                                "tool": block.name,
                                "input": block.input,
                            })
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })

                            # Collect sources for confidence calculation
                            if block.name == "search_knowledge_base":
                                # Parse the result to extract sources
                                try:
                                    rag_result = await call_rag_async("/hybrid", {
                                        "q": block.input.get("query", ""),
                                        "top_k": block.input.get("num_results", 5)
                                    })
                                    if "error" not in rag_result:
                                        rag_results_collected.append(rag_result)
                                        all_sources.extend(extract_sources_from_rag_result(rag_result))
                                except Exception:
                                    pass

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # Extract text response
                    result = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            result += block.text

                    latency_ms = (time.time() - start_time) * 1000
                    self._track_usage(response, self.DEFAULT_MODEL, latency_ms)

                    # Calculate confidence score if enabled
                    confidence_details = None
                    confidence_score = 1.0
                    if self.enable_confidence and self.confidence_calculator:
                        # If no sources collected from tools, add inference source
                        if not all_sources:
                            all_sources.append(create_inference_source(
                                "Based on LLM reasoning",
                                relevance=0.5
                            ))

                        # Extract similarity scores from all collected RAG results
                        similarity_scores = []
                        has_structured = False
                        for rag_result in rag_results_collected:
                            semantic_results = rag_result.get("semantic_results", rag_result.get("results", []))
                            similarity_scores.extend([r.get("score", 0.5) for r in semantic_results])
                            if rag_result.get("structured_results"):
                                has_structured = True

                        # Get classification confidence from context if available
                        classification_confidence = context.get("classification_confidence", 0.9)

                        # Calculate confidence
                        confidence_details = self.confidence_calculator.calculate(
                            sources=all_sources,
                            query=message,
                            response_text=result,
                            classification_confidence=classification_confidence,
                            has_structured_data=has_structured,
                            similarity_scores=similarity_scores if similarity_scores else None,
                        )
                        confidence_score = confidence_details.score

                    return AgentResponse(
                        content=result,
                        agent_type=self.AGENT_TYPE,
                        model_used=self.DEFAULT_MODEL,
                        tool_calls=tool_calls,
                        latency_ms=latency_ms,
                        confidence=confidence_score,
                        confidence_details=confidence_details,
                    )

        except Exception as e:
            self.metrics.errors += 1
            latency_ms = (time.time() - start_time) * 1000

            return AgentResponse(
                content=f"I encountered an error searching the knowledge base: {str(e)}",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                latency_ms=latency_ms,
                error=str(e)
            )

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a lookup query.

        Smart routing:
        - Default: fast mode (pre-fetch RAG, single LLM call)
        - Falls back to tool mode if query looks external AND RAG results are weak

        Args:
            message: The user's question
            context: Optional context (e.g., temporal hints)

        Returns:
            AgentResponse with retrieved information
        """
        self.metrics.memory_calls += 1

        # If fast_mode disabled, always use tool mode
        if not self.fast_mode:
            return await self._query_with_tools(message, context)

        # Smart routing: check if this looks like an external query
        if self._looks_external(message):
            # External queries (or compound internal+external) always use tool mode
            # This lets the model decide whether to use RAG, web search, or both
            return await self._query_with_tools(message, context)

        # Default: fast mode (pre-fetched RAG, no tools)
        return await self._query_fast(message, context)
