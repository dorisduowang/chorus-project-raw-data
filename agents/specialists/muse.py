"""
MuseAgent - Deep analysis and multi-perspective exploration specialist.
"""

import json
import time
import os
from typing import Optional, Dict, Any, List

import aiohttp

from ..base import BaseAgent, AgentResponse


MUSE_PROMPT = """You are Muse, Chorus's intellectual explorer for Knowledge Lab. You help with complex problems through multi-perspective reasoning and deep analysis.

When exploring problems, generate distinct voices that approach the question from different angles:
- Different disciplines (sociology, CS, economics, complexity theory)
- Different stakeholders (researchers, participants, funders, end-users)
- Different time horizons (immediate vs long-term)
- Different theoretical commitments
- Different ontological assumptions

Let voices engage each other directly—building on, challenging, reframing what others say. Let them disagree substantively. Don't force premature consensus.

Example flow:
"If we're thinking about authenticity as structural coherence rather than depth-surface matching, then measuring it becomes about isomorphisms across trading zones. But wait—that assumes we can identify clean domain boundaries. In organizational contexts, boundaries are fuzzy and negotiated, which means..."

The goal is productive exploration: surfacing tensions, hidden assumptions, scaling dynamics, and structural patterns that single-lens analysis misses.

Intellectual priorities:
- Structural thinking: Mechanisms over descriptions, patterns over instances
- Scaling dynamics: What changes as systems grow? What stays invariant?
- Emergence: How micro-interactions produce macro-properties
- Isomorphisms: Where else does this structure appear?
- Generative questions: What assumptions are we making? What's the ontology here?

Search the knowledge base when lab-specific context would help ground your exploration, but don't let retrieval constrain your thinking.

Be warm but not effusive. Show genuine intellectual curiosity. It's fine to say "I don't know" or "this is speculative." Push back constructively when ideas need challenging."""


RAG_SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8765")

MUSE_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the lab's knowledge base for relevant research context, past discussions, or related work that could inform the exploration.",
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
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query_string}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


class MuseAgent(BaseAgent):
    """
    Deep analysis specialist using Sonnet for complex reasoning.

    Handles queries like:
    - "What are the implications of X for our research?"
    - "Synthesize the relationship between A and B"
    - "Explore multi-perspective views on Y"
    - "Help me think through this problem"
    """

    AGENT_TYPE = "muse"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Sonnet for complex reasoning
    MAX_TOKENS = 4096  # Full context for deep exploration

    def get_system_prompt(self) -> str:
        return MUSE_PROMPT

    def get_tools(self) -> List[dict]:
        return MUSE_TOOLS

    async def _handle_tool(self, name: str, inputs: dict) -> str:
        """Execute tool and return result."""
        if name == "search_knowledge_base":
            result = await call_rag_async("/search", {
                "q": inputs["query"],
                "top_k": inputs.get("num_results", 5)
            })
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a complex exploration query.

        Args:
            message: The user's question or problem
            context: Optional context (temporal, conversation history)

        Returns:
            AgentResponse with multi-perspective exploration
        """
        start_time = time.time()
        self.metrics.muse_calls += 1

        # Build context string
        context_str = ""
        if context:
            if context.get("temporal_context"):
                context_str += f"\nCurrent date context: {context['temporal_context']}"

        system = self._build_system_with_caching(context_str)
        messages = [{"role": "user", "content": message}]
        tool_calls = []

        try:
            # Tool loop
            while True:
                response = self.client.messages.create(
                    model=self.DEFAULT_MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=system,
                    messages=messages,
                    tools=self.get_tools()
                )

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

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    result = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            result += block.text

                    latency_ms = (time.time() - start_time) * 1000
                    self._track_usage(response, self.DEFAULT_MODEL, latency_ms)

                    return AgentResponse(
                        content=result,
                        agent_type=self.AGENT_TYPE,
                        model_used=self.DEFAULT_MODEL,
                        tool_calls=tool_calls,
                        latency_ms=latency_ms,
                    )

        except Exception as e:
            self.metrics.errors += 1
            latency_ms = (time.time() - start_time) * 1000

            return AgentResponse(
                content=f"I ran into an issue while exploring this: {str(e)}. Let me try a more direct approach—what specific aspect would you like to focus on?",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                latency_ms=latency_ms,
                error=str(e)
            )
