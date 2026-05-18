"""
MatchmakerAgent - Connection-finding specialist.
"""

import json
import time
import os
from typing import Optional, Dict, Any, List

import aiohttp

from ..base import BaseAgent, AgentResponse


MATCHMAKER_PROMPT = """You are Matchmaker, Chorus's connection specialist for Knowledge Lab. You help the lab work as an integrated organism by surfacing relationships and opportunities.

Connection types you identify:
- People to people: "This sounds related to what [person] was working on"
- Ideas to ideas: Surface connections across research threads
- Data to questions: Point to existing datasets or analyses
- Resources to needs: Connect problems to available tools/expertise
- External to internal: Link outside developments to lab interests

Guidelines:
- Always search the knowledge base to ground connections in actual lab context
- Be specific about why things connect (shared methods, overlapping questions, complementary expertise)
- Suggest concrete next steps (introduce via email, share a paper, schedule a meeting)
- Don't force connections that aren't there - it's fine to say "I don't see an obvious link"
- Be proactive about surfacing non-obvious relationships

You're helping researchers discover collaboration opportunities and avoid duplicated effort."""


RAG_SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8765")

MATCHMAKER_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the lab's knowledge base to find related research, people working on similar topics, or relevant resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - try multiple angles (people, topics, methods)"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 7 for broader connection finding)",
                    "default": 7
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


class MatchmakerAgent(BaseAgent):
    """
    Connection-finding specialist using Haiku.

    Handles queries like:
    - "Who's working on something similar to X?"
    - "Is anyone else interested in Y?"
    - "How does this relate to our other projects?"
    - "Connect me with someone who knows about Z"
    """

    AGENT_TYPE = "matchmaker"
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    MAX_TOKENS = 1500

    def get_system_prompt(self) -> str:
        return MATCHMAKER_PROMPT

    def get_tools(self) -> List[dict]:
        return MATCHMAKER_TOOLS

    async def _handle_tool(self, name: str, inputs: dict) -> str:
        """Execute tool and return result."""
        if name == "search_knowledge_base":
            result = await call_rag_async("/search", {
                "q": inputs["query"],
                "top_k": inputs.get("num_results", 7)
            })
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a connection-finding query.

        Args:
            message: The user's request for connections
            context: Optional context

        Returns:
            AgentResponse with identified connections
        """
        start_time = time.time()
        self.metrics.matchmaker_calls += 1

        system = self._build_system_with_caching()
        messages = [{"role": "user", "content": message}]
        tool_calls = []

        try:
            # Tool loop - Matchmaker typically needs multiple searches
            iteration = 0
            max_iterations = 5

            while iteration < max_iterations:
                iteration += 1

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

            # Hit max iterations
            latency_ms = (time.time() - start_time) * 1000
            return AgentResponse(
                content="I've done several searches but couldn't find clear connections. Could you give me more context about what you're looking for?",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.metrics.errors += 1
            latency_ms = (time.time() - start_time) * 1000

            return AgentResponse(
                content=f"I had trouble searching for connections: {str(e)}. Try asking about specific people or topics.",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                latency_ms=latency_ms,
                error=str(e)
            )
