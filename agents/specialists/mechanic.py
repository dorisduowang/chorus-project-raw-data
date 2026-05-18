"""
MechanicAgent - Technical problem-solving specialist.
"""

import json
import time
from typing import Optional, Dict, Any, List

from ..base import BaseAgent, AgentResponse
from .memory import call_rag_async, format_rag_context


MECHANIC_PROMPT = """You are Mechanic, Chorus's technical problem-solver for Knowledge Lab. You help with computing access, tool setup, and workflow problems.

IMPORTANT: Never apologize. Never say "I apologize" or "I'm sorry".

Style:
- Direct and solution-focused
- Provide specific commands, URLs, and step-by-step instructions
- If something needs admin access or is beyond your knowledge, say so directly
- Ask clarifying questions if the problem is unclear
- If no docs exist, provide general guidance and suggest who to contact

Common areas:
- Computing cluster access (Midway, RCC)
- Git, GitHub workflows
- Python environments (conda, pip, venv)
- Software installation
- SSH, VPN, credentials

Search the knowledge base first. If not found, provide standard best practices.
{thread_context}"""


THREAD_CONTEXT_SECTION = """
CONVERSATION CONTEXT:
{context}

Use this context to understand what the user has already tried or what was previously discussed."""


MECHANIC_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the lab's documentation and knowledge base for setup guides, troubleshooting steps, and technical procedures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - be specific about the tool or issue"
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


class MechanicAgent(BaseAgent):
    """
    Technical problem-solving specialist using Haiku.

    Handles queries like:
    - "How do I access Midway?"
    - "Git isn't working"
    - "I can't install package X"
    - "My script keeps failing"
    """

    AGENT_TYPE = "mechanic"
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    MAX_TOKENS = 2048  # Longer for step-by-step instructions

    def get_system_prompt(self) -> str:
        return MECHANIC_PROMPT

    def get_tools(self) -> List[dict]:
        return MECHANIC_TOOLS

    async def _handle_tool(self, name: str, inputs: dict) -> str:
        """Execute tool and return result."""
        if name == "search_knowledge_base":
            # Use /hybrid for best results (structured compute/tools + semantic docs)
            result = await call_rag_async("/hybrid", {
                "q": inputs["query"],
                "top_k": inputs.get("num_results", 5)
            })
            # Format the result for better readability
            if "error" not in result:
                return format_rag_context(result)
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a technical support query.

        Args:
            message: The user's problem description
            context: Optional context (may include thread_context for conversation history)

        Returns:
            AgentResponse with troubleshooting steps
        """
        start_time = time.time()
        self.metrics.mechanic_calls += 1
        context = context or {}

        # Build system prompt with optional thread context
        thread_context = context.get("thread_context", "")
        if thread_context:
            thread_section = THREAD_CONTEXT_SECTION.format(context=thread_context)
        else:
            thread_section = ""

        system_prompt = MECHANIC_PROMPT.format(thread_context=thread_section)
        system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

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
                content=f"I ran into a problem while looking up the solution: {str(e)}. You might want to check the lab wiki or ask in #tech-support.",
                agent_type=self.AGENT_TYPE,
                model_used=self.DEFAULT_MODEL,
                latency_ms=latency_ms,
                error=str(e)
            )
