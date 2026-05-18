"""
Query Preprocessor for Pronoun Resolution

Detects queries that need reformulation (pronouns, references) and
uses Claude Haiku for fast, cheap resolution.
"""

import os
import re
import json
import ssl
import urllib.request
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import ConversationContext

# Pronouns and references that indicate context-dependent queries
CONTEXT_INDICATORS = re.compile(
    r'\b(he|him|his|she|her|hers|they|them|their|theirs|'
    r'it|its|that|this|these|those|'
    r'the person|the researcher|the professor|the author|'
    r'same|above|mentioned|discussed)\b',
    re.IGNORECASE
)

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


class QueryPreprocessor:
    """Preprocesses queries to resolve pronouns and context-dependent references."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self):
        """Create SSL context, trying certifi first."""
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    def needs_reformulation(self, query: str) -> bool:
        """Check if query contains pronouns or context references."""
        return bool(CONTEXT_INDICATORS.search(query))

    def reformulate(self, query: str, context: "ConversationContext") -> str:
        """
        Resolve pronouns using Haiku for fast, cheap reformulation.

        Returns the reformulated query, or the original if reformulation fails.
        """
        if not self.api_key:
            print("[PREPROCESSOR] No API key, skipping reformulation")
            return query

        context_str = context.get_context_for_reformulation()
        if not context_str:
            return query

        prompt = REFORMULATION_PROMPT.format(context=context_str, query=query)

        try:
            request_body = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(request_body).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, context=self._ssl_context, timeout=10) as response:
                result = json.loads(response.read().decode())
                reformulated = result["content"][0]["text"].strip()

                if reformulated and reformulated != query:
                    print(f"[PREPROCESSOR] Reformulated: '{query}' -> '{reformulated}'")
                    return reformulated
                return query

        except Exception as e:
            print(f"[PREPROCESSOR] Reformulation failed: {e}")
            return query

    async def reformulate_async(self, query: str, context: "ConversationContext") -> str:
        """
        Async version of reformulate for use with aiohttp.
        Falls back to sync version if aiohttp not available.
        """
        try:
            import aiohttp
        except ImportError:
            return self.reformulate(query, context)

        if not self.api_key:
            return query

        context_str = context.get_context_for_reformulation()
        if not context_str:
            return query

        prompt = REFORMULATION_PROMPT.format(context=context_str, query=query)

        try:
            request_body = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    reformulated = result["content"][0]["text"].strip()

                    if reformulated and reformulated != query:
                        print(f"[PREPROCESSOR] Reformulated: '{query}' -> '{reformulated}'")
                        return reformulated
                    return query

        except Exception as e:
            print(f"[PREPROCESSOR] Async reformulation failed: {e}")
            return query
