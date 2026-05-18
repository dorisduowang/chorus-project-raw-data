"""
QueryClassifier - Fast intent classification using hybrid regex + LLM approach.
"""

import re
import json
import time
from typing import Optional, Dict, Any, Tuple

from .base import BaseAgent, QueryClassification, AgentResponse


# =============================================================================
# Classification Patterns (Fast Path)
# =============================================================================

# LOOKUP: Factual retrieval questions
LOOKUP_PATTERNS = [
    r"^what is (the |a )?",
    r"^who (is|was|are|were)\b",
    r"^when (is|was|did|does)\b",
    r"^where (is|was|are|were)\b",
    r"^how many\b",
    r"^which\b",
    r"^list (the |all )?",
    r"^show me (the )?",
    r"^find (the |a )?(chat|conversation|message|document|file|email)",
    r"^what (was|is) (the )?(date|time|name|title|deadline)",
    r"^link to\b",
    r"^get (me )?(the )?",
    r"\b(grants?|funding|budget)\b.*(have|get|receive)",
    r"\bhave\b.*(grants?|funding)",
    r"^what\b.*(grants?|projects?|papers?|publications?)\b",
]

# TECHNICAL: Setup, debugging, access issues
TECHNICAL_PATTERNS = [
    r"\b(access|permission|login|credential|ssh|vpn)\b",
    r"\b(setup|install|configure|config)\b",
    r"\b(error|bug|broken|not working|doesn't work|failed|failing)\b",
    r"\b(debug|troubleshoot|fix|solve|resolve)\b",
    r"\b(midway|cluster|slurm|conda|pip|docker)\b",
    r"\b(how do i|how can i|how to)\b.*(setup|install|access|connect|run)",
    r"\bhelp (me )?(with |to )?(setup|install|configure|debug)",
    r"\b(git|push|pull|commit|merge)\b.*(isn't|not|won't|can't|doesn't)",
    r"\bcan't (connect|access|install|run|push|pull)\b",
    r"\b(won't|doesn't|isn't) work",
]

# CONNECT: Finding relationships
CONNECT_PATTERNS = [
    r"\bwho('s| is| was) working on\b",
    r"\bconnect (me |us )?(with|to)\b",
    r"\brelated to\b",
    r"\bsimilar to\b",
    r"\bcollaborat(e|ion|ing)\b",
    r"\bintroduce (me )?to\b",
    r"\bwho (else |also )?(knows|works|worked)\b",
    r"\bconnection between\b",
    r"\bhow does .* relate to\b",
    r"\banyone (else )?(interested|working)\b",
    r"\bwho (should i|can i) (talk|speak) to\b",
    r"\bwho else\b",
]

# EXPLORE: Deep analysis, synthesis
EXPLORE_PATTERNS = [
    r"\bsynthesize\b",
    r"\bcompare .* (to|with|and)\b",
    r"\brelationship between\b",
    r"\bimplications of\b",
    r"\bwhat are (all |the )?(ways|methods|approaches)\b",
    r"\banalyze\b",
    r"\bcritique\b",
    r"\bevaluate\b",
    r"\bmulti.?perspective\b",
    r"\bdeep dive\b",
    r"\bexplore\b",
    r"\bthink (about|through)\b",
    r"\bwhat (do you think|are your thoughts)\b",
    r"\bhow (might|could|should) we\b",
    r"\bwhat if\b",
    r"\bways (to|we could|of)\b",
    r"\bmeasure\b.*\b(how|ways|methods)\b",
    r"\bhow (can|could|might|would) (we|i|you)\b",
    # Chorus self-awareness patterns (queries about the assistant itself)
    r"\bwhat can you\b",
    r"\bhow (do|can|should) i use you\b",
    r"\bwhat do you know\b",
    r"\bwhat are you\b",
    r"\btell me about yourself\b",
    r"\byour capabilities\b",
    r"\bwhat are your\b",
    r"\bhelp me with\b.*\?",
]


CLASSIFICATION_PROMPT = """You are a query classifier for a research lab assistant. Classify the user's intent into exactly one primary category:

- LOOKUP: Factual retrieval (who, what, when, where, how many, list, find, show me)
- TECHNICAL: Setup, configuration, debugging, access issues, tool problems
- EXPLORE: Deep analysis, synthesis, multi-perspective reasoning, theoretical questions
- CONNECT: Finding relationships between people, ideas, projects, or resources
- AMBIGUOUS: Unclear intent or requires multiple specialists

Consider:
- LOOKUP is for quick facts that can be retrieved from a knowledge base
- TECHNICAL is for practical problems that need step-by-step solutions
- EXPLORE is for questions requiring deep thinking, analysis, or creative insight
- CONNECT is for finding or suggesting relationships and collaborations

Respond with JSON only:
{"intent": "<CATEGORY>", "confidence": <0.0-1.0>, "requires_rag": <true/false>, "secondary_intent": "<CATEGORY or null>", "reasoning": "<one sentence>"}"""


class QueryClassifier(BaseAgent):
    """
    Fast query classification using hybrid regex + LLM approach.

    Uses regex patterns for high-confidence matches (no API call).
    Falls back to Haiku for ambiguous queries.
    """

    AGENT_TYPE = "classifier"
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    MAX_TOKENS = 256

    # Confidence threshold for regex-only classification
    REGEX_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._classification_cache: Dict[str, QueryClassification] = {}

    def get_system_prompt(self) -> str:
        return CLASSIFICATION_PROMPT

    def _classify_with_regex(self, query: str) -> Tuple[Optional[str], float]:
        """
        Attempt to classify using regex patterns.

        Returns:
            (intent, confidence) or (None, 0.0) if no confident match
        """
        query_lower = query.lower().strip()

        # Count matches for each category
        scores = {
            "lookup": 0,
            "technical": 0,
            "connect": 0,
            "explore": 0,
        }

        for pattern in LOOKUP_PATTERNS:
            if re.search(pattern, query_lower):
                scores["lookup"] += 1

        for pattern in TECHNICAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores["technical"] += 1

        for pattern in CONNECT_PATTERNS:
            if re.search(pattern, query_lower):
                scores["connect"] += 1

        for pattern in EXPLORE_PATTERNS:
            if re.search(pattern, query_lower):
                scores["explore"] += 1

        # Find the best match
        total_matches = sum(scores.values())
        if total_matches == 0:
            return None, 0.0

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Calculate confidence based on dominance
        if total_matches == 1:
            confidence = 0.9  # Single clear match
        elif best_score >= 2 and best_score > total_matches * 0.6:
            confidence = 0.85  # Strong dominant match
        elif best_score > total_matches * 0.5:
            confidence = 0.7  # Moderate match
        else:
            confidence = 0.5  # Weak/ambiguous

        # Length-based adjustment for EXPLORE
        word_count = len(query.split())
        if word_count > 30 and best_intent != "explore":
            # Long queries often need exploration
            scores["explore"] += 1
            if scores["explore"] > best_score:
                best_intent = "explore"
                confidence = 0.7

        return best_intent, confidence

    async def _classify_with_llm(self, query: str) -> QueryClassification:
        """Use Haiku to classify ambiguous queries."""
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=self.MAX_TOKENS,
                system=self.get_system_prompt(),
                messages=[{"role": "user", "content": f"Classify this query: {query}"}]
            )

            latency_ms = (time.time() - start_time) * 1000
            self._track_usage(response, self.DEFAULT_MODEL, latency_ms)

            # Parse JSON response
            text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            return QueryClassification(
                intent=data.get("intent", "ambiguous").lower(),
                confidence=float(data.get("confidence", 0.7)),
                secondary_intents=[data["secondary_intent"]] if data.get("secondary_intent") else [],
                requires_rag=data.get("requires_rag", True),
                reasoning=data.get("reasoning", "")
            )

        except Exception as e:
            self.metrics.errors += 1
            # Return ambiguous on error
            return QueryClassification(
                intent="ambiguous",
                confidence=0.5,
                requires_rag=True,
                reasoning=f"Classification error: {str(e)}"
            )

    async def classify(self, query: str) -> QueryClassification:
        """
        Classify a query's intent.

        Uses fast regex matching first, falls back to LLM for ambiguous cases.

        Args:
            query: The user's query

        Returns:
            QueryClassification with intent and metadata
        """
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self._classification_cache:
            self.metrics.cache_hits += 1
            return self._classification_cache[cache_key]

        self.metrics.classifier_calls += 1

        # Try regex classification first
        intent, confidence = self._classify_with_regex(query)

        if intent and confidence >= self.REGEX_CONFIDENCE_THRESHOLD:
            # High confidence regex match - no API call needed
            classification = QueryClassification(
                intent=intent,
                confidence=confidence,
                requires_rag=intent in ("lookup", "connect", "technical"),
                reasoning="regex pattern match"
            )
        else:
            # Need LLM for classification
            classification = await self._classify_with_llm(query)

            # If regex had a weak match, consider it
            if intent and classification.confidence < 0.7:
                # Blend with regex result
                if intent == classification.intent:
                    classification.confidence = min(0.9, classification.confidence + 0.2)
                elif confidence > 0.5:
                    classification.secondary_intents.append(intent)

        # Cache the result
        self._classification_cache[cache_key] = classification

        # Limit cache size
        if len(self._classification_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys = list(self._classification_cache.keys())
            for k in keys[:100]:
                del self._classification_cache[k]

        return classification

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Implement BaseAgent interface - wraps classify() in AgentResponse.
        """
        start_time = time.time()

        classification = await self.classify(message)

        latency_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            content=json.dumps({
                "intent": classification.intent,
                "confidence": classification.confidence,
                "secondary_intents": classification.secondary_intents,
                "requires_rag": classification.requires_rag,
                "reasoning": classification.reasoning,
            }),
            agent_type=self.AGENT_TYPE,
            model_used=self.DEFAULT_MODEL,
            latency_ms=latency_ms,
            confidence=classification.confidence,
        )
