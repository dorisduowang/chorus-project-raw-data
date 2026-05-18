"""
Conversation Context with Entity Tracking

Provides enhanced conversation memory with entity tracking and summarization.
Extracted from slack_bot.py for reuse across chat interfaces.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Try to import entity extractor
try:
    from registry.entity_extractor import get_extractor
    ENTITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTOR_AVAILABLE = False


@dataclass
class EntityMention:
    """Tracks an entity mention with recency weighting."""
    entity_id: str
    entity_type: str  # people, datasets, projects, topics, institutions
    display_name: str
    mention_count: int = 1
    last_mentioned: float = field(default_factory=time.time)

    @property
    def recency_score(self) -> float:
        """Score based on recency (higher = more recent)."""
        age_hours = (time.time() - self.last_mentioned) / 3600
        return max(0, 1.0 - (age_hours / 24))  # Decay over 24 hours

    @property
    def relevance_score(self) -> float:
        """Combined score of mention frequency and recency."""
        return (self.mention_count * 0.4) + (self.recency_score * 0.6)


@dataclass
class ConversationContext:
    """Enhanced conversation memory with entity tracking and summarization."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, EntityMention] = field(default_factory=dict)
    summary: str = ""
    summarized_count: int = 0
    last_activity: float = field(default_factory=time.time)

    # Configuration
    MAX_RECENT_MESSAGES: int = 10
    SUMMARIZE_THRESHOLD: int = 15
    MAX_ENTITIES: int = 15

    def add_message(self, role: str, text: str):
        """Add a message and extract entities."""
        self.messages.append({"role": role, "content": text})
        self.last_activity = time.time()
        self._extract_entities(text)
        self._maybe_summarize()

    def add_user_message(self, text: str):
        """Add a user message."""
        self.add_message("user", text)

    def add_assistant_message(self, text: str):
        """Add an assistant response."""
        self.add_message("assistant", text)

    def _extract_entities(self, text: str):
        """Extract and track entities from text."""
        if not ENTITY_EXTRACTOR_AVAILABLE:
            return

        try:
            extractor = get_extractor()
            extracted = extractor.extract(text)

            for entity_type, entity_ids in extracted.to_dict().items():
                for entity_id in entity_ids:
                    if entity_id in self.entities:
                        self.entities[entity_id].mention_count += 1
                        self.entities[entity_id].last_mentioned = time.time()
                    else:
                        display_name = entity_id.replace("-", " ").title()
                        self.entities[entity_id] = EntityMention(
                            entity_id=entity_id,
                            entity_type=entity_type,
                            display_name=display_name
                        )

            # Prune to max entities by relevance
            if len(self.entities) > self.MAX_ENTITIES:
                sorted_entities = sorted(
                    self.entities.items(),
                    key=lambda x: x[1].relevance_score,
                    reverse=True
                )
                self.entities = dict(sorted_entities[:self.MAX_ENTITIES])

        except Exception as e:
            print(f"[CONTEXT] Error extracting entities: {e}")

    def _maybe_summarize(self):
        """Summarize older messages if we've hit the threshold."""
        if len(self.messages) < self.SUMMARIZE_THRESHOLD:
            return

        messages_to_summarize = self.messages[:-self.MAX_RECENT_MESSAGES]
        if not messages_to_summarize:
            return

        summary_points = []
        for msg in messages_to_summarize:
            content = msg["content"]
            first_sentence = content.split(".")[0][:100]
            if msg["role"] == "user":
                summary_points.append(f"User asked: {first_sentence}")
            else:
                if len(content) > 50:
                    summary_points.append(f"Discussed: {first_sentence}")

        if self.summary:
            new_summary = self.summary + " | " + " | ".join(summary_points[-3:])
        else:
            new_summary = " | ".join(summary_points[-5:])

        if len(new_summary) > 500:
            new_summary = new_summary[-500:]

        self.summary = new_summary
        self.summarized_count += len(messages_to_summarize)
        self.messages = self.messages[-self.MAX_RECENT_MESSAGES:]
        print(f"[CONTEXT] Summarized {len(messages_to_summarize)} messages")

    def get_top_entities(self, n: int = 5) -> List[EntityMention]:
        """Get top N entities by relevance score."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )
        return sorted_entities[:n]

    def get_context_for_reformulation(self) -> str:
        """Get structured context for query reformulation."""
        parts = []

        # Key entities
        top_entities = self.get_top_entities(5)
        if top_entities:
            entity_lines = []
            for e in top_entities:
                entity_lines.append(f"- {e.display_name} ({e.entity_type})")
            parts.append("Key entities discussed:\n" + "\n".join(entity_lines))

        # Summary of older messages
        if self.summary:
            parts.append(f"Earlier context: {self.summary}")

        # Recent messages (last 4)
        if self.messages:
            recent = []
            for msg in self.messages[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:200]
                recent.append(f"{role}: {content}")
            parts.append("Recent conversation:\n" + "\n".join(recent))

        return "\n\n".join(parts) if parts else ""

    def get_context_string(self) -> str:
        """Get structured context for the agent (compatibility with slack_bot)."""
        parts = []

        top_entities = self.get_top_entities(5)
        if top_entities:
            entity_lines = []
            for e in top_entities:
                entity_lines.append(f"- {e.display_name} ({e.entity_type})")
            parts.append("**Key entities in this conversation:**\n" + "\n".join(entity_lines))

        if self.summary:
            parts.append(f"**Earlier in conversation ({self.summarized_count} messages summarized):**\n{self.summary}")

        if self.messages:
            recent_parts = ["**Recent messages:**"]
            for msg in self.messages[-6:]:
                role = "User" if msg["role"] == "user" else "Chorus"
                content = msg["content"][:300]
                recent_parts.append(f"{role}: {content}")
            parts.append("\n".join(recent_parts))

        return "\n\n".join(parts) if parts else ""

    def clear(self):
        """Reset the conversation context."""
        self.messages = []
        self.entities = {}
        self.summary = ""
        self.summarized_count = 0
        self.last_activity = time.time()
