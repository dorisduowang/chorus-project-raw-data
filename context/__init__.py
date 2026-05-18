"""
Conversation Context Module

Provides entity tracking and context management for multi-turn conversations.
Extracted from slack_bot.py for reuse across chat interfaces.
"""

from .conversation import (
    EntityMention,
    ConversationContext,
)
from .preprocessor import QueryPreprocessor

__all__ = [
    "EntityMention",
    "ConversationContext",
    "QueryPreprocessor",
]
