#!/usr/bin/env python3
"""
Slack bot for CHORUS.

Responds to @mentions and DMs with answers from the Knowledge Lab knowledge base.
Uses the multi-agent ChorusOrchestrator for intelligent query routing.

Requires environment variables:
    - SLACK_BOT_TOKEN: Bot User OAuth Token (xoxb-...)
    - SLACK_APP_TOKEN: App-Level Token for Socket Mode (xapp-...)
    - ANTHROPIC_API_KEY: Anthropic API key

Setup:
    1. Create Slack app at https://api.slack.com/apps
    2. Enable Socket Mode
    3. Add bot scopes: app_mentions:read, chat:write, im:history
    4. Subscribe to app_mention and message.im events
    5. Install to workspace
"""

import asyncio
import os
import sys
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime

# Load .env file before anything else
from config import load_dotenv

load_dotenv()

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
except ImportError:
    print("Please install slack-bolt: pip install slack-bolt")
    sys.exit(1)

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)

# Import the multi-agent orchestrator (falls back to legacy ChorusAgent)
try:
    from agents import ChorusOrchestrator
    USE_ORCHESTRATOR = True
except ImportError:
    from agents.legacy import ChorusAgent
    USE_ORCHESTRATOR = False
    print("Warning: ChorusOrchestrator not available, using legacy ChorusAgent")

# Import async RAG helper - try memory agent first (has better formatting), fallback to legacy
try:
    from agents.specialists.memory import call_rag_async
except ImportError:
    from agents.legacy import call_rag_async

# Import entity extractor for conversation memory
try:
    from registry.entity_extractor import get_extractor, ExtractedEntities
    ENTITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTOR_AVAILABLE = False
    print("Warning: EntityExtractor not available - entity tracking disabled")


# =============================================================================
# Configuration
# =============================================================================

# RAG server URL - use localhost for local development
RAG_SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8765")

# Initialize Slack app
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize the agent (will be set up in main)
# Can be either ChorusOrchestrator (new) or ChorusAgent (legacy)
agent = None


# =============================================================================
# Thread Memory - Enhanced conversation history per thread
# =============================================================================

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
class ThreadConversation:
    """Enhanced conversation memory with entity tracking and summarization."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, EntityMention] = field(default_factory=dict)  # entity_id -> EntityMention
    summary: str = ""  # Summary of older messages
    summarized_count: int = 0  # Number of messages that have been summarized
    last_activity: float = field(default_factory=time.time)
    bot_active: bool = True

    # Configuration
    MAX_RECENT_MESSAGES: int = 10  # Keep this many recent messages in full
    SUMMARIZE_THRESHOLD: int = 15  # Summarize when we reach this many messages
    MAX_ENTITIES: int = 15  # Max entities to track

    def add_user_message(self, text: str):
        """Add a user message and extract entities."""
        self.messages.append({"role": "user", "content": text})
        self.last_activity = time.time()
        self._extract_entities(text)
        self._maybe_summarize()

    def add_assistant_message(self, text: str):
        """Add an assistant response and extract entities."""
        self.messages.append({"role": "assistant", "content": text})
        self.last_activity = time.time()
        self._extract_entities(text)
        self._maybe_summarize()

    def _extract_entities(self, text: str):
        """Extract and track entities from text."""
        if not ENTITY_EXTRACTOR_AVAILABLE:
            return

        try:
            extractor = get_extractor()
            extracted = extractor.extract(text)

            # Process each entity type
            for entity_type, entity_ids in extracted.to_dict().items():
                for entity_id in entity_ids:
                    if entity_id in self.entities:
                        # Update existing mention
                        self.entities[entity_id].mention_count += 1
                        self.entities[entity_id].last_mentioned = time.time()
                    else:
                        # New entity
                        # Get display name from entity_id (convert hyphens to spaces, title case)
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
            print(f"[ENTITY] Error extracting entities: {e}")

    def _maybe_summarize(self):
        """Summarize older messages if we've hit the threshold."""
        if len(self.messages) < self.SUMMARIZE_THRESHOLD:
            return

        # Get messages to summarize (older ones beyond the recent window)
        messages_to_summarize = self.messages[:-self.MAX_RECENT_MESSAGES]
        if not messages_to_summarize:
            return

        # Create a simple extractive summary (key points)
        summary_points = []
        for msg in messages_to_summarize:
            content = msg["content"]
            # Extract first sentence or up to 100 chars
            first_sentence = content.split(".")[0][:100]
            if msg["role"] == "user":
                summary_points.append(f"User asked: {first_sentence}")
            else:
                # For assistant, extract key info
                if len(content) > 50:
                    summary_points.append(f"Discussed: {first_sentence}")

        # Combine with existing summary
        if self.summary:
            new_summary = self.summary + " | " + " | ".join(summary_points[-3:])
        else:
            new_summary = " | ".join(summary_points[-5:])

        # Keep summary reasonable length
        if len(new_summary) > 500:
            new_summary = new_summary[-500:]

        self.summary = new_summary
        self.summarized_count += len(messages_to_summarize)

        # Keep only recent messages
        self.messages = self.messages[-self.MAX_RECENT_MESSAGES:]
        print(f"[MEMORY] Summarized {len(messages_to_summarize)} messages, "
              f"keeping {len(self.messages)} recent")

    def get_top_entities(self, n: int = 5) -> List[EntityMention]:
        """Get top N entities by relevance score."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )
        return sorted_entities[:n]

    def get_context_string(self) -> str:
        """Get structured context for the agent."""
        parts = []

        # Section 1: Tracked entities (most important context)
        top_entities = self.get_top_entities(5)
        if top_entities:
            entity_lines = []
            for e in top_entities:
                entity_lines.append(f"- {e.display_name} ({e.entity_type})")
            parts.append("**Key entities in this conversation:**\n" + "\n".join(entity_lines))

        # Section 2: Summary of older messages
        if self.summary:
            parts.append(f"**Earlier in conversation ({self.summarized_count} messages summarized):**\n{self.summary}")

        # Section 3: Recent messages
        if self.messages:
            recent_parts = ["**Recent messages:**"]
            for msg in self.messages[-6:]:  # Last 6 messages
                role = "User" if msg["role"] == "user" else "Chorus"
                content = msg["content"][:300]  # Truncate for context
                recent_parts.append(f"{role}: {content}")
            parts.append("\n".join(recent_parts))

        if not parts:
            return ""

        return "\n\n".join(parts)


class ThreadMemoryManager:
    """Manages conversation memory across all threads."""

    def __init__(self, ttl_hours: int = 24):
        self.threads: Dict[str, ThreadConversation] = {}
        self.ttl_seconds = ttl_hours * 3600

    def get_thread_key(self, channel: str, thread_ts: Optional[str]) -> str:
        """Generate a unique key for a thread."""
        # If no thread_ts, use the message ts as the thread key (for top-level messages)
        return f"{channel}:{thread_ts or 'dm'}"

    def get_or_create(self, channel: str, thread_ts: Optional[str]) -> ThreadConversation:
        """Get existing thread conversation or create a new one."""
        key = self.get_thread_key(channel, thread_ts)

        # Clean up old threads periodically
        self._cleanup_old_threads()

        if key not in self.threads:
            self.threads[key] = ThreadConversation()

        return self.threads[key]

    def is_bot_active_in_thread(self, channel: str, thread_ts: Optional[str]) -> bool:
        """Check if the bot should respond in this thread."""
        key = self.get_thread_key(channel, thread_ts)
        if key in self.threads:
            thread = self.threads[key]
            # Bot is active if it has participated and thread isn't too old
            if time.time() - thread.last_activity < self.ttl_seconds:
                return thread.bot_active
        return False

    def activate_thread(self, channel: str, thread_ts: Optional[str]):
        """Mark that the bot should respond to all messages in this thread."""
        thread = self.get_or_create(channel, thread_ts)
        thread.bot_active = True

    def _cleanup_old_threads(self):
        """Remove threads that haven't been active for a while."""
        current_time = time.time()
        keys_to_remove = [
            key for key, thread in self.threads.items()
            if current_time - thread.last_activity > self.ttl_seconds
        ]
        for key in keys_to_remove:
            del self.threads[key]


# Global thread memory manager
thread_memory = ThreadMemoryManager()


# =============================================================================
# Slash Commands
# =============================================================================

@app.command("/chorus-status")
async def handle_status_command(ack, respond):
    """Handle /chorus-status command - show system status and metrics."""
    await ack()

    try:
        # Get RAG server health
        health = await call_rag_async("/health")

        if "error" in health:
            status_text = f":red_circle: *RAG Server*: Offline ({health['error']})"
        else:
            status_text = f":large_green_circle: *RAG Server*: Online\n" \
                         f"• Chunks indexed: {health.get('chunks', 0):,}\n" \
                         f"• Reranking: {'enabled' if health.get('reranking_enabled') else 'disabled'}\n" \
                         f"• LLM reformulation: {'enabled' if health.get('llm_reformulation_enabled') else 'disabled'}"

        # Get agent metrics and config if available
        if agent and hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            orch = metrics.get('orchestrator', {})
            agent_config = metrics.get('agent_config', {})

            # Show enabled/disabled agents
            enabled = agent_config.get('enabled', [])
            disabled = agent_config.get('disabled', [])

            status_text += f"\n\n*Agent Configuration*\n" \
                          f"• Enabled: {', '.join(enabled) if enabled else 'none'}\n" \
                          f"• Disabled: {', '.join(disabled) if disabled else 'none'}"

            status_text += f"\n\n*Agent Metrics*\n" \
                          f"• Total queries: {orch.get('queries', 0)}\n" \
                          f"• Cache hit rate: {orch.get('cache_hit_rate', 0):.1%}\n" \
                          f"• Avg latency: {orch.get('avg_latency_ms', 0):.0f}ms\n" \
                          f"• Haiku/Sonnet ratio: {orch.get('haiku_rate', 0):.1%}"

        # Add thread memory stats (enhanced with entity tracking)
        active_threads = len(thread_memory.threads)
        total_messages = sum(len(t.messages) for t in thread_memory.threads.values())
        total_entities = sum(len(t.entities) for t in thread_memory.threads.values())
        total_summarized = sum(t.summarized_count for t in thread_memory.threads.values())
        status_text += f"\n\n*Thread Memory (Enhanced)*\n" \
                      f"• Active threads: {active_threads}\n" \
                      f"• Messages in window: {total_messages}\n" \
                      f"• Messages summarized: {total_summarized}\n" \
                      f"• Entities tracked: {total_entities}\n" \
                      f"• Entity extraction: {'enabled' if ENTITY_EXTRACTOR_AVAILABLE else 'disabled'}"

        await respond(status_text)

    except Exception as e:
        await respond(f":warning: Error getting status: {str(e)}")


@app.command("/chorus-reload")
async def handle_reload_command(ack, respond):
    """Handle /chorus-reload command - reload the registry data."""
    await ack()

    try:
        # Trigger registry reload via RAG server
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{RAG_SERVER_URL}/reload", timeout=aiohttp.ClientTimeout(total=10)) as response:
                result = await response.json()

        if result.get("status") == "ok":
            await respond(f":arrows_counterclockwise: Registry reloaded successfully!\n" \
                         f"• People: {result.get('stats', {}).get('people', 0)}\n" \
                         f"• Publications: {result.get('stats', {}).get('publications', 0)}")
        else:
            await respond(f":warning: Reload returned: {result}")

    except Exception as e:
        await respond(f":x: Error reloading registry: {str(e)}")


@app.command("/chorus-help")
async def handle_help_command(ack, respond):
    """Handle /chorus-help command - show available commands and query examples."""
    await ack()

    help_text = """*CHORUS - Knowledge Lab Assistant*

*How to Talk to Chorus:*
• *In channels:* @mention Chorus to start a conversation
• *In threads:* After mentioning once, just reply - no @mention needed!
• *In DMs:* Message directly anytime

*Commands:*
• `/chorus-status` - Check system status and metrics
• `/chorus-reload` - Reload the registry data
• `/chorus-help` - Show this help message

*Query Examples:*
• "Who is James Evans?" - Person lookup with publication metrics
• "Who works on NLP?" - Topic search (aliases like NLP, ML, AI work)
• "UChicago researchers in network science" - Compound filters
• "Papers by James Evans" - Publication search
• "What compute resources are available?" - Lab infrastructure
• "Tell me about the APTO project" - Project info

*Conversation Memory:*
Chorus remembers your conversation within each thread. Ask follow-up questions like:
• "What's their h-index?"
• "Tell me more about their recent work"
• "Who else works on similar topics?"

*Tips:*
• Topic abbreviations work: NLP, ML, AI, CSS, HCI, etc.
• Combine filters: "faculty at UChicago working on AI"
• Chorus can think with you - try exploring ideas together!
"""
    await respond(help_text)


# =============================================================================
# Helper Functions
# =============================================================================

async def process_message_with_memory(
    text: str,
    channel: str,
    thread_ts: Optional[str],
    say,
    is_dm: bool = False
):
    """
    Process a message with conversation memory.

    Args:
        text: The user's message text
        channel: The Slack channel ID
        thread_ts: The thread timestamp (None for top-level messages)
        say: The Slack say function
        is_dm: Whether this is a direct message
    """
    # Get or create thread conversation
    thread = thread_memory.get_or_create(channel, thread_ts)

    # Get conversation context for the agent
    thread_context = thread.get_context_string()

    print(f"[PROCESS] Query with {len(thread.messages)} messages of context")

    try:
        # Store user message in memory
        thread.add_user_message(text)

        # Query the agent with thread context for reformulation
        response = await agent.query(text, thread_context=thread_context)

        # Store assistant response in memory
        thread.add_assistant_message(response)

        # Send response (in thread if applicable)
        if is_dm:
            await say(text=response)
        else:
            await say(text=response, thread_ts=thread_ts)

        print(f"[PROCESS] Response sent ({len(response)} chars)")
        return True

    except Exception as e:
        print(f"[PROCESS] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        error_msg = f"Sorry, I encountered an error: {str(e)}"
        if is_dm:
            await say(text=error_msg)
        else:
            await say(text=error_msg, thread_ts=thread_ts)
        return False


# =============================================================================
# Event Handlers
# =============================================================================

@app.event("app_mention")
async def handle_mention(event, say):
    """Handle @mentions of the bot in channels."""
    print(f"[MENTION] Received app_mention event")
    print(f"[MENTION] Text: {event.get('text', '')[:100]}")

    channel = event.get("channel")
    # Use thread_ts if in a thread, otherwise use ts (this message starts a new thread)
    thread_ts = event.get("thread_ts") or event.get("ts")

    # Extract the message text (remove the bot mention)
    text = event.get("text", "")
    text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    print(f"[MENTION] Cleaned text: {text}")

    # Mark this thread as active for the bot
    thread_memory.activate_thread(channel, thread_ts)
    print(f"[MENTION] Activated thread {channel}:{thread_ts}")

    if not text:
        print(f"[MENTION] Empty text, sending greeting")
        await say(
            text="Hi! I'm Chorus, the lab's intellectual companion. Ask me about:\n" \
                 "• Lab members and their research (e.g., \"Who works on NLP?\")\n" \
                 "• Publications and citations (e.g., \"Papers by James Evans\")\n" \
                 "• Projects and grants (e.g., \"Tell me about APTO\")\n" \
                 "• Compute resources and tools (e.g., \"What GPUs are available?\")\n" \
                 "Or just think out loud with me. Type `/chorus-help` for more examples.\n\n" \
                 "_I'll respond to all messages in this thread - no need to @mention me again._",
            thread_ts=thread_ts
        )
        return

    print(f"[MENTION] Processing query: {text}")
    await process_message_with_memory(text, channel, thread_ts, say)


@app.event("message")
async def handle_message(event, say):
    """Handle messages - DMs and thread replies where bot is active."""
    # Skip messages with subtypes (edits, deletes, bot messages, etc.)
    if event.get("subtype"):
        return

    # Skip bot messages to prevent loops
    if event.get("bot_id"):
        return

    channel = event.get("channel")
    channel_type = event.get("channel_type")
    thread_ts = event.get("thread_ts")  # None if not in a thread
    text = event.get("text", "").strip()

    if not text:
        return

    # Case 1: Direct Message - always respond
    if channel_type == "im":
        print(f"[DM] Processing: {text[:100]}")
        # Activate thread memory for DM conversations
        thread_memory.activate_thread(channel, "dm")
        await process_message_with_memory(text, channel, "dm", say, is_dm=True)
        return

    # Case 2: Thread reply in a channel where bot is active
    if thread_ts and thread_memory.is_bot_active_in_thread(channel, thread_ts):
        # Check if this message contains a bot mention (handled by app_mention)
        if re.search(r'<@[A-Z0-9]+>', text):
            print(f"[THREAD] Skipping - contains mention (handled by app_mention)")
            return

        print(f"[THREAD] Auto-responding in active thread: {text[:100]}")
        await process_message_with_memory(text, channel, thread_ts, say)
        return

    # Case 3: Regular channel message - ignore (requires @mention)
    # This is handled by app_mention event


# =============================================================================
# Main
# =============================================================================

async def main():
    """Start the Slack bot."""
    global agent

    # Check required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("\nRequired variables:")
        print("  SLACK_BOT_TOKEN   - Bot User OAuth Token (xoxb-...)")
        print("  SLACK_APP_TOKEN   - App-Level Token for Socket Mode (xapp-...)")
        print("  ANTHROPIC_API_KEY - Anthropic API key")
        sys.exit(1)

    # Check RAG server health
    health = await call_rag_async("/health")
    rag_available = "error" not in health

    if rag_available:
        print(f"Connected to RAG server ({health.get('chunks', 0)} chunks indexed)")
    else:
        print(f"Warning: RAG server not reachable at {RAG_SERVER_URL}")
        print(f"  {health.get('error', 'Unknown error')}")
        print("  Bot will start anyway - RAG queries will fail until server is up")

    # Initialize the agent (orchestrator or legacy)
    if USE_ORCHESTRATOR:
        agent = ChorusOrchestrator(
            enable_caching=True,
            enable_parallel=True,
            rag_available=rag_available
        )
        print("\nChorus Orchestrator initialized with:")
        print(f"  - Multi-agent routing: enabled")
        print(f"  - Specialists: Memory (Haiku), Mechanic (Haiku), Muse (Sonnet), Matchmaker (Haiku)")
        print(f"  - Response caching: enabled")
        print(f"  - Parallel execution: enabled")
        print(f"  - RAG tools: {'enabled' if rag_available else 'disabled'}")
        print(f"  - Hybrid search: structured registry + semantic")
        print(f"  - Topic aliases: NLP, ML, AI, CSS, etc.")
        print(f"  - Compound queries: institution + topic filters")
        print(f"  - Thread memory: conversation context preserved")
        print(f"  - Auto-reply in threads: no @mention needed after first")
    else:
        agent = ChorusAgent(
            enable_caching=True,
            enable_routing=True,
            enable_prompt_caching=True,
            rag_available=rag_available
        )
        print("\nChorus Agent (legacy) initialized with:")
        print(f"  - Response caching: enabled")
        print(f"  - Model routing: enabled (Haiku for simple, Sonnet for complex)")
        print(f"  - Prompt caching: enabled")
        print(f"  - RAG tools: {'enabled' if rag_available else 'disabled'}")

    # Start Socket Mode handler
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    print("\nCHORUS Slack bot starting...")
    print("Listening for @mentions in channels and direct messages.\n")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
