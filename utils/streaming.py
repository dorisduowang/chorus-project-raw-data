"""
Streaming utilities for CHORUS.

Provides callback-based streaming for Anthropic API responses,
with support for different output targets (CLI, Slack, etc.).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List


# =============================================================================
# Stream Callback Protocol
# =============================================================================

class StreamCallback(ABC):
    """
    Abstract base class for streaming callbacks.

    Implement this to handle streaming output to different targets
    (CLI, Slack, WebSocket, etc.).
    """

    @abstractmethod
    def on_text(self, text: str) -> None:
        """Called when a text chunk is received."""
        pass

    @abstractmethod
    def on_complete(self, full_text: str) -> None:
        """Called when streaming is complete."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs during streaming."""
        pass

    def on_tool_use(self, tool_name: str, tool_input: dict) -> None:
        """Called when a tool use block is detected (optional)."""
        pass


# =============================================================================
# Text Accumulator
# =============================================================================

@dataclass
class TextAccumulator:
    """
    Accumulates text chunks with smart flushing.

    Useful for buffering text before sending updates
    (e.g., to avoid too-frequent Slack updates).
    """

    buffer: str = ""
    total_text: str = ""
    min_flush_chars: int = 50
    last_flush_time: float = field(default_factory=time.time)
    min_flush_interval: float = 0.5  # seconds

    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer. Returns accumulated text if should flush.

        Returns:
            Accumulated text if flush threshold reached, None otherwise.
        """
        self.buffer += text
        self.total_text += text

        # Check if should flush
        now = time.time()
        time_elapsed = now - self.last_flush_time >= self.min_flush_interval
        enough_chars = len(self.buffer) >= self.min_flush_chars

        if time_elapsed and enough_chars:
            result = self.total_text
            self.buffer = ""
            self.last_flush_time = now
            return result

        return None

    def flush(self) -> str:
        """Force flush and return all accumulated text."""
        self.buffer = ""
        self.last_flush_time = time.time()
        return self.total_text

    def reset(self) -> None:
        """Reset the accumulator."""
        self.buffer = ""
        self.total_text = ""
        self.last_flush_time = time.time()


# =============================================================================
# CLI Print Callback
# =============================================================================

class PrintStreamCallback(StreamCallback):
    """
    Streams text directly to stdout for CLI usage.

    Prints each chunk immediately without buffering.
    """

    def __init__(self, prefix: str = "", end_newline: bool = True):
        """
        Args:
            prefix: Optional prefix to print before streaming starts
            end_newline: Whether to print newline on completion
        """
        self.prefix = prefix
        self.end_newline = end_newline
        self._started = False
        self._full_text = ""

    def on_text(self, text: str) -> None:
        """Print text chunk immediately."""
        if not self._started and self.prefix:
            print(self.prefix, end="", flush=True)
            self._started = True

        print(text, end="", flush=True)
        self._full_text += text

    def on_complete(self, full_text: str) -> None:
        """Print newline on completion."""
        if self.end_newline:
            print()  # New line after streaming

    def on_error(self, error: Exception) -> None:
        """Print error to stderr."""
        import sys
        print(f"\nStreaming error: {error}", file=sys.stderr)

    def get_text(self) -> str:
        """Get accumulated text."""
        return self._full_text


# =============================================================================
# Slack Stream Callback
# =============================================================================

class SlackStreamCallback(StreamCallback):
    """
    Streams text to Slack with rate-limited message updates.

    Posts an initial message, then updates it periodically as
    new text arrives. Rate limiting prevents hitting Slack API limits.
    """

    def __init__(
        self,
        client: Any,  # slack_sdk.WebClient
        channel: str,
        thread_ts: Optional[str] = None,
        initial_text: str = "Thinking...",
        min_update_interval: float = 0.5,
        min_text_delta: int = 50,
    ):
        """
        Args:
            client: Slack WebClient instance
            channel: Channel ID to post to
            thread_ts: Optional thread timestamp for replies
            initial_text: Initial message text
            min_update_interval: Minimum seconds between updates
            min_text_delta: Minimum characters before updating
        """
        self.client = client
        self.channel = channel
        self.thread_ts = thread_ts
        self.initial_text = initial_text
        self.min_update_interval = min_update_interval
        self.min_text_delta = min_text_delta

        self._message_ts: Optional[str] = None
        self._accumulator = TextAccumulator(
            min_flush_chars=min_text_delta,
            min_flush_interval=min_update_interval,
        )
        self._update_count = 0

    def post_initial(self) -> str:
        """Post the initial 'thinking' message. Returns message ts."""
        result = self.client.chat_postMessage(
            channel=self.channel,
            text=self.initial_text,
            thread_ts=self.thread_ts,
        )
        self._message_ts = result["ts"]
        return self._message_ts

    def on_text(self, text: str) -> None:
        """Accumulate text and update Slack when threshold reached."""
        if not self._message_ts:
            self.post_initial()

        # Add to accumulator and check if should update
        accumulated = self._accumulator.add(text)
        if accumulated:
            self._update_message(accumulated)

    def on_complete(self, full_text: str) -> None:
        """Final update with complete text."""
        if self._message_ts:
            self._update_message(full_text)

    def on_error(self, error: Exception) -> None:
        """Update message with error indicator."""
        if self._message_ts:
            error_text = f"Error: {str(error)[:200]}"
            self._update_message(error_text)

    def _update_message(self, text: str) -> None:
        """Update the Slack message with new text."""
        try:
            self.client.chat_update(
                channel=self.channel,
                ts=self._message_ts,
                text=text,
            )
            self._update_count += 1
        except Exception as e:
            # Log but don't fail on update errors
            print(f"Slack update error: {e}")

    @property
    def message_ts(self) -> Optional[str]:
        """Get the message timestamp."""
        return self._message_ts

    @property
    def update_count(self) -> int:
        """Get number of updates made."""
        return self._update_count


# =============================================================================
# Streaming Helper Functions
# =============================================================================

def stream_claude_response(
    client: Any,  # anthropic.Anthropic
    messages: List[dict],
    callback: StreamCallback,
    model: str = "claude-sonnet-4-20250514",
    system: Optional[str] = None,
    max_tokens: int = 4096,
    tools: Optional[List[dict]] = None,
    extra_headers: Optional[dict] = None,
) -> str:
    """
    Stream a Claude response using the provided callback.

    Args:
        client: Anthropic client instance
        messages: Conversation messages
        callback: StreamCallback to receive text chunks
        model: Model to use
        system: Optional system prompt
        max_tokens: Maximum tokens in response
        tools: Optional tools list
        extra_headers: Optional extra headers (e.g., for web search beta)

    Returns:
        Complete response text

    Example:
        callback = PrintStreamCallback()
        text = stream_claude_response(client, messages, callback)
    """
    # Build request kwargs
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if system:
        kwargs["system"] = system

    if tools:
        kwargs["tools"] = tools

    # Handle extra headers
    if extra_headers:
        # For web search beta header
        kwargs["extra_headers"] = extra_headers

    full_text = ""

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                full_text += text
                callback.on_text(text)

        callback.on_complete(full_text)

    except Exception as e:
        callback.on_error(e)
        raise

    return full_text


async def stream_claude_response_async(
    client: Any,  # anthropic.AsyncAnthropic
    messages: List[dict],
    callback: StreamCallback,
    model: str = "claude-sonnet-4-20250514",
    system: Optional[str] = None,
    max_tokens: int = 4096,
    tools: Optional[List[dict]] = None,
    extra_headers: Optional[dict] = None,
) -> str:
    """
    Async version of stream_claude_response.

    Args:
        client: AsyncAnthropic client instance
        messages: Conversation messages
        callback: StreamCallback to receive text chunks
        model: Model to use
        system: Optional system prompt
        max_tokens: Maximum tokens in response
        tools: Optional tools list
        extra_headers: Optional extra headers

    Returns:
        Complete response text
    """
    # Build request kwargs
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if system:
        kwargs["system"] = system

    if tools:
        kwargs["tools"] = tools

    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    full_text = ""

    try:
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                full_text += text
                callback.on_text(text)

        callback.on_complete(full_text)

    except Exception as e:
        callback.on_error(e)
        raise

    return full_text


# =============================================================================
# Factory Functions
# =============================================================================

def create_print_callback(prefix: str = "\nCHORUS: ") -> PrintStreamCallback:
    """Create a PrintStreamCallback with standard settings."""
    return PrintStreamCallback(prefix=prefix, end_newline=True)


def create_slack_callback(
    client: Any,
    channel: str,
    thread_ts: Optional[str] = None,
) -> SlackStreamCallback:
    """Create a SlackStreamCallback with standard settings."""
    return SlackStreamCallback(
        client=client,
        channel=channel,
        thread_ts=thread_ts,
        min_update_interval=0.5,
        min_text_delta=50,
    )
