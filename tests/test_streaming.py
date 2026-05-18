"""
Unit tests for streaming utilities.

Tests the streaming callback system for CLI and Slack output.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from utils.streaming import (
    StreamCallback,
    PrintStreamCallback,
    SlackStreamCallback,
    TextAccumulator,
    stream_claude_response,
    create_print_callback,
    create_slack_callback,
)


# =============================================================================
# TextAccumulator Tests
# =============================================================================

class TestTextAccumulator:
    """Tests for TextAccumulator class."""

    def test_add_below_threshold(self):
        """Text below threshold doesn't flush."""
        acc = TextAccumulator(min_flush_chars=50, min_flush_interval=0.1)
        result = acc.add("hello")
        assert result is None
        assert acc.buffer == "hello"
        assert acc.total_text == "hello"

    def test_add_above_threshold_but_time_not_elapsed(self):
        """Text above char threshold but time not elapsed doesn't flush."""
        acc = TextAccumulator(min_flush_chars=10, min_flush_interval=1.0)
        result = acc.add("x" * 20)
        # Time hasn't elapsed, so no flush
        assert result is None

    def test_add_flushes_when_both_thresholds_met(self):
        """Flush when both char and time thresholds met."""
        acc = TextAccumulator(min_flush_chars=10, min_flush_interval=0.01)
        acc.add("x" * 15)
        time.sleep(0.02)  # Wait for time threshold
        result = acc.add("y" * 5)  # Add more to trigger check
        assert result == "x" * 15 + "y" * 5

    def test_flush_returns_all_text(self):
        """Force flush returns all accumulated text."""
        acc = TextAccumulator()
        acc.add("hello ")
        acc.add("world")
        result = acc.flush()
        assert result == "hello world"
        assert acc.buffer == ""

    def test_reset_clears_everything(self):
        """Reset clears all state."""
        acc = TextAccumulator()
        acc.add("hello")
        acc.reset()
        assert acc.buffer == ""
        assert acc.total_text == ""


# =============================================================================
# PrintStreamCallback Tests
# =============================================================================

class TestPrintStreamCallback:
    """Tests for PrintStreamCallback class."""

    def test_on_text_accumulates(self):
        """on_text accumulates text."""
        callback = PrintStreamCallback(end_newline=False)
        with patch('builtins.print') as mock_print:
            callback.on_text("hello")
            callback.on_text(" world")
            assert callback.get_text() == "hello world"

    def test_prefix_printed_once(self):
        """Prefix is printed only on first text."""
        callback = PrintStreamCallback(prefix=">>> ", end_newline=False)
        with patch('builtins.print') as mock_print:
            callback.on_text("first")
            callback.on_text("second")
            # First call should have prefix
            calls = mock_print.call_args_list
            assert calls[0][0][0] == ">>> "

    def test_on_complete_prints_newline(self):
        """on_complete prints newline when configured."""
        callback = PrintStreamCallback(end_newline=True)
        with patch('builtins.print') as mock_print:
            callback.on_complete("full text")
            mock_print.assert_called_once_with()

    def test_on_error_prints_to_stderr(self):
        """on_error prints error message."""
        callback = PrintStreamCallback()
        with patch('builtins.print') as mock_print:
            callback.on_error(ValueError("test error"))
            mock_print.assert_called()


# =============================================================================
# SlackStreamCallback Tests
# =============================================================================

class TestSlackStreamCallback:
    """Tests for SlackStreamCallback class."""

    def test_post_initial_creates_message(self):
        """post_initial creates Slack message and returns ts."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "12345.67890"}

        callback = SlackStreamCallback(
            client=mock_client,
            channel="C123",
            initial_text="Thinking...",
        )

        ts = callback.post_initial()

        assert ts == "12345.67890"
        assert callback.message_ts == "12345.67890"
        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="Thinking...",
            thread_ts=None,
        )

    def test_on_text_posts_initial_if_needed(self):
        """on_text automatically posts initial message."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "12345"}

        callback = SlackStreamCallback(
            client=mock_client,
            channel="C123",
            min_update_interval=0.01,
            min_text_delta=5,
        )

        callback.on_text("hello world this is a test")
        time.sleep(0.02)
        callback.on_text(" more text")

        # Should have posted initial message
        assert mock_client.chat_postMessage.called

    def test_on_complete_updates_final(self):
        """on_complete sends final update."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "12345"}

        callback = SlackStreamCallback(
            client=mock_client,
            channel="C123",
        )

        callback.post_initial()
        callback.on_complete("Final response text")

        mock_client.chat_update.assert_called_with(
            channel="C123",
            ts="12345",
            text="Final response text",
        )

    def test_rate_limiting(self):
        """Updates are rate limited."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "12345"}

        callback = SlackStreamCallback(
            client=mock_client,
            channel="C123",
            min_update_interval=0.5,
            min_text_delta=10,
        )

        callback.post_initial()

        # Rapid updates should be buffered
        for i in range(10):
            callback.on_text(f"chunk{i}")

        # Should have fewer updates than chunks due to rate limiting
        update_count = mock_client.chat_update.call_count
        assert update_count < 10

    def test_thread_ts_passed_to_message(self):
        """thread_ts is included in message."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "12345"}

        callback = SlackStreamCallback(
            client=mock_client,
            channel="C123",
            thread_ts="11111.22222",
        )

        callback.post_initial()

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="Thinking...",
            thread_ts="11111.22222",
        )


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_print_callback(self):
        """create_print_callback returns configured callback."""
        callback = create_print_callback(prefix="Test: ")
        assert isinstance(callback, PrintStreamCallback)
        assert callback.prefix == "Test: "

    def test_create_slack_callback(self):
        """create_slack_callback returns configured callback."""
        mock_client = Mock()
        callback = create_slack_callback(
            client=mock_client,
            channel="C123",
            thread_ts="12345",
        )
        assert isinstance(callback, SlackStreamCallback)
        assert callback.channel == "C123"
        assert callback.thread_ts == "12345"


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================

class TestStreamClaudeResponse:
    """Integration tests for stream_claude_response."""

    def test_stream_response_basic(self):
        """Test basic streaming flow with mock client."""
        # Create mock stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)
        mock_stream.text_stream = ["Hello", " ", "World", "!"]

        mock_client = Mock()
        mock_client.messages.stream.return_value = mock_stream

        callback = PrintStreamCallback(end_newline=False)

        with patch('builtins.print'):
            result = stream_claude_response(
                client=mock_client,
                messages=[{"role": "user", "content": "test"}],
                callback=callback,
            )

        assert result == "Hello World!"

    def test_stream_response_with_system(self):
        """Test streaming with system prompt."""
        mock_stream = MagicMock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)
        mock_stream.text_stream = ["Response"]

        mock_client = Mock()
        mock_client.messages.stream.return_value = mock_stream

        callback = PrintStreamCallback(end_newline=False)

        with patch('builtins.print'):
            stream_claude_response(
                client=mock_client,
                messages=[{"role": "user", "content": "test"}],
                callback=callback,
                system="You are a helpful assistant.",
            )

        # Verify system was passed
        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs.get("system") == "You are a helpful assistant."

    def test_stream_response_with_tools(self):
        """Test streaming with tools."""
        mock_stream = MagicMock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)
        mock_stream.text_stream = ["Tool response"]

        mock_client = Mock()
        mock_client.messages.stream.return_value = mock_stream

        callback = PrintStreamCallback(end_newline=False)
        tools = [{"type": "web_search_20250305", "name": "web_search"}]

        with patch('builtins.print'):
            stream_claude_response(
                client=mock_client,
                messages=[{"role": "user", "content": "test"}],
                callback=callback,
                tools=tools,
            )

        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs.get("tools") == tools

    def test_stream_response_error_handling(self):
        """Test error handling during streaming."""
        mock_client = Mock()
        mock_client.messages.stream.side_effect = Exception("API Error")

        callback = PrintStreamCallback(end_newline=False)

        with patch('builtins.print'):
            with pytest.raises(Exception, match="API Error"):
                stream_claude_response(
                    client=mock_client,
                    messages=[{"role": "user", "content": "test"}],
                    callback=callback,
                )


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
