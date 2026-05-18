"""
Slack Data Collector

Parses Slack export data to extract messages by user for disposition analysis.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime

from ..processors.disposition_extractor import TextSample


class SlackCollector:
    """
    Collects text samples from Slack export data.

    Expects Slack export format:
    - channels.json: List of channels
    - users.json: List of users
    - {channel_name}/*.json: Messages per day
    """

    def __init__(self, export_path: str):
        """
        Initialize collector with path to Slack export directory.

        Args:
            export_path: Path to unzipped Slack export
        """
        self.export_path = Path(export_path)
        self._users: Dict[str, Dict] = {}
        self._channels: Dict[str, Dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load users and channels metadata."""
        users_path = self.export_path / "users.json"
        if users_path.exists():
            with open(users_path) as f:
                users_data = json.load(f)
                self._users = {u["id"]: u for u in users_data}

        channels_path = self.export_path / "channels.json"
        if channels_path.exists():
            with open(channels_path) as f:
                channels_data = json.load(f)
                self._channels = {c["id"]: c for c in channels_data}

    def get_user_name(self, user_id: str) -> str:
        """Get display name for a user ID."""
        user = self._users.get(user_id, {})
        return (
            user.get("real_name") or
            user.get("profile", {}).get("real_name") or
            user.get("name") or
            user_id
        )

    def get_user_id_by_name(self, name: str) -> Optional[str]:
        """Find user ID by name (partial match)."""
        name_lower = name.lower()
        for uid, user in self._users.items():
            real_name = user.get("real_name", "").lower()
            display_name = user.get("profile", {}).get("display_name", "").lower()
            username = user.get("name", "").lower()

            if name_lower in real_name or name_lower in display_name or name_lower == username:
                return uid
        return None

    def list_users(self) -> List[Dict]:
        """List all users with basic info."""
        return [
            {
                "id": uid,
                "name": self.get_user_name(uid),
                "is_bot": user.get("is_bot", False),
            }
            for uid, user in self._users.items()
        ]

    def list_channels(self) -> List[str]:
        """List all channel names."""
        # Also check for channel directories
        channel_dirs = [
            d.name for d in self.export_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return channel_dirs

    def iter_messages(
        self,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
        min_length: int = 20,
    ) -> Iterator[Dict]:
        """
        Iterate over messages, optionally filtered.

        Args:
            channel: Filter to specific channel
            user_id: Filter to specific user
            min_length: Minimum message length

        Yields:
            Message dicts with text, user, channel, timestamp
        """
        channels = [channel] if channel else self.list_channels()

        for ch in channels:
            channel_path = self.export_path / ch

            if not channel_path.is_dir():
                continue

            for json_file in sorted(channel_path.glob("*.json")):
                try:
                    with open(json_file) as f:
                        messages = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                for msg in messages:
                    # Skip non-message types
                    if msg.get("subtype") in ["channel_join", "channel_leave", "bot_message"]:
                        continue

                    msg_user = msg.get("user")
                    msg_text = msg.get("text", "")

                    # Apply filters
                    if user_id and msg_user != user_id:
                        continue

                    if len(msg_text) < min_length:
                        continue

                    yield {
                        "text": msg_text,
                        "user": msg_user,
                        "user_name": self.get_user_name(msg_user) if msg_user else None,
                        "channel": ch,
                        "timestamp": msg.get("ts"),
                        "thread_ts": msg.get("thread_ts"),
                        "reactions": msg.get("reactions", []),
                    }

    def collect_samples_for_user(
        self,
        user_id: str,
        max_samples: int = 200,
        min_length: int = 30,
        include_threads: bool = True,
    ) -> List[TextSample]:
        """
        Collect text samples for a specific user.

        Args:
            user_id: Slack user ID
            max_samples: Maximum samples to collect
            min_length: Minimum message length
            include_threads: Include thread replies

        Returns:
            List of TextSample objects
        """
        samples = []

        for msg in self.iter_messages(user_id=user_id, min_length=min_length):
            if len(samples) >= max_samples:
                break

            # Convert timestamp
            ts = msg.get("timestamp")
            timestamp = None
            if ts:
                try:
                    timestamp = datetime.fromtimestamp(float(ts)).isoformat()
                except (ValueError, TypeError):
                    pass

            # Build context
            context = f"#{msg['channel']}"
            if msg.get("thread_ts") and msg.get("thread_ts") != msg.get("timestamp"):
                context += " (thread reply)"

            samples.append(TextSample(
                text=msg["text"],
                source_type="slack",
                source_id=f"slack:{msg['channel']}:{ts}",
                timestamp=timestamp,
                context=context,
            ))

        return samples

    def collect_all_users(
        self,
        min_messages: int = 20,
        max_samples_per_user: int = 100,
    ) -> Dict[str, List[TextSample]]:
        """
        Collect samples for all active users.

        Args:
            min_messages: Minimum messages required to include user
            max_samples_per_user: Max samples per user

        Returns:
            Dict mapping user_id -> list of TextSamples
        """
        # First pass: count messages per user
        user_counts: Dict[str, int] = {}
        for msg in self.iter_messages(min_length=20):
            uid = msg.get("user")
            if uid:
                user_counts[uid] = user_counts.get(uid, 0) + 1

        # Filter to active users
        active_users = [
            uid for uid, count in user_counts.items()
            if count >= min_messages and not self._users.get(uid, {}).get("is_bot", False)
        ]

        # Collect samples
        result = {}
        for uid in active_users:
            samples = self.collect_samples_for_user(uid, max_samples=max_samples_per_user)
            if samples:
                result[uid] = samples

        return result

    def extract_questions(
        self,
        user_id: str,
        max_messages: int = 500,
    ) -> List[Dict]:
        """
        Extract messages that appear to be questions.

        Args:
            user_id: User to analyze
            max_messages: Max messages to scan

        Returns:
            List of question messages with context
        """
        questions = []
        count = 0

        for msg in self.iter_messages(user_id=user_id, min_length=10):
            if count >= max_messages:
                break
            count += 1

            text = msg["text"]

            # Simple heuristics for questions
            is_question = (
                "?" in text or
                text.lower().startswith(("what", "why", "how", "when", "where", "who", "is ", "are ", "do ", "does ", "can ", "could ", "would ", "should "))
            )

            if is_question:
                questions.append({
                    "text": text,
                    "channel": msg["channel"],
                    "timestamp": msg["timestamp"],
                    "is_thread": bool(msg.get("thread_ts")),
                })

        return questions
