"""Shared configuration utilities for CHORUS scripts."""

import os
from pathlib import Path


def load_dotenv():
    """Load environment variables from .env file in project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key, value)


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get a boolean from an environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def get_env_float(name: str, default: float = 0.0) -> float:
    """Get a float from an environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default
