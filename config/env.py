"""
CHORUS Environment Configuration

Centralized environment variable loading and typed accessors.
This module eliminates duplicate load_dotenv implementations across the codebase.

Features:
- Single .env file loading implementation
- Typed environment variable getters (bool, int, float, str)
- Project root detection
- Thread-safe loading (idempotent)
"""

import os
import threading
from pathlib import Path
from typing import Optional


# Thread-safe loading flag
_env_loaded = False
_env_lock = threading.Lock()


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns the directory containing this config module's parent.
    """
    return Path(__file__).parent.parent


def load_dotenv(env_path: Optional[Path] = None) -> bool:
    """
    Load environment variables from .env file.

    This function is idempotent - calling it multiple times has no effect
    after the first successful load.

    Args:
        env_path: Optional path to .env file. Defaults to project root/.env

    Returns:
        True if .env was loaded, False if already loaded or file not found
    """
    global _env_loaded

    # Fast path - already loaded
    if _env_loaded:
        return False

    with _env_lock:
        # Double-check after acquiring lock
        if _env_loaded:
            return False

        if env_path is None:
            env_path = get_project_root() / ".env"

        if not env_path.exists():
            _env_loaded = True  # Mark as loaded even if no file
            return False

        with open(env_path) as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Must have = separator
                if "=" not in line:
                    continue

                # Parse key=value
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Remove quotes from value
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                else:
                    # Strip quotes from either end
                    value = value.strip('"').strip("'")

                # Only set if key is valid and not already in environment
                if key and key not in os.environ:
                    os.environ[key] = value

        _env_loaded = True
        return True


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.

    Truthy values: "true", "1", "yes", "on" (case-insensitive)

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value
    """
    val = os.environ.get(key, "").lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value
    """
    val = os.environ.get(key, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Float value
    """
    val = os.environ.get(key, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_env_str(key: str, default: str = "") -> str:
    """
    Get string environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        String value
    """
    return os.environ.get(key, default)


def reset_env_loaded() -> None:
    """
    Reset the env loaded flag (for testing purposes).
    """
    global _env_loaded
    with _env_lock:
        _env_loaded = False
