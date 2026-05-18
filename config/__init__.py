"""
CHORUS Configuration Module

Centralized configuration loading and environment variable handling.
All modules should import environment utilities from here.

Usage:
    from config import load_dotenv, get_env_bool, get_env_int, get_env_float

    # At module start
    load_dotenv()

    # Get typed environment variables
    DEBUG = get_env_bool("DEBUG", False)
    PORT = get_env_int("PORT", 8765)
    THRESHOLD = get_env_float("THRESHOLD", 0.5)
"""

from config.env import (
    load_dotenv,
    get_env_bool,
    get_env_int,
    get_env_float,
    get_env_str,
    get_project_root,
)

__all__ = [
    "load_dotenv",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    "get_env_str",
    "get_project_root",
]
