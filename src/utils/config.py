"""
Configuration and Environment Loader.
Reads settings from .env file or system environment variables.
"""

import os
from pathlib import Path


def load_env(env_path: str = None) -> None:
    """Manually parse .env file without external dependencies."""
    if env_path is None:
        # Search in current working directory and project root
        env_file = Path(".env")
        if not env_file.exists():
            env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    else:
        env_file = Path(env_path)

    if not env_file.exists():
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val


# Automatically attempt to load .env on import
load_env()


def get_gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def get_judge_model() -> str:
    """Default model for the Critic (LLMJudge). Defaults to gemini-3.8-flash."""
    return os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")


def get_generator_model() -> str:
    """Default model for the Actor (ExplanationGenerator). Defaults to gemini-3.5-flash-lite."""
    return os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash-lite")


