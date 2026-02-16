"""Bootstrap configuration for rlmagents with DeepSeek + MiniMax."""

import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from rlmagents import create_rlm_agent


def _load_dotenv_if_available() -> None:
    """Load ``.env`` if possible without requiring ``python-dotenv``."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ModuleNotFoundError:
        for line in env_path.read_text().splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _init_model(*, model_id: str, model_alias: str, api_key: str, base_url: str) -> Any:
    """Initialize a chat model using provider fallbacks.

    Tries the explicit provider first, then an OpenAI-compatible fallback.
    """
    errors: list[str] = []

    attempts = [
        {
            "label": f"{model_id}",
            "model": model_id,
            "kwargs": {"api_key": api_key, "base_url": base_url},
        },
        {
            "label": f"openai:{model_alias}",
            "model": model_alias,
            "kwargs": {
                "api_key": api_key,
                "base_url": base_url,
                "model_provider": "openai",
            },
        },
    ]

    for attempt in attempts:
        try:
            return init_chat_model(attempt["model"], **attempt["kwargs"])
        except Exception as exc:  # pragma: no cover
            errors.append(f"{attempt['label']}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Unable to initialize configured model. Verify provider packages and model ids.\n"
        + "\n".join(errors)
    )


def create_configured_agent(**kwargs: Any) -> Any:
    """Create an rlmagents agent with DeepSeek (main) + MiniMax (sub-query)."""

    _load_dotenv_if_available()

    main_model = _init_model(
        model_id="deepseek-chat",
        model_alias="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )

    sub_query_model = _init_model(
        model_id="MiniMax-Text-01",
        model_alias="MiniMax-Text-01",
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimaxi.chat/v1",
    )

    return create_rlm_agent(
        model=main_model,
        sub_query_model=sub_query_model,
        sub_query_timeout=120.0,
        sandbox_timeout=300.0,
        **kwargs,
    )
