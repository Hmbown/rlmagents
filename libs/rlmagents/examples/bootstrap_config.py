"""Bootstrap configuration for rlmagents with DeepSeek + MiniMax."""

import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from rlmagents import create_rlm_agent

_MODEL_INIT_CACHE: dict[str, tuple[bool, Any | str]] = {}
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MINIMAX_MODEL = "minimax-m2.5"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.chat/v1"
_MODEL_ALIAS_MAP = {
    "deepseek": "deepseek-chat",
    "deepseek-chat": "deepseek-chat",
    "minimax": "minimax-m2.5",
    "minimax-m2.5": "minimax-m2.5",
    "minimax-text-01": "minimax-m2.5",
}


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


def _normalize_alias(alias: str) -> str:
    """Normalize model aliases for consistent provider initialization."""
    key = alias.strip()
    if not key:
        return alias
    mapped = _MODEL_ALIAS_MAP.get(key.lower())
    return mapped if mapped is not None else alias


def _parse_env_value(name: str, default: str) -> str:
    """Read environment value with fallback."""
    value = os.environ.get(name, "")
    value = value.strip() if value else ""
    return value if value else default


def _classify_model_error(exc: Exception) -> str:
    """Classify provider/model errors for clearer sub_query diagnostics."""
    msg = str(exc).lower()
    if any(token in msg for token in ("unauthorized", "invalid api key", "authentication", "401")):
        return "auth"
    if any(token in msg for token in ("rate limit", "429", "too many requests", "quota")):
        return "rate_limit"
    if any(token in msg for token in ("model not found", "unknown model", "404", "does not exist")):
        return "model_not_found"
    return "unknown"


def _init_model(*, model_id: str, model_alias: str, api_key: str, base_url: str) -> Any:
    """Initialize a chat model using provider fallbacks.

    Tries the explicit provider first, then an OpenAI-compatible fallback.
    """
    normalized_alias = _normalize_alias(model_alias)
    cache_key = f"{model_id}|{normalized_alias}|{base_url}"
    if cache_key in _MODEL_INIT_CACHE:
        ok, value = _MODEL_INIT_CACHE[cache_key]
        if ok:
            return value
        raise RuntimeError(str(value))

    errors: list[str] = []

    attempts: list[dict[str, Any]] = [
        {
            "label": f"{model_id}",
            "model": model_id,
            "kwargs": {"api_key": api_key, "base_url": base_url},
        },
        {
            "label": f"{normalized_alias}",
            "model": normalized_alias,
            "kwargs": {"api_key": api_key, "base_url": base_url},
        },
        {
            "label": f"openai:{normalized_alias}",
            "model": normalized_alias,
            "kwargs": {
                "api_key": api_key,
                "base_url": base_url,
                "model_provider": "openai",
            },
        },
    ]
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for attempt in attempts:
        attempt_key = (
            str(attempt["model"]),
            tuple(sorted((k, str(v)) for k, v in dict(attempt["kwargs"]).items())),
        )
        if attempt_key in seen:
            continue
        seen.add(attempt_key)
        try:
            model = init_chat_model(attempt["model"], **attempt["kwargs"])
            _MODEL_INIT_CACHE[cache_key] = (True, model)
            return model
        except Exception as exc:  # pragma: no cover
            category = _classify_model_error(exc)
            errors.append(f"{attempt['label']} [{category}]: {type(exc).__name__}: {exc}")

    msg = (
        "Unable to initialize configured model. Verify provider packages and model ids.\n"
        + "\n".join(errors)
    )
    _MODEL_INIT_CACHE[cache_key] = (False, msg)
    raise RuntimeError(msg)


def create_configured_agent(**kwargs: Any) -> Any:
    """Create an rlmagents agent with DeepSeek (main) + MiniMax (sub-query)."""

    _load_dotenv_if_available()

    main_model_name = _parse_env_value("DEEPSEEK_MODEL_ID", DEFAULT_DEEPSEEK_MODEL)
    main_model_base_url = _parse_env_value("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    sub_query_model_name = _parse_env_value("MINIMAX_MODEL_ID", DEFAULT_MINIMAX_MODEL)
    sub_query_base_url = _parse_env_value("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL)

    main_model = _init_model(
        model_id=main_model_name,
        model_alias=main_model_name,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=main_model_base_url,
    )

    sub_query_model = _init_model(
        model_id=sub_query_model_name,
        model_alias=sub_query_model_name,
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url=sub_query_base_url,
    )

    return create_rlm_agent(
        model=main_model,
        sub_query_model=sub_query_model,
        sub_query_timeout=120.0,
        sandbox_timeout=300.0,
        **kwargs,
    )
