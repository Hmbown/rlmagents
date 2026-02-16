"""Dogfood rlmagents by analyzing its own source using its own tools."""

import os
from pathlib import Path

from rlmagents.session_manager import RLMSessionManager
from rlmagents.middleware._tools import _build_rlm_tools


def run_tooled_dogfood() -> None:
    """Run a small RLM tooling-based self-analysis."""
    sm = RLMSessionManager()

    for name in [
        "graph.py",
        "session_manager.py",
        "middleware/rlm.py",
        "middleware/_tools.py",
        "types.py",
        "repl/sandbox.py",
    ]:
        path = Path("rlmagents") / name
        sm.create_session(path.read_text(), context_id=name)

    print(f"Loaded {len(sm.sessions)} contexts")

    tools = _build_rlm_tools(sm)
    tool_map = {tool.name: tool for tool in tools}

    result = tool_map["search_context"].invoke({
        "pattern": "sub_query",
        "context_id": "session_manager.py",
    })
    print(result)

    result = tool_map["exec_python"].invoke(
        {
            "code": "print(f'Lines: {line_count()}, Words: {word_count()}')",
            "context_id": "graph.py",
        }
    )
    print(result)

    result = tool_map["cross_context_search"].invoke({"pattern": "def create_"})
    print(result)


def run_agent() -> None:
    """Run a full agent loop to analyze rlmagents/graph.py."""
    from examples.bootstrap_config import _load_dotenv_if_available

    _load_dotenv_if_available()

    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("MINIMAX_API_KEY"):
        print("Skipping full-agent dogfood: DEEPSEEK_API_KEY / MINIMAX_API_KEY not set.")
        return

    from examples.bootstrap_config import create_configured_agent

    try:
        agent = create_configured_agent()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Load the file rlmagents/graph.py into an RLM context, "
                            "then analyze it: How many parameters does create_rlm_agent accept? "
                            "What middleware does it configure? Give me a structured summary."
                        ),
                    }
                ]
            }
        )
    except Exception as exc:
        print(f"Skipping full-agent dogfood: {type(exc).__name__}: {exc}")
        return

    for msg in result["messages"]:
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            print(msg.content)


def main() -> None:
    run_tooled_dogfood()
    run_agent()


if __name__ == "__main__":
    main()
