"""rlmagents -- RLM-enhanced deep agents.

Brings Aleph's full RLM capabilities (context isolation, evidence tracking,
sandboxed REPL, recipe pipelines, semantic search) into DeepAgents' middleware stack.
"""

from rlmagents.graph import create_rlm_agent
from rlmagents.middleware import RLMMiddleware
from rlmagents.session_manager import RLMSessionManager

__all__ = ["create_rlm_agent", "RLMMiddleware", "RLMSessionManager"]
