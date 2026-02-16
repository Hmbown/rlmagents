"""Tests for RLMMiddleware."""

from __future__ import annotations

from unittest.mock import MagicMock

from rlmagents.middleware.rlm import RLMMiddleware, _RLM_TOOL_NAMES


class TestRLMMiddleware:
    def test_init(self):
        mw = RLMMiddleware()
        assert len(mw.tools) == 22
        assert mw.manager is not None

    def test_custom_config(self):
        mw = RLMMiddleware(
            sandbox_timeout=60.0,
            context_policy="isolated",
            auto_load_threshold=5000,
        )
        assert mw._manager.sandbox_config.timeout_seconds == 60.0
        assert mw._manager.context_policy == "isolated"
        assert mw._auto_load_threshold == 5000

    def test_tool_names_match(self):
        mw = RLMMiddleware()
        names = {t.name for t in mw.tools}
        assert names == _RLM_TOOL_NAMES

    def test_custom_prompt(self):
        mw = RLMMiddleware(system_prompt="Custom RLM prompt")
        assert mw._get_prompt() == "Custom RLM prompt"

    def test_default_prompt_loads(self):
        mw = RLMMiddleware()
        prompt = mw._get_prompt()
        assert "RLM" in prompt
        assert "load_context" in prompt or "context" in prompt.lower()

    def test_state_schema(self):
        assert RLMMiddleware.state_schema is not None
        # Check that it has the rlm_context_ids field
        hints = RLMMiddleware.state_schema.__annotations__
        assert "rlm_context_ids" in hints


class TestAutoLoad:
    def test_auto_load_large_content(self):
        mw = RLMMiddleware(auto_load_threshold=100)
        # Simulate a large ToolMessage
        from langchain_core.messages import ToolMessage
        large_content = "x" * 200
        msg = ToolMessage(content=large_content, tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert isinstance(result, ToolMessage)
        assert "auto_read_file" in result.content
        assert "auto_read_file" in mw._manager.sessions

    def test_auto_load_skips_small_content(self):
        mw = RLMMiddleware(auto_load_threshold=1000)
        from langchain_core.messages import ToolMessage
        msg = ToolMessage(content="small", tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert result is msg  # unchanged

    def test_auto_load_skips_rlm_tools(self):
        mw = RLMMiddleware(auto_load_threshold=10)
        from langchain_core.messages import ToolMessage
        msg = ToolMessage(content="x" * 100, tool_call_id="tc1", name="search_context")
        result = mw._maybe_auto_load(msg, "search_context")
        assert result is msg  # unchanged (RLM tool)

    def test_auto_load_disabled(self):
        mw = RLMMiddleware(auto_load_threshold=0)
        from langchain_core.messages import ToolMessage
        msg = ToolMessage(content="x" * 1000, tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert result is msg  # unchanged
