"""Tests for RLMMiddleware and core RLM functionality."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from rlmagents.middleware.rlm import _RLM_TOOL_NAMES, RLMMiddleware
from rlmagents.session_manager import RLMSessionManager
from rlmagents.types import Evidence


class TestRLMMiddleware:
    def test_init(self):
        mw = RLMMiddleware()
        assert len(mw.tools) == 23
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

    def test_tool_count_23(self):
        mw = RLMMiddleware()
        assert len(_RLM_TOOL_NAMES) == 23
        assert len(mw.tools) == 23

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
        hints = RLMMiddleware.state_schema.__annotations__
        assert "rlm_context_ids" in hints


class TestAutoLoad:
    def test_auto_load_large_content(self):
        mw = RLMMiddleware(auto_load_threshold=100)
        large_content = "x" * 200
        msg = ToolMessage(content=large_content, tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert isinstance(result, ToolMessage)
        assert "auto_read_file" in result.content
        assert "auto_read_file" in mw._manager.sessions

    def test_auto_load_skips_small_content(self):
        mw = RLMMiddleware(auto_load_threshold=1000)
        msg = ToolMessage(content="small", tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert result is msg

    def test_auto_load_skips_rlm_tools(self):
        mw = RLMMiddleware(auto_load_threshold=10)
        msg = ToolMessage(content="x" * 100, tool_call_id="tc1", name="search_context")
        result = mw._maybe_auto_load(msg, "search_context")
        assert result is msg

    def test_auto_load_disabled(self):
        mw = RLMMiddleware(auto_load_threshold=0)
        msg = ToolMessage(content="x" * 1000, tool_call_id="tc1", name="read_file")
        result = mw._maybe_auto_load(msg, "read_file")
        assert result is msg


class TestSessionManager:
    def test_create_session(self):
        sm = RLMSessionManager()
        meta = sm.create_session("hello world", context_id="test")
        assert meta.size_chars == 11
        assert meta.size_lines == 1
        assert "test" in sm.sessions

    def test_get_or_create(self):
        sm = RLMSessionManager()
        session = sm.get_or_create_session("new")
        assert session is not None
        assert "new" in sm.sessions

    def test_delete_session(self):
        sm = RLMSessionManager()
        sm.create_session("data", context_id="del")
        assert sm.delete_session("del") is True
        assert sm.delete_session("del") is False

    def test_list_sessions(self):
        sm = RLMSessionManager()
        sm.create_session("abc", context_id="a")
        sm.create_session("def", context_id="b")
        listing = sm.list_sessions()
        assert "a" in listing
        assert "b" in listing
        assert listing["a"]["size_chars"] == 3

    def test_sub_query_injection(self):
        sm = RLMSessionManager()
        sm.create_session("data", context_id="sq")
        session = sm.get_session("sq")
        ns = session.repl._namespace
        assert "sub_query" in ns
        assert "llm_query" in ns

    def test_update_config(self):
        sm = RLMSessionManager()
        config = sm.update_config(sandbox_timeout=60.0, context_policy="isolated")
        assert config["sandbox_timeout"] == 60.0
        assert config["context_policy"] == "isolated"


class TestREPLExecution:
    def test_basic_exec(self):
        sm = RLMSessionManager()
        sm.create_session("hello world", context_id="exec")
        session = sm.get_session("exec")
        result = session.repl.execute("print(len(ctx))")
        assert result.stdout.strip() == "11"

    def test_search_helper(self):
        sm = RLMSessionManager()
        sm.create_session(
            "line1\nfoo bar\nline3\nfoo baz\nline5",
            context_id="search",
        )
        session = sm.get_session("search")
        search_fn = session.repl.get_helper("search")
        assert search_fn is not None
        results = search_fn("foo")
        assert len(results) == 2

    def test_peek_helper(self):
        sm = RLMSessionManager()
        sm.create_session("0123456789", context_id="peek")
        session = sm.get_session("peek")
        result = session.repl.execute("print(peek(0, 5))")
        assert result.stdout.strip() == "01234"

    def test_ctx_variable(self):
        sm = RLMSessionManager()
        sm.create_session("test content", context_id="ctx")
        session = sm.get_session("ctx")
        val = session.repl.get_variable("ctx")
        assert val == "test content"

    def test_variable_persistence(self):
        sm = RLMSessionManager()
        sm.create_session("data", context_id="var")
        session = sm.get_session("var")
        session.repl.execute("x = 42")
        val = session.repl.get_variable("x")
        assert val == 42
        session.repl.execute("x = x + 1")
        val = session.repl.get_variable("x")
        assert val == 43


class TestEvidenceTracking:
    def test_add_evidence(self):
        sm = RLMSessionManager()
        sm.create_session("data", context_id="ev")
        session = sm.get_session("ev")
        session.add_evidence(
            Evidence(
                source="search",
                line_range=None,
                pattern="test",
                snippet="test snippet",
            )
        )
        assert len(session.evidence) == 1
        assert session.evidence[0].source == "search"

    def test_evidence_pruning(self):

        sm = RLMSessionManager()
        sm.create_session("data", context_id="prune")
        session = sm.get_session("prune")
        session.max_evidence = 5
        for i in range(10):
            session.add_evidence(
                Evidence(
                    source="search",
                    line_range=None,
                    pattern=f"p{i}",
                    snippet=f"s{i}",
                )
            )
        assert len(session.evidence) <= 5


class TestSerialization:
    def test_save_and_load(self, tmp_path):
        sm = RLMSessionManager()
        sm.create_session("test data for serialization", context_id="ser")
        session = sm.get_session("ser")
        session.add_evidence(
            Evidence(
                source="search",
                line_range=(1, 5),
                pattern="test",
                snippet="test data",
            )
        )

        path = tmp_path / "session.json"
        payload = sm.save_session(context_id="ser", path=str(path))
        assert path.exists()
        assert payload["ctx"] == "test data for serialization"

        # Load into a fresh manager
        sm2 = RLMSessionManager()
        meta = sm2.load_session_from_file(str(path), context_id="loaded")
        assert "loaded" in sm2.sessions
        assert meta.size_chars == len("test data for serialization")
        loaded = sm2.get_session("loaded")
        assert loaded.repl.get_variable("ctx") == "test data for serialization"
        assert len(loaded.evidence) == 1


class TestRLMTools:
    """Test individual RLM tool invocations."""

    def _build_tools(self):
        from rlmagents.middleware._tools import _build_rlm_tools

        sm = RLMSessionManager()
        tools = _build_rlm_tools(sm)
        return sm, {t.name: t for t in tools}

    def test_load_and_search(self):
        sm, tools = self._build_tools()
        result = tools["load_context"].invoke(
            {"content": "alpha\nbeta\ngamma\nalpha again", "context_id": "t1"}
        )
        assert "loaded" in result.lower()

        result = tools["search_context"].invoke({"pattern": "alpha", "context_id": "t1"})
        assert "alpha" in result

    def test_peek_context(self):
        sm, tools = self._build_tools()
        tools["load_context"].invoke({"content": "line0\nline1\nline2", "context_id": "peek"})
        result = tools["peek_context"].invoke(
            {
                "context_id": "peek",
                "start": 0,
                "end": 2,
                "unit": "lines",
            }
        )
        assert "line0" in result
        assert "line1" in result

    def test_exec_python(self):
        sm, tools = self._build_tools()
        tools["load_context"].invoke({"content": "hello world", "context_id": "py"})
        result = tools["exec_python"].invoke({"code": "print(len(ctx))", "context_id": "py"})
        assert "11" in result

    def test_think_and_evaluate(self):
        sm, tools = self._build_tools()
        result = tools["think"].invoke({"question": "What is the meaning?", "context_id": "r"})
        assert "Think Step" in result

        result = tools["evaluate_progress"].invoke(
            {
                "current_understanding": "Found some patterns",
                "confidence_score": 0.7,
                "context_id": "r",
            }
        )
        assert "70%" in result

    def test_finalize(self):
        sm, tools = self._build_tools()
        result = tools["finalize"].invoke(
            {
                "answer": "The answer is 42",
                "confidence": "high",
                "context_id": "f",
            }
        )
        assert "42" in result
        assert "high" in result.lower()

    def test_cross_context_search(self):
        sm, tools = self._build_tools()
        tools["load_context"].invoke({"content": "needle in haystack", "context_id": "c1"})
        tools["load_context"].invoke({"content": "no match here", "context_id": "c2"})
        result = tools["cross_context_search"].invoke({"pattern": "needle"})
        assert "c1" in result
        assert "needle" in result

    def test_get_status(self):
        sm, tools = self._build_tools()
        tools["load_context"].invoke({"content": "data", "context_id": "st"})
        result = tools["get_status"].invoke({"context_id": "st"})
        assert "st" in result
        assert "4" in result  # size_chars

    def test_configure_rlm(self):
        sm, tools = self._build_tools()
        result = tools["configure_rlm"].invoke({"sandbox_timeout": 60.0})
        assert "60" in result
