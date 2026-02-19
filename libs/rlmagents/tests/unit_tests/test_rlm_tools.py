"""Tests for RLM LangChain tools."""

from __future__ import annotations

import gzip
import subprocess

from rlmagents.middleware._tools import _build_rlm_tools
from rlmagents.session_manager import RLMSessionManager


class TestToolBuilding:
    def test_build_all_tools_in_full_profile(self):
        mgr = RLMSessionManager()
        tools = _build_rlm_tools(mgr)
        assert len(tools) == 26

    def test_tool_names_unique(self):
        mgr = RLMSessionManager()
        tools = _build_rlm_tools(mgr)
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self):
        mgr = RLMSessionManager()
        tools = _build_rlm_tools(mgr)
        names = {t.name for t in tools}
        expected = {
            "load_context",
            "load_file_context",
            "list_contexts",
            "diff_contexts",
            "save_session",
            "load_session",
            "peek_context",
            "search_context",
            "semantic_search",
            "chunk_context",
            "cross_context_search",
            "rg_search",
            "exec_python",
            "get_variable",
            "think",
            "evaluate_progress",
            "summarize_so_far",
            "get_evidence",
            "finalize",
            "get_status",
            "rlm_tasks",
            "validate_recipe",
            "estimate_recipe",
            "run_recipe",
            "run_recipe_code",
            "configure_rlm",
        }
        assert names == expected

    def test_core_profile_has_reduced_toolset(self):
        mgr = RLMSessionManager()
        tools = _build_rlm_tools(mgr, profile="core")
        names = {t.name for t in tools}
        assert "run_recipe" not in names
        assert "configure_rlm" not in names
        assert "load_file_context" in names
        assert "chunk_context" in names
        assert "rg_search" not in names

    def test_profile_include_and_exclude(self):
        mgr = RLMSessionManager()
        tools = _build_rlm_tools(
            mgr,
            profile="core",
            include_tools=("run_recipe", "rg_search"),
            exclude_tools=("semantic_search",),
        )
        names = {t.name for t in tools}
        assert "run_recipe" in names
        assert "semantic_search" not in names
        assert "rg_search" in names


class TestLoadContextTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "load_context")

    def test_load_context(self):
        mgr = RLMSessionManager()
        tool = self._get_tool(mgr)
        result = tool.invoke({"content": "hello world", "context_id": "t1"})
        assert "t1" in result
        assert "loaded" in result.lower()
        assert "t1" in mgr.sessions


class TestLoadFileContextTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "load_file_context")

    def test_load_file_context(self, tmp_path):
        path = tmp_path / "context.txt"
        path.write_text("hello from file")
        mgr = RLMSessionManager()
        tool = self._get_tool(mgr)
        result = tool.invoke({"path": str(path), "context_id": "file_ctx"})
        assert "file_ctx" in result
        session = mgr.get_session("file_ctx")
        assert session is not None
        assert session.repl.get_variable("ctx") == "hello from file"

    def test_load_file_context_gzip(self, tmp_path):
        path = tmp_path / "context.txt.gz"
        path.write_bytes(gzip.compress(b"hello from compressed file"))
        mgr = RLMSessionManager()
        tool = self._get_tool(mgr)
        result = tool.invoke({"path": str(path), "context_id": "compressed"})
        assert "decompressed:gzip" in result
        session = mgr.get_session("compressed")
        assert session is not None
        assert session.repl.get_variable("ctx") == "hello from compressed file"


class TestChunkContextTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "chunk_context")

    def test_chunk_context_creates_chunk_metadata(self):
        mgr = RLMSessionManager()
        mgr.create_session("abcdefghijklmnopqrstuvwxyz", context_id="chunks")
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "chunks", "chunk_size": 10, "overlap": 2})
        assert "Created" in result
        session = mgr.get_session("chunks")
        assert session is not None
        assert session.chunks is not None
        assert len(session.chunks) >= 3


class TestRgSearchTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "rg_search")

    def test_rg_search_loads_results_into_context(self, monkeypatch):
        mgr = RLMSessionManager()
        tool = self._get_tool(mgr)

        monkeypatch.setattr("rlmagents.middleware._tools.shutil.which", lambda _: "/usr/bin/rg")

        def _fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=["rg"],
                returncode=0,
                stdout="a.py:1:alpha\nb.py:2:alpha",
                stderr="",
            )

        monkeypatch.setattr("rlmagents.middleware._tools.subprocess.run", _fake_run)
        result = tool.invoke(
            {
                "pattern": "alpha",
                "paths": ".",
                "load_context_id": "rg_hits",
            }
        )
        assert "rg_hits" in result
        session = mgr.get_session("rg_hits")
        assert session is not None
        assert "a.py:1:alpha" in session.repl.get_variable("ctx")


class TestSearchContextTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "search_context")

    def test_search_with_matches(self):
        mgr = RLMSessionManager()
        mgr.create_session("line 1 foo\nline 2 bar\nline 3 foo", context_id="s")
        tool = self._get_tool(mgr)
        result = tool.invoke({"pattern": "foo", "context_id": "s"})
        assert "foo" in result

    def test_search_no_matches(self):
        mgr = RLMSessionManager()
        mgr.create_session("hello world", context_id="s")
        tool = self._get_tool(mgr)
        result = tool.invoke({"pattern": "zzz_nonexistent", "context_id": "s"})
        assert "No matches" in result


class TestSemanticSearchTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "semantic_search")

    def test_semantic_search_returns_chunk_preview(self):
        mgr = RLMSessionManager()
        mgr.create_session(
            "alpha beta gamma\nnetwork timeout happened during upload",
            context_id="sem",
        )
        tool = self._get_tool(mgr)
        result = tool.invoke(
            {
                "query": "network timeout",
                "context_id": "sem",
                "chunk_size": 500,
                "top_k": 1,
            }
        )
        assert "score=" in result
        assert "network timeout" in result.lower()


class TestExecPythonTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "exec_python")

    def test_exec_basic(self):
        mgr = RLMSessionManager()
        mgr.create_session("test content", context_id="e")
        tool = self._get_tool(mgr)
        result = tool.invoke({"code": "len(ctx)", "context_id": "e"})
        assert "12" in result

    def test_exec_with_helpers(self):
        mgr = RLMSessionManager()
        mgr.create_session("apple banana cherry", context_id="e")
        tool = self._get_tool(mgr)
        result = tool.invoke({"code": "word_count()", "context_id": "e"})
        assert "3" in result

    def test_exec_error(self):
        mgr = RLMSessionManager()
        mgr.create_session("", context_id="e")
        tool = self._get_tool(mgr)
        result = tool.invoke({"code": "1/0", "context_id": "e"})
        assert "ERROR" in result or "ZeroDivision" in result


class TestPeekContextTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "peek_context")

    def test_peek_chars(self):
        mgr = RLMSessionManager()
        mgr.create_session("abcdefghij", context_id="p")
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "p", "start": 0, "end": 5})
        assert "abcde" in result

    def test_peek_lines(self):
        mgr = RLMSessionManager()
        mgr.create_session("line1\nline2\nline3", context_id="p")
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "p", "start": 0, "end": 2, "unit": "lines"})
        assert "line1" in result
        assert "line2" in result

    def test_peek_empty(self):
        mgr = RLMSessionManager()
        mgr.create_session("", context_id="p")
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "p"})
        assert "empty" in result.lower()


class TestThinkTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "think")

    def test_think_records_history(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="th")
        tool = self._get_tool(mgr)
        result = tool.invoke({"question": "What patterns exist?", "context_id": "th"})
        assert "Think Step" in result
        session = mgr.get_session("th")
        assert session is not None
        assert len(session.think_history) == 1


class TestEvidenceTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "get_evidence")

    def test_no_evidence(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="ev")
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "ev"})
        assert "No evidence" in result

    def test_evidence_after_search(self):
        mgr = RLMSessionManager()
        mgr.create_session("foo bar baz", context_id="ev")
        # Do a search to generate evidence
        search_tool = next(t for t in _build_rlm_tools(mgr) if t.name == "search_context")
        search_tool.invoke({"pattern": "bar", "context_id": "ev"})
        # Now check evidence
        tool = self._get_tool(mgr)
        result = tool.invoke({"context_id": "ev"})
        assert "search" in result.lower()


class TestFinalizeTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "finalize")

    def test_finalize(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="f")
        tool = self._get_tool(mgr)
        result = tool.invoke(
            {
                "answer": "The answer is 42",
                "confidence": "high",
                "context_id": "f",
            }
        )
        assert "42" in result
        assert "high" in result.lower()


class TestListContextsTool:
    def test_empty(self):
        mgr = RLMSessionManager()
        tool = next(t for t in _build_rlm_tools(mgr) if t.name == "list_contexts")
        result = tool.invoke({})
        assert "No active" in result

    def test_with_contexts(self):
        mgr = RLMSessionManager()
        mgr.create_session("data1", context_id="a")
        mgr.create_session("data2", context_id="b")
        tool = next(t for t in _build_rlm_tools(mgr) if t.name == "list_contexts")
        result = tool.invoke({})
        assert "a" in result
        assert "b" in result


class TestTasksTool:
    def _get_tool(self, mgr):
        tools = _build_rlm_tools(mgr)
        return next(t for t in tools if t.name == "rlm_tasks")

    def test_add_and_list(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="t")
        tool = self._get_tool(mgr)
        tool.invoke({"action": "add", "context_id": "t", "description": "Do something"})
        result = tool.invoke({"action": "list", "context_id": "t"})
        assert "Do something" in result

    def test_update_task(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="t")
        tool = self._get_tool(mgr)
        tool.invoke({"action": "add", "context_id": "t", "description": "Task 1"})
        result = tool.invoke(
            {"action": "update", "context_id": "t", "task_id": "1", "status": "done"}
        )
        assert "done" in result

    def test_clear_tasks(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="t")
        tool = self._get_tool(mgr)
        tool.invoke({"action": "add", "context_id": "t", "description": "Task 1"})
        result = tool.invoke({"action": "clear", "context_id": "t"})
        assert "Cleared" in result


class TestConfigureTool:
    def test_configure(self):
        mgr = RLMSessionManager()
        tool = next(t for t in _build_rlm_tools(mgr) if t.name == "configure_rlm")
        result = tool.invoke({"sandbox_timeout": 60.0})
        assert "60" in result
