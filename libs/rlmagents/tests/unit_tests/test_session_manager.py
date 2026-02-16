"""Tests for RLMSessionManager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rlmagents.session_manager import RLMSessionManager, _detect_format

from aleph.types import ContentFormat


class TestDetectFormat:
    def test_auto_text(self):
        assert _detect_format("hello world") == ContentFormat.TEXT

    def test_auto_json(self):
        assert _detect_format('{"key": "value"}') == ContentFormat.JSON

    def test_explicit_hint(self):
        assert _detect_format("anything", "code") == ContentFormat.CODE

    def test_invalid_hint_falls_back(self):
        assert _detect_format("text", "invalid_format") == ContentFormat.TEXT


class TestRLMSessionManager:
    def test_create_session(self):
        mgr = RLMSessionManager()
        meta = mgr.create_session("Hello, World!", context_id="test")
        assert meta.size_chars == 13
        assert meta.size_lines == 1
        assert "test" in mgr.sessions

    def test_get_or_create_session(self):
        mgr = RLMSessionManager()
        session = mgr.get_or_create_session("new")
        assert session is not None
        assert "new" in mgr.sessions
        # Getting again returns same session
        same = mgr.get_or_create_session("new")
        assert same is session

    def test_get_session_nonexistent(self):
        mgr = RLMSessionManager()
        assert mgr.get_session("nope") is None

    def test_delete_session(self):
        mgr = RLMSessionManager()
        mgr.create_session("data", context_id="del")
        assert mgr.delete_session("del") is True
        assert mgr.delete_session("del") is False

    def test_list_sessions(self):
        mgr = RLMSessionManager()
        mgr.create_session("abc", context_id="a")
        mgr.create_session("xyz", context_id="b")
        listing = mgr.list_sessions()
        assert "a" in listing
        assert "b" in listing
        assert listing["a"]["size_chars"] == 3

    def test_session_repl_access(self):
        mgr = RLMSessionManager()
        mgr.create_session("test data here", context_id="ctx1")
        session = mgr.get_session("ctx1")
        assert session is not None
        ctx_val = session.repl.get_variable("ctx")
        assert ctx_val == "test data here"

    def test_session_repl_execute(self):
        mgr = RLMSessionManager()
        mgr.create_session("hello world foo bar", context_id="exec")
        session = mgr.get_session("exec")
        assert session is not None
        result = session.repl.execute("len(ctx)")
        assert result.error is None
        assert result.return_value == 19

    def test_session_helpers_available(self):
        mgr = RLMSessionManager()
        mgr.create_session("line one\nline two\nline three", context_id="h")
        session = mgr.get_session("h")
        assert session is not None
        search_fn = session.repl.get_helper("search")
        assert search_fn is not None
        results = search_fn("two")
        assert len(results) >= 1

    def test_save_and_load_session(self):
        mgr = RLMSessionManager()
        mgr.create_session("persistent data", context_id="save_test")
        session = mgr.get_session("save_test")
        assert session is not None
        session.iterations = 5
        session.think_history.append("step 1")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr.save_session("save_test", path=path)
            # Verify file was written
            payload = json.loads(Path(path).read_text())
            assert payload["iterations"] == 5
            assert "step 1" in payload["think_history"]

            # Load into new manager
            mgr2 = RLMSessionManager()
            meta = mgr2.load_session_from_file(path, context_id="loaded")
            assert meta.size_chars == len("persistent data")
            loaded = mgr2.get_session("loaded")
            assert loaded is not None
            assert loaded.iterations == 5
            assert loaded.think_history == ["step 1"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_update_config(self):
        mgr = RLMSessionManager()
        result = mgr.update_config(sandbox_timeout=60.0, context_policy="isolated")
        assert result["sandbox_timeout"] == 60.0
        assert result["context_policy"] == "isolated"
        assert mgr.context_policy == "isolated"

    def test_create_session_replaces_existing(self):
        mgr = RLMSessionManager()
        mgr.create_session("first", context_id="r")
        mgr.create_session("second", context_id="r")
        session = mgr.get_session("r")
        assert session is not None
        assert session.repl.get_variable("ctx") == "second"
