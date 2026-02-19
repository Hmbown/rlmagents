from unittest.mock import Mock

import pytest

from rlmagents import graph


def _mocked_defaults(_model):
    return {
        "trigger": "end",
        "keep": 0,
        "trim_tokens_to_summarize": None,
        "truncate_args_settings": None,
    }


def _configure_graph_for_test(monkeypatch):
    monkeypatch.setattr(graph, "_compute_summarization_defaults", _mocked_defaults)
    graph._assert_langchain_compatibility.cache_clear()


def test_create_agent_uses_backend_when_supported(monkeypatch):
    captured = {}

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["backend"] = backend
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(model=Mock())

    assert captured["backend"] is graph.StateBackend


def test_create_agent_falls_back_to_checkpointer_store_when_backend_unsupported(monkeypatch):
    captured = {}
    checkpointer = Mock()
    store = Mock()

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        checkpointer=None,
        store=None,
        debug=False,
        name=None,
        cache=None,
    ):
        captured["checkpointer"] = checkpointer
        captured["store"] = store
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(
        model=Mock(),
        checkpointer=checkpointer,
        store=store,
    )

    assert captured["checkpointer"] is checkpointer
    assert captured["store"] is store


def test_create_agent_rejects_unsupported_langchain_version(monkeypatch):
    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
    ):
        return Mock()

    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "2.0.0")
    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    _configure_graph_for_test(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="rlmagents supports langchain version",
    ):
        graph.create_rlm_agent(model=Mock())


def test_create_agent_rejects_missing_persistence_signature(monkeypatch):
    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
    ):
        return Mock()

    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    _configure_graph_for_test(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="missing required persistence args",
    ):
        graph.create_rlm_agent(model=Mock())


def test_create_agent_defaults_coding_response_format(monkeypatch):
    captured = {}

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["response_format"] = response_format
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(
        model=Mock(),
        system_prompt="Please implement this feature.",
    )

    response = captured["response_format"]
    assert isinstance(response, dict)
    assert response["schema"]["required"] == [
        "plan",
        "edits",
        "verification",
        "risks",
        "confidence",
    ]
    assert set(response["schema"]["properties"]) == {
        "plan",
        "edits",
        "verification",
        "risks",
        "confidence",
    }


def test_create_agent_keeps_explicit_response_format(monkeypatch):
    captured = {}
    explicit_format = {"type": "json_schema", "title": "UserFormat"}

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["response_format"] = response_format
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(
        model=Mock(),
        system_prompt="Please implement this feature.",
        response_format=explicit_format,
    )

    assert captured["response_format"] is explicit_format


def test_create_agent_always_includes_rlm_middleware(monkeypatch):
    captured = {}

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["middleware"] = list(middleware)
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(model=Mock())

    assert any(
        isinstance(layer, graph.RLMMiddleware) for layer in captured["middleware"]
    )


def test_create_agent_ignores_disable_flag_and_keeps_subagent_rlm(monkeypatch):
    captured = {}

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["middleware"] = list(middleware)
        return Mock()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    subagents = [
        {
            "name": "code-reviewer",
            "description": "Review code for regressions.",
            "system_prompt": "Focus on defects and risks.",
            "tools": [],
        }
    ]

    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        graph.create_rlm_agent(
            model=Mock(),
            subagents=subagents,
            enable_rlm_in_subagents=False,
        )

    subagent_middleware = next(
        layer
        for layer in captured["middleware"]
        if isinstance(layer, graph.SubAgentMiddleware)
    )

    configured_subagents = [
        spec for spec in subagent_middleware._subagents if "runnable" not in spec
    ]
    assert configured_subagents
    for spec in configured_subagents:
        middleware_stack = spec.get("middleware", [])
        assert any(
            isinstance(layer, graph.RLMMiddleware) for layer in middleware_stack
        )


def test_create_agent_propagates_sub_query_model_to_subagent_rlm(monkeypatch):
    captured = {}
    sub_query_model = Mock()
    build_calls = []
    original_builder = graph._build_rlm_middleware

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        captured["middleware"] = list(middleware)
        return Mock()

    def recording_builder(**kwargs):
        build_calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(graph, "_build_rlm_middleware", recording_builder)
    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    subagents = [
        {
            "name": "analyst",
            "description": "Analyze context-heavy tasks.",
            "system_prompt": "Use tools carefully.",
            "tools": [],
        }
    ]

    graph.create_rlm_agent(
        model=Mock(),
        subagents=subagents,
        sub_query_model=sub_query_model,
    )

    subagent_middleware = next(
        layer
        for layer in captured["middleware"]
        if isinstance(layer, graph.SubAgentMiddleware)
    )
    configured_subagents = [
        spec for spec in subagent_middleware._subagents if "runnable" not in spec
    ]
    rlm_layers = []
    for spec in configured_subagents:
        rlm_layers.extend(
            [
                layer
                for layer in spec.get("middleware", [])
                if isinstance(layer, graph.RLMMiddleware)
            ]
        )

    assert rlm_layers
    assert build_calls
    for call in build_calls:
        assert call["sub_query_model"] is sub_query_model


def test_create_agent_propagates_tool_profile_configuration(monkeypatch):
    build_calls = []
    original_builder = graph._build_rlm_middleware

    def fake_create_agent(
        model,
        *,
        system_prompt=None,
        tools=None,
        middleware=(),
        response_format=None,
        context_schema=None,
        backend=None,
        debug=False,
        name=None,
        cache=None,
        **_kwargs,
    ):
        return Mock()

    def recording_builder(**kwargs):
        build_calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(graph, "_build_rlm_middleware", recording_builder)
    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "_get_langchain_version", lambda: "1.2.10")
    _configure_graph_for_test(monkeypatch)

    graph.create_rlm_agent(
        model=Mock(),
        rlm_tool_profile="core",
        rlm_include_tools=("run_recipe",),
        rlm_exclude_tools=("semantic_search",),
        auto_load_preview_chars=0,
    )

    assert build_calls
    for call in build_calls:
        assert call["rlm_tool_profile"] == "core"
        assert call["rlm_include_tools"] == ("run_recipe",)
        assert call["rlm_exclude_tools"] == ("semantic_search",)
        assert call["auto_load_preview_chars"] == 0
