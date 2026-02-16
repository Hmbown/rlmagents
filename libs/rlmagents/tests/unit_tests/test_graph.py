from unittest.mock import Mock

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
