"""Terminal-bench style scenario coverage for rlmagents."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples import bootstrap_config, dogfood as dogfood_example
from rlmagents.graph import create_rlm_agent
from rlmagents.middleware._tools import _build_rlm_tools
from rlmagents.session_manager import RLMSessionManager


class _QueuedChatModel(BaseChatModel):
    """Chat model that returns a predefined sequence of responses."""

    def __init__(self, responses: list[AIMessage | str]) -> None:
        super().__init__()
        self._responses = iter(responses)

    def bind_tools(self, tools: object, **kwargs: object) -> BaseChatModel:
        _ = tools
        _ = kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = messages
        _ = stop
        _ = run_manager
        _ = kwargs
        response = next(self._responses)
        if isinstance(response, str):
            response = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "queued-fake-chat-model"


class _AsyncSubQueryModel:
    """Async-only model used for sub_query stubbing."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str, **kwargs: object) -> AIMessage:
        del kwargs
        self.calls.append(prompt)
        return AIMessage(content=self.output)


class _FakeAgent:
    """Agent stub returning a fixed tool-output-like payload."""

    def __init__(self) -> None:
        self.invocations: list[dict[str, object]] = []

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        self.invocations.append(payload)
        return {"messages": [AIMessage(content="DOGFOOD_OK")]}


def _record_benchmark_score(scenario: str) -> None:
    score_path = os.getenv("RLMAGENTS_BENCHMARK_SCORE_PATH")
    if not score_path:
        return

    path = Path(score_path)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}

    data[scenario] = "passed"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


class TestTerminalBenchScenarios:
    def test_terminal_bench_read_edit_verify_loop(self) -> None:
        model = _QueuedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "id": "w1",
                            "type": "tool_call",
                            "args": {
                                "file_path": "/scenario/file.txt",
                                "content": "original=1\n",
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "id": "r1",
                            "type": "tool_call",
                            "args": {"file_path": "/scenario/file.txt"},
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "id": "e1",
                            "type": "tool_call",
                            "args": {
                                "file_path": "/scenario/file.txt",
                                "old_string": "original=1",
                                "new_string": "patched=1",
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "id": "r2",
                            "type": "tool_call",
                            "args": {"file_path": "/scenario/file.txt"},
                        }
                    ],
                ),
                AIMessage(content="Read/edit/verify loop complete"),
            ]
        )

        agent = create_rlm_agent(model=model)
        result = agent.invoke({"messages": [HumanMessage(content="apply the edit")]})

        tool_names = [msg.name for msg in result["messages"] if msg.type == "tool"]
        assert tool_names == [
            "write_file",
            "read_file",
            "edit_file",
            "read_file",
        ]
        assert "patched=1" in result["files"]["/scenario/file.txt"]["content"][0]
        assert result["messages"][-1].content == "Read/edit/verify loop complete"
        _record_benchmark_score("read_edit_verify_loop")

    def test_terminal_bench_long_context_compaction(self) -> None:
        manager = RLMSessionManager()
        manager.create_session("Long-context harness content", context_id="bench")
        tools = {tool.name: tool for tool in _build_rlm_tools(manager)}

        for index in range(24):
            tools["think"].invoke(
                {
                    "question": f"Think step {index} about compaction",
                    "context_id": "bench",
                }
            )

        summary = tools["summarize_so_far"].invoke(
            {
                "context_id": "bench",
                "include_evidence": False,
                "include_variables": False,
                "clear_history": True,
            }
        )

        assert "Think Steps (10)" in summary
        assert "(think_history cleared)" in summary
        session = manager.get_session("bench")
        assert session is not None
        assert session.think_history == []
        _record_benchmark_score("long_context_compaction")

    @pytest.mark.asyncio
    async def test_terminal_bench_sub_query_stubbed_path(self) -> None:
        model = _AsyncSubQueryModel("subquery-result")
        manager = RLMSessionManager(
            sub_query_model=model,
            rlm_max_recursion_depth=0,
        )
        manager.create_session("line 1\nline 2", context_id="query")
        session = manager.get_session("query")

        loop = asyncio.new_event_loop()
        started = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            started.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        started.wait(timeout=5)
        session.repl.set_loop(loop)

        try:
            result = await session.repl.execute_async(
                "print(sub_query('Summarize harness context', context_slice='line 1\\nline 2'))"
            )
            assert "subquery-result" in result.stdout
            assert model.calls
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()

        _record_benchmark_score("sub_query_stubbed_path")

    def test_terminal_bench_dogfood_mocked_end_to_end(self, monkeypatch, capsys) -> None:
        marker: list[str] = []

        def _fake_tooled() -> None:
            marker.append("tooled")

        fake_agent = _FakeAgent()
        monkeypatch.setattr(dogfood_example, "run_tooled_dogfood", _fake_tooled)
        monkeypatch.setattr(
            bootstrap_config,
            "create_configured_agent",
            lambda **kwargs: fake_agent,  # type: ignore[return-value]
        )
        monkeypatch.setattr(bootstrap_config, "_load_dotenv_if_available", lambda: None)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
        monkeypatch.setenv("MINIMAX_API_KEY", "fake-minimax")

        dogfood_example.main()
        captured = capsys.readouterr()

        assert marker == ["tooled"]
        assert fake_agent.invocations, "Expected mocked agent to be invoked"
        assert "DOGFOOD_OK" in captured.out
        _record_benchmark_score("dogfood_mocked_provider")
