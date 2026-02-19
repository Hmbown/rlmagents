"""Unit tests for Harbor runtime compatibility fixes."""

from __future__ import annotations

import base64
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.protocol import EditResult, ExecuteResponse, WriteResult
from deepagents_harbor.backend import HarborSandbox
from deepagents_harbor.deepagents_wrapper import DeepAgentsWrapper


@dataclass
class _ExecResult:
    """Simple environment exec payload."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class _DummyEnvironment:
    """Minimal Harbor-like environment for backend tests."""

    def __init__(self) -> None:
        self.session_id = "sess-123"

    async def exec(self, command: str) -> _ExecResult:
        return _ExecResult(stdout=f"ran: {command}", stderr="", return_code=0)


class _DelegationSandbox(HarborSandbox):
    """Sandbox that exposes async method calls for sync wrapper tests."""

    def __init__(self) -> None:
        super().__init__(_DummyEnvironment())
        self.calls: dict[str, object] = {}

    async def aexecute(self, command: str) -> ExecuteResponse:
        self.calls["execute"] = command
        return ExecuteResponse(output="ok", exit_code=0)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        self.calls["read"] = (file_path, offset, limit)
        return "read-ok"

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.calls["write"] = (file_path, content)
        return WriteResult(path=file_path, files_update=None)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        self.calls["edit"] = (file_path, old_string, new_string, replace_all)
        return EditResult(path=file_path, files_update=None, occurrences=1)

    async def als_info(self, path: str) -> list[dict[str, object]]:
        self.calls["ls_info"] = path
        return [{"path": "/app", "is_dir": True}]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[dict[str, object]] | str:
        self.calls["grep_raw"] = (pattern, path, glob)
        return [{"path": "/app/a.txt", "line": 1, "text": "hit"}]

    async def aglob_info(self, pattern: str, path: str = "/") -> list[dict[str, object]]:
        self.calls["glob_info"] = (pattern, path)
        return [{"path": "a.txt", "is_dir": False}]


class _CommandSandbox(HarborSandbox):
    """Sandbox that returns pre-canned execute responses."""

    def __init__(self, responses: list[ExecuteResponse]) -> None:
        super().__init__(_DummyEnvironment())
        self.responses = responses
        self.commands: list[str] = []

    def execute(self, command: str) -> ExecuteResponse:
        self.commands.append(command)
        return self.responses.pop(0)


class _ConvertedInput:
    """Container that mimics model input conversion."""

    def __init__(self, messages: list[object]) -> None:
        self._messages = messages

    def to_messages(self) -> list[object]:
        return self._messages


class _FakeDeepSeekModel:
    """Minimal model object for payload patch tests."""

    def __init__(self, payload: dict[str, object], messages: list[object]) -> None:
        self._payload = payload
        self._messages = messages

    def _get_request_payload(
        self,
        input_: object,
        *,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        del input_, stop, kwargs
        return self._payload

    def _convert_input(self, input_: object) -> _ConvertedInput:
        del input_
        return _ConvertedInput(self._messages)


def test_sync_wrappers_delegate_to_async_methods() -> None:
    """Sync protocol methods should bridge to async implementations."""
    sandbox = _DelegationSandbox()

    execute_res = sandbox.execute("echo hi")
    assert execute_res.output == "ok"
    assert sandbox.calls["execute"] == "echo hi"

    read_res = sandbox.read("/app/a.txt", offset=10, limit=20)
    assert read_res == "read-ok"
    assert sandbox.calls["read"] == ("/app/a.txt", 10, 20)

    write_res = sandbox.write("/app/new.txt", "data")
    assert write_res.path == "/app/new.txt"
    assert sandbox.calls["write"] == ("/app/new.txt", "data")

    edit_res = sandbox.edit("/app/new.txt", "a", "b", replace_all=True)
    assert edit_res.path == "/app/new.txt"
    assert sandbox.calls["edit"] == ("/app/new.txt", "a", "b", True)

    ls_res = sandbox.ls_info("/app")
    assert ls_res == [{"path": "/app", "is_dir": True}]
    assert sandbox.calls["ls_info"] == "/app"

    grep_res = sandbox.grep_raw("hit", path="/app", glob="*.txt")
    assert grep_res == [{"path": "/app/a.txt", "line": 1, "text": "hit"}]
    assert sandbox.calls["grep_raw"] == ("hit", "/app", "*.txt")

    glob_res = sandbox.glob_info("*.txt", path="/app")
    assert glob_res == [{"path": "a.txt", "is_dir": False}]
    assert sandbox.calls["glob_info"] == ("*.txt", "/app")


def test_upload_files_handles_success_and_errors() -> None:
    """upload_files should map command outcomes into structured errors."""
    sandbox = _CommandSandbox(
        [
            ExecuteResponse(output="", exit_code=0),
            ExecuteResponse(output="permission_denied", exit_code=2),
        ]
    )

    responses = sandbox.upload_files(
        [
            ("/app/ok.bin", b"abc"),
            ("/app/denied.bin", b"def"),
            ("relative.bin", b"x"),
        ]
    )

    assert responses[0].path == "/app/ok.bin"
    assert responses[0].error is None
    assert responses[1].path == "/app/denied.bin"
    assert responses[1].error == "permission_denied"
    assert responses[2].path == "relative.bin"
    assert responses[2].error == "invalid_path"
    assert len(sandbox.commands) == 2


def test_download_files_decodes_bytes_and_maps_errors() -> None:
    """download_files should decode payloads and map standard error codes."""
    encoded = base64.b64encode(b"payload").decode("ascii")
    sandbox = _CommandSandbox(
        [
            ExecuteResponse(output=encoded, exit_code=0),
            ExecuteResponse(output="is_directory", exit_code=2),
            ExecuteResponse(output="permission_denied", exit_code=4),
            ExecuteResponse(output="file_not_found", exit_code=1),
        ]
    )

    responses = sandbox.download_files(
        [
            "/app/ok.bin",
            "/app/dir",
            "/app/private",
            "/app/missing",
            "relative.bin",
        ]
    )

    assert responses[0].path == "/app/ok.bin"
    assert responses[0].content == b"payload"
    assert responses[0].error is None

    assert responses[1].path == "/app/dir"
    assert responses[1].error == "is_directory"

    assert responses[2].path == "/app/private"
    assert responses[2].error == "permission_denied"

    assert responses[3].path == "/app/missing"
    assert responses[3].error == "file_not_found"

    assert responses[4].path == "relative.bin"
    assert responses[4].error == "invalid_path"

    # relative path is rejected before command execution
    assert len(sandbox.commands) == 4


def test_model_name_deepseek_reasoner_detection() -> None:
    """Reasoner detection should only trigger for DeepSeek reasoner models."""
    assert DeepAgentsWrapper._model_is_deepseek_reasoner("deepseek:deepseek-reasoner")
    assert DeepAgentsWrapper._model_is_deepseek_reasoner("openai/deepseek-reasoner")
    assert not DeepAgentsWrapper._model_is_deepseek_reasoner("deepseek:deepseek-chat")
    assert not DeepAgentsWrapper._model_is_deepseek_reasoner("openai:gpt-5")
    assert not DeepAgentsWrapper._model_is_deepseek_reasoner(None)


def test_deepseek_reasoner_patch_preserves_reasoning_content() -> None:
    """Patch should carry reasoning_content for assistant tool-call messages."""
    payload = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "t1"}], "content": ""},
            {"role": "assistant", "content": "plain"},
        ]
    }
    messages = [
        AIMessage(content="", additional_kwargs={"reasoning_content": "chain-of-thought-summary"}),
        HumanMessage(content="ignored"),
    ]
    model = _FakeDeepSeekModel(payload=payload, messages=messages)

    DeepAgentsWrapper._patch_deepseek_reasoner_payload(model)
    result = model._get_request_payload(input_={})

    payload_messages = result["messages"]
    assert isinstance(payload_messages, list)
    assert payload_messages[0]["reasoning_content"] == "chain-of-thought-summary"
    assert "reasoning_content" not in payload_messages[1]

    # Patch should be idempotent.
    DeepAgentsWrapper._patch_deepseek_reasoner_payload(model)
    result2 = model._get_request_payload(input_={})
    payload_messages2 = result2["messages"]
    assert isinstance(payload_messages2, list)
    assert payload_messages2[0]["reasoning_content"] == "chain-of-thought-summary"
