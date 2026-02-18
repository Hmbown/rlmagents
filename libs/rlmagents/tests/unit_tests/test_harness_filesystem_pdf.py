"""Tests for PDF handling in harness filesystem read_file."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain.tools import ToolRuntime

from rlmagents._harness import filesystem as filesystem_module
from rlmagents._harness.backends.protocol import FileDownloadResponse
from rlmagents._harness.filesystem import FilesystemMiddleware, FilesystemState


class _Backend:
    """Simple backend stub for filesystem middleware tests."""

    def __init__(
        self,
        *,
        download_response: list[FileDownloadResponse] | None = None,
        adownload_response: list[FileDownloadResponse] | None = None,
    ) -> None:
        self._download_response = download_response or []
        self.download_files_calls: list[list[str]] = []
        self.adownload_files_calls: list[list[str]] = []

        if adownload_response is None:
            self.adownload_files = AsyncMock(return_value=self._download_response)
        else:
            self.adownload_files = AsyncMock(return_value=adownload_response)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self.download_files_calls.append(paths)
        return self._download_response


def _runtime(tool_call_id: str = "test-pdf") -> ToolRuntime[None, FilesystemState]:
    """Create a minimal runtime payload for tool invocation."""
    return ToolRuntime(
        state=FilesystemState(messages=[]),
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _read_file_tool(middleware: FilesystemMiddleware):
    return next(tool for tool in middleware.tools if tool.name == "read_file")


def test_read_file_pdf_extracts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """read_file should extract PDF bytes to paginated text output."""
    backend = _Backend(
        download_response=[FileDownloadResponse(path="/app/paper.pdf", content=b"%PDF-1.4", error=None)]
    )
    middleware = FilesystemMiddleware(backend=backend)
    tool = _read_file_tool(middleware)

    monkeypatch.setattr(
        filesystem_module,
        "_extract_pdf_text",
        lambda pdf_bytes, file_path: "alpha\nbeta",  # noqa: ARG005
    )

    result = tool.invoke({"file_path": "/app/paper.pdf", "runtime": _runtime()})
    assert isinstance(result, str)
    assert "Extracted text from PDF" in result
    assert "1\talpha" in result
    assert "2\tbeta" in result


def test_read_file_pdf_returns_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_file should surface extraction errors when no text can be produced."""
    backend = _Backend(
        download_response=[FileDownloadResponse(path="/app/paper.pdf", content=b"%PDF-1.4", error=None)]
    )
    middleware = FilesystemMiddleware(backend=backend)
    tool = _read_file_tool(middleware)

    monkeypatch.setattr(
        filesystem_module,
        "_extract_pdf_text",
        lambda pdf_bytes, file_path: f"Error: could not parse {file_path}",  # noqa: ARG005
    )

    result = tool.invoke({"file_path": "/app/paper.pdf", "runtime": _runtime()})
    assert isinstance(result, str)
    assert result.startswith("Error: could not parse /app/paper.pdf")


@pytest.mark.asyncio
async def test_read_file_pdf_async_extracts_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async read_file path should also extract PDF text successfully."""
    backend = _Backend(
        adownload_response=[
            FileDownloadResponse(path="/app/paper.pdf", content=b"%PDF-1.4", error=None)
        ],
    )
    middleware = FilesystemMiddleware(backend=backend)
    tool = _read_file_tool(middleware)

    monkeypatch.setattr(
        filesystem_module,
        "_extract_pdf_text",
        lambda pdf_bytes, file_path: "gamma\ndelta",  # noqa: ARG005
    )

    result = await tool.ainvoke({"file_path": "/app/paper.pdf", "runtime": _runtime("async-pdf")})
    assert isinstance(result, str)
    assert "1\tgamma" in result
    assert "2\tdelta" in result


def test_extract_pdf_text_uses_ocr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary extractors should fall back to OCR when they produce no text."""
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_pypdf", lambda _: None)
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_pdftotext", lambda _: "")
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_ocr", lambda _bytes: "ocr text")

    result = filesystem_module._extract_pdf_text(b"%PDF-1.4", "/app/scan.pdf")
    assert result == "ocr text"


def test_extract_pdf_text_returns_error_when_all_methods_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should return an actionable error when all backends fail."""
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_pypdf", lambda _: None)
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_pdftotext", lambda _: None)
    monkeypatch.setattr(filesystem_module, "_extract_pdf_text_with_ocr", lambda _bytes: None)

    result = filesystem_module._extract_pdf_text(b"%PDF-1.4", "/app/scan.pdf")
    assert result.startswith("Error: Could not extract text from PDF '/app/scan.pdf'")
