"""Tests for create_rlm_agent factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rlmagents.middleware.rlm import RLMMiddleware


class TestCreateRLMAgent:
    @patch("deepagents.graph.create_deep_agent")
    def test_creates_agent_with_rlm_middleware(self, mock_create):
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        create_rlm_agent(model="test-model")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware") or call_kwargs[1].get("middleware")
        assert any(isinstance(m, RLMMiddleware) for m in middleware)

    @patch("deepagents.graph.create_deep_agent")
    def test_rlm_middleware_is_first(self, mock_create):
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        create_rlm_agent()

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware") or call_kwargs[1].get("middleware")
        assert isinstance(middleware[0], RLMMiddleware)

    @patch("deepagents.graph.create_deep_agent")
    def test_custom_sandbox_timeout(self, mock_create):
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        create_rlm_agent(sandbox_timeout=60.0)

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware") or call_kwargs[1].get("middleware")
        rlm_mw = middleware[0]
        assert isinstance(rlm_mw, RLMMiddleware)
        assert rlm_mw._manager.sandbox_config.timeout_seconds == 60.0

    @patch("deepagents.graph.create_deep_agent")
    def test_passes_through_kwargs(self, mock_create):
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        create_rlm_agent(
            model="test",
            system_prompt="Hello",
            debug=True,
        )

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["system_prompt"] == "Hello"
        assert call_kwargs.kwargs["debug"] is True

    @patch("deepagents.graph.create_deep_agent")
    def test_additional_middleware_appended(self, mock_create):
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        extra_mw = MagicMock()
        create_rlm_agent(middleware=[extra_mw])

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware") or call_kwargs[1].get("middleware")
        assert isinstance(middleware[0], RLMMiddleware)
        assert middleware[1] is extra_mw
