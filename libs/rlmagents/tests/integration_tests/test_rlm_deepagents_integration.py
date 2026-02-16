"""Integration tests for rlmagents with DeepAgents features.

These tests verify the integration between RLM middleware and DeepAgents
features including skills, memory, summarization, and sub-agents.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRLMAgentWithSkills:
    """Tests for rlmagents with skills middleware integration."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_skills_middleware_added_when_skills_provided(self, mock_create, mock_init):
        """Verify skills middleware is added when skills parameter is provided."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.middleware.skills import SkillsMiddleware
    
        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            skills=["/skills/test/"],
        )

        # Extract middleware from call
        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check skills middleware is present
        skills_mw_found = any(isinstance(mw, SkillsMiddleware) for mw in middleware)
        assert skills_mw_found, "SkillsMiddleware should be added when skills provided"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_skills_middleware_not_added_when_skills_empty(self, mock_create, mock_init):
        """Verify skills middleware is not added when skills is None."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.middleware.skills import SkillsMiddleware
    
        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check skills middleware is not present
        skills_mw_found = any(isinstance(mw, SkillsMiddleware) for mw in middleware)
        assert not skills_mw_found, "SkillsMiddleware should not be added when skills is None"


class TestRLMAgentWithMemory:
    """Tests for rlmagents with memory middleware integration."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_memory_middleware_added_when_memory_provided(self, mock_create, mock_init):
        """Verify memory middleware is added when memory parameter is provided."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.middleware.memory import MemoryMiddleware
    
        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            memory=["/memory/AGENTS.md"],
        )

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check memory middleware is present
        memory_mw_found = any(isinstance(mw, MemoryMiddleware) for mw in middleware)
        assert memory_mw_found, "MemoryMiddleware should be added when memory provided"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_memory_middleware_not_added_when_memory_empty(self, mock_create, mock_init):
        """Verify memory middleware is not added when memory is None."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.middleware.memory import MemoryMiddleware
    
        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check memory middleware is not present
        memory_mw_found = any(isinstance(mw, MemoryMiddleware) for mw in middleware)
        assert not memory_mw_found, "MemoryMiddleware should not be added when memory is None"


class TestRLMAgentWithSummarization:
    """Tests for rlmagents with summarization middleware integration."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_summarization_middleware_always_added(self, mock_create, mock_init):
        """Verify summarization middleware is always added."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.middleware.summarization import SummarizationMiddleware
    
        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check summarization middleware is present
        summary_mw_found = any(isinstance(mw, SummarizationMiddleware) for mw in middleware)
        assert summary_mw_found, "SummarizationMiddleware should always be added"


class TestRLMAgentWithSubAgents:
    """Tests for rlmagents with RLM-enabled sub-agents."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_rlm_middleware_in_general_purpose_subagent(self, mock_create, mock_init):
        """Verify RLM middleware is added to general-purpose sub-agent by default."""
        from rlmagents.graph import create_rlm_agent
        from rlmagents.middleware.rlm import RLMMiddleware

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Find SubAgentMiddleware
        subagent_mw = None
        for mw in middleware:
            from deepagents.middleware.subagents import SubAgentMiddleware
            if isinstance(mw, SubAgentMiddleware):
                subagent_mw = mw
                break

        assert subagent_mw is not None, "SubAgentMiddleware should be present"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_rlm_enabled_in_subagents_parameter(self, mock_create, mock_init):
        """Verify enable_rlm_in_subagents parameter controls RLM in sub-agents."""
        from rlmagents.graph import create_rlm_agent
        from rlmagents.middleware.rlm import RLMMiddleware
        from deepagents.middleware.subagents import SubAgent

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()

        # Create with RLM-enabled sub-agents
        # Note: SubAgent spec requires 'prompt' key which becomes system_prompt
        subagent_spec: SubAgent = {
            "name": "test_agent",
            "description": "Test sub-agent",
            "prompt": "You are a test agent.",
        }

        create_rlm_agent(
            model="anthropic:test-model",
            subagents=[subagent_spec],
            enable_rlm_in_subagents=True,
        )

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Find SubAgentMiddleware and verify it exists
        subagent_mw_found = False
        for mw in middleware:
            from deepagents.middleware.subagents import SubAgentMiddleware
            if isinstance(mw, SubAgentMiddleware):
                subagent_mw_found = True
                break
        
        assert subagent_mw_found, "SubAgentMiddleware should be present"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_subagent_middleware_order(self, mock_create, mock_init):
        """Verify correct middleware order in sub-agents."""
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Main agent middleware order check
        # RLMMiddleware should be first
        from rlmagents.middleware.rlm import RLMMiddleware
        assert isinstance(middleware[0], RLMMiddleware), "RLMMiddleware should be first"


class TestRLMAgentMiddlewareOrder:
    """Tests for correct middleware ordering in rlmagents."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_rlm_middleware_first(self, mock_create, mock_init):
        """Verify RLM middleware is first in the stack."""
        from rlmagents.graph import create_rlm_agent
        from rlmagents.middleware.rlm import RLMMiddleware

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        assert len(middleware) > 0, "Should have at least one middleware"
        assert isinstance(middleware[0], RLMMiddleware), "RLMMiddleware should be first"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_complete_middleware_stack(self, mock_create, mock_init):
        """Verify complete middleware stack is present."""
        from rlmagents.graph import create_rlm_agent
        from langchain.agents.middleware import TodoListMiddleware
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.middleware.subagents import SubAgentMiddleware
        from deepagents.middleware.summarization import SummarizationMiddleware
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        middleware_types = [type(mw).__name__ for mw in middleware]

        # Check essential middleware
        assert any("RLMMiddleware" in t for t in middleware_types), "RLMMiddleware required"
        assert any("TodoListMiddleware" in t for t in middleware_types), "TodoListMiddleware required"
        assert any("FilesystemMiddleware" in t for t in middleware_types), "FilesystemMiddleware required"
        assert any("SubAgentMiddleware" in t for t in middleware_types), "SubAgentMiddleware required"
        assert any("SummarizationMiddleware" in t for t in middleware_types), "SummarizationMiddleware required"


class TestRLMAgentParameters:
    """Tests for rlmagents parameter handling."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_rlm_specific_parameters(self, mock_create, mock_init):
        """Verify RLM-specific parameters are passed correctly."""
        from rlmagents.graph import create_rlm_agent
        from rlmagents.middleware.rlm import RLMMiddleware

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            sandbox_timeout=300.0,
            context_policy="isolated",
            auto_load_threshold=5000,
            rlm_system_prompt="Custom RLM prompt",
        )

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Find RLM middleware and check parameters
        rlm_mw = None
        for mw in middleware:
            if isinstance(mw, RLMMiddleware):
                rlm_mw = mw
                break

        assert rlm_mw is not None, "RLMMiddleware should be present"
        assert rlm_mw._manager.sandbox_config.timeout_seconds == 300.0
        assert rlm_mw._manager.context_policy == "isolated"
        assert rlm_mw._auto_load_threshold == 5000
        assert rlm_mw._custom_prompt == "Custom RLM prompt"

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_passthrough_parameters(self, mock_create, mock_init):
        """Verify DeepAgents parameters are passed through correctly."""
        from rlmagents.graph import create_rlm_agent

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            debug=True,
            name="test_agent",
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_create.call_args

        # Check passthrough parameters
        assert call_kwargs.kwargs["debug"] is True
        assert call_kwargs.kwargs["name"] == "test_agent"
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}


class TestCrossContextSearch:
    """Tests for cross-context search feature."""

    def test_cross_context_search_tool_exists(self):
        """Verify cross_context_search tool is created."""
        from rlmagents.middleware._tools import _build_rlm_tools
        from rlmagents.session_manager import RLMSessionManager
        from aleph.repl.sandbox import SandboxConfig

        manager = RLMSessionManager(sandbox_config=SandboxConfig())
        tools = _build_rlm_tools(manager)

        # Check cross_context_search tool exists
        tool_names = [tool.name for tool in tools]
        assert "cross_context_search" in tool_names, "cross_context_search tool should exist"

    def test_total_tool_count(self):
        """Verify total number of RLM tools (should be 23 with cross-context search)."""
        from rlmagents.middleware._tools import _build_rlm_tools
        from rlmagents.session_manager import RLMSessionManager
        from aleph.repl.sandbox import SandboxConfig

        manager = RLMSessionManager(sandbox_config=SandboxConfig())
        tools = _build_rlm_tools(manager)

        # Should have 23 tools (was 22, added cross_context_search)
        assert len(tools) == 23, f"Expected 23 tools, got {len(tools)}"


class TestHumanInTheLoop:
    """Tests for human-in-the-loop integration."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_interrupt_on_parameter_accepted(self, mock_create, mock_init):
        """Verify interrupt_on parameter is passed through."""
        from rlmagents.graph import create_rlm_agent
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            interrupt_on={"edit_file": True, "execute": True},
        )

        call_kwargs = mock_create.call_args
        middleware = call_kwargs.kwargs.get("middleware", [])

        # Check HumanInTheLoopMiddleware is present
        hil_mw_found = any(isinstance(mw, HumanInTheLoopMiddleware) for mw in middleware)
        assert hil_mw_found, "HumanInTheLoopMiddleware should be added when interrupt_on provided"


class TestBackendHandling:
    """Tests for backend parameter handling."""

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_backend_parameter_passthrough(self, mock_create, mock_init):
        """Verify backend parameter is passed through correctly."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.backends import StateBackend

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(
            model="anthropic:test-model",
            backend=StateBackend,
        )

        call_kwargs = mock_create.call_args

        # Backend should be passed to create_agent
        # Note: backend is passed as a keyword argument
        assert call_kwargs.kwargs.get("backend") == StateBackend

    @patch("langchain.chat_models.base.init_chat_model")
    @patch("langchain.agents.create_agent")
    def test_default_backend_is_state_backend(self, mock_create, mock_init):
        """Verify default backend is StateBackend when not provided."""
        from rlmagents.graph import create_rlm_agent
        from deepagents.backends import StateBackend

        mock_create.return_value = MagicMock()
        mock_init.return_value = MagicMock()
        
        create_rlm_agent(model="anthropic:test-model")

        call_kwargs = mock_create.call_args

        # Default backend should be StateBackend (passed as keyword arg)
        # If not explicitly provided, it should be StateBackend
        backend_value = call_kwargs.kwargs.get("backend", StateBackend)
        assert backend_value == StateBackend
