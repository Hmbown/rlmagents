"""Pytest configuration for rlmagents tests."""

import pytest


@pytest.fixture
def anyio_backend():
    """Configure anyio backend for async tests."""
    return "asyncio"
