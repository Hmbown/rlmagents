"""Tests for dynamic filtering and programmatic web workflows."""

from unittest.mock import patch

import responses

from deepagents_cli.tools import fetch_url, web_research, web_search


def test_web_search_dynamic_filtering_returns_compact_results() -> None:
    """web_search should return filtered snippets when dynamic filtering is enabled."""
    mock_response = {
        "query": "find database timeout fixes",
        "results": [
            {
                "title": "Timeout Guide",
                "url": "https://example.com/timeout",
                "content": "generic text",
                "raw_content": (
                    "Connection timeout failures usually come from DNS, proxy, "
                    "or keepalive misconfiguration. Use shorter idle keepalive "
                    "and verify socket timeout settings."
                ),
                "score": 0.71,
            },
            {
                "title": "Unrelated",
                "url": "https://example.com/unrelated",
                "content": "nothing useful",
                "raw_content": "This page discusses gardening and travel itineraries.",
                "score": 0.62,
            },
        ],
    }

    with patch("deepagents_cli.tools.tavily_client") as mock_client:
        mock_client.search.return_value = mock_response

        result = web_search(
            "find database timeout fixes",
            max_results=1,
            dynamic_filter=True,
            focus_query="connection timeout keepalive",
            snippet_size=180,
        )

    assert "error" not in result
    assert "dynamic_filtering" in result
    assert result["dynamic_filtering"]["enabled"] is True
    assert result["dynamic_filtering"]["returned_results"] == 1
    assert len(result["results"]) == 1
    assert "raw_content" not in result["results"][0]
    assert "timeout" in str(result["results"][0].get("content", "")).lower()


@responses.activate
def test_fetch_url_dynamic_filtering_focuses_content() -> None:
    """fetch_url should reduce output to focus-relevant snippets."""
    responses.add(
        responses.GET,
        "https://example.com/doc",
        body=(
            "<html><body>"
            "<h1>System Setup</h1>"
            "<p>General introduction and unrelated context.</p>"
            "<p>Set database port to 3306 and verify connection timeout values.</p>"
            "<p>For retries, use exponential backoff and keepalive settings.</p>"
            "</body></html>"
        ),
        status=200,
    )

    result = fetch_url(
        "https://example.com/doc",
        dynamic_filter=True,
        focus_query="database port 3306 timeout",
        max_chars=400,
        snippet_size=140,
    )

    assert "error" not in result
    assert result["content_length"] <= 400
    assert "dynamic_filtering" in result
    assert result["dynamic_filtering"]["enabled"] is True
    assert "3306" in result["markdown_content"]


def test_web_research_runs_programmatic_workflow() -> None:
    """web_research should orchestrate search and follow-up fetch calls."""
    with (
        patch("deepagents_cli.tools.web_search") as mock_search,
        patch("deepagents_cli.tools.fetch_url") as mock_fetch,
    ):
        mock_search.return_value = {
            "query": "compare model latency",
            "results": [
                {
                    "title": "Bench 1",
                    "url": "https://example.com/bench-1",
                    "content": "benchmark summary",
                },
                {
                    "title": "Bench 2",
                    "url": "https://example.com/bench-2",
                    "content": "benchmark summary",
                },
            ],
        }
        mock_fetch.return_value = {
            "url": "https://example.com/bench-1",
            "markdown_content": "filtered page content",
            "dynamic_filtering": {"enabled": True},
        }

        result = web_research(
            "compare model latency",
            max_results=2,
            fetch_top_n=1,
            dynamic_filter=True,
        )

    mock_search.assert_called_once()
    mock_fetch.assert_called_once()
    assert result["workflow"]["mode"] == "programmatic_web_research"
    assert result["workflow"]["pages_fetched"] == 1
    assert len(result["fetched_pages"]) == 1
