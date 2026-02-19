"""Custom tools for the CLI agent."""

import re
from typing import Any, Literal

import requests
from markdownify import markdownify
from tavily import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)
from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = (
    TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None
)

_MAX_FOCUS_TOKENS = 32
_MAX_SNIPPET_CANDIDATES = 256
_FOCUS_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
_SNIPPET_SPLIT_PATTERN = re.compile(r"\n{2,}|(?<=[.!?])\s+")


def _focus_tokens(query: str) -> list[str]:
    """Extract normalized focus tokens from a query string.

    Returns:
        Unique lowercase tokens ordered by appearance.
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _FOCUS_TOKEN_PATTERN.finditer(query.lower()):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= _MAX_FOCUS_TOKENS:
            break
    return tokens


def _snippet_windows(text: str, snippet_size: int) -> list[str]:
    """Split text into semantically useful windows for scoring.

    Returns:
        Ordered snippet candidates with bounded size.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= snippet_size:
        return [normalized]

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for segment in _SNIPPET_SPLIT_PATTERN.split(normalized):
        piece = segment.strip()
        if not piece:
            continue
        piece_len = len(piece)
        if current and current_len + 1 + piece_len > snippet_size:
            windows.append(" ".join(current))
            current = [piece]
            current_len = piece_len
        elif not current and piece_len > snippet_size:
            start = 0
            while start < piece_len:
                windows.append(piece[start : start + snippet_size])
                start += snippet_size
            current = []
            current_len = 0
        else:
            current.append(piece)
            current_len = current_len + piece_len + (1 if current_len > 0 else 0)
        if len(windows) >= _MAX_SNIPPET_CANDIDATES:
            break
    if current and len(windows) < _MAX_SNIPPET_CANDIDATES:
        windows.append(" ".join(current))
    return windows


def _relevance_score(snippet: str, tokens: list[str]) -> float:
    """Compute a lightweight lexical relevance score.

    Returns:
        Score in `[0.0, 1.0]` where larger is more relevant.
    """
    if not tokens:
        return 0.0
    lowered = snippet.lower()
    hit_count = sum(1 for token in tokens if token in lowered)
    if hit_count <= 0:
        return 0.0
    density = hit_count / len(tokens)
    length_bonus = min(len(snippet), 280) / 280
    return density * 0.8 + length_bonus * 0.2


def _rank_content_snippets(
    text: str,
    *,
    focus_query: str,
    snippet_size: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return top ranked snippets from text for a given focus query."""
    tokens = _focus_tokens(focus_query)
    if not tokens:
        return []
    windows = _snippet_windows(text, snippet_size)
    scored: list[dict[str, Any]] = []
    for snippet in windows:
        score = _relevance_score(snippet, tokens)
        if score <= 0:
            continue
        scored.append({"snippet": snippet, "score": round(score, 4)})
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    return scored[:top_k]


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text while preserving readability.

    Returns:
        Truncated text with a stable marker when shortened.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n\n[truncated]"
    keep = max_chars - len(suffix)
    if keep <= 0:
        return suffix[:max_chars]
    return text[:keep] + suffix


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs: dict[str, Any] = {}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(method.upper(), url, timeout=timeout, **kwargs)

        try:
            content = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
    *,
    dynamic_filter: bool = True,
    focus_query: str | None = None,
    snippet_size: int = 420,
) -> dict[str, Any]:
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)
        dynamic_filter: Apply local relevance filtering before returning results.
        focus_query: Optional relevance query override for dynamic filtering.
        snippet_size: Character window size used when selecting snippets.

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query
        - dynamic_filtering: Filtering metadata when enabled

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    response_payload: dict[str, Any]
    try:
        expanded_limit = max_results * 2 if dynamic_filter else max_results
        response = tavily_client.search(
            query,
            max_results=max(max_results, expanded_limit),
            include_raw_content=include_raw_content,
            topic=topic,
        )
        if not isinstance(response, dict):
            return {
                "query": query,
                "results": [],
                "error": "Unexpected web search response format",
            }

        if not dynamic_filter:
            return response

        result_list = response.get("results")
        if not isinstance(result_list, list):
            return response

        effective_focus = focus_query or query
        filtered: list[dict[str, Any]] = []
        snippets: list[dict[str, Any]] = []
        for item in result_list:
            if not isinstance(item, dict):
                continue
            source_text = str(item.get("raw_content") or item.get("content") or "")
            ranked = _rank_content_snippets(
                source_text,
                focus_query=effective_focus,
                snippet_size=max(200, snippet_size),
                top_k=1,
            )
            best = ranked[0] if ranked else None
            entry = dict(item)
            if best is not None:
                entry["content"] = best["snippet"]
                entry["dynamic_filter_score"] = best["score"]
                snippets.append(
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "score": best["score"],
                        "snippet": best["snippet"],
                    }
                )
            if dynamic_filter and "raw_content" in entry:
                entry.pop("raw_content", None)
            filtered.append(entry)

        def _result_score(item: dict[str, Any]) -> float:
            return float(
                item.get(
                    "dynamic_filter_score",
                    item.get("score", 0.0),
                )
            )

        filtered.sort(key=_result_score, reverse=True)
        compact_results = filtered[:max_results]
        enriched = dict(response)
        enriched["results"] = compact_results
        enriched["dynamic_filtering"] = {
            "enabled": True,
            "focus_query": effective_focus,
            "original_results": len(result_list),
            "returned_results": len(compact_results),
            "snippets": snippets[:max_results],
        }
        response_payload = enriched
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}
    else:
        return response_payload


def fetch_url(
    url: str,
    timeout: int = 30,
    *,
    dynamic_filter: bool = True,
    focus_query: str | None = None,
    max_chars: int = 12_000,
    snippet_size: int = 600,
) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)
        dynamic_filter: Apply local relevance filtering before returning content.
        focus_query: Optional relevance query override for dynamic filtering.
        max_chars: Maximum content length returned after filtering.
        snippet_size: Character window size used when selecting snippets.

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters
        - dynamic_filtering: Filtering metadata when enabled

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)
        rendered_content = markdown_content
        filtering_info: dict[str, Any] | None = None
        if dynamic_filter:
            effective_focus = (focus_query or "").strip()
            filtered_snippets: list[dict[str, Any]] = []
            if effective_focus:
                filtered_snippets = _rank_content_snippets(
                    markdown_content,
                    focus_query=effective_focus,
                    snippet_size=max(240, snippet_size),
                    top_k=6,
                )
            if filtered_snippets:
                rendered_content = "\n\n".join(
                    f"- {item['snippet']}" for item in filtered_snippets
                )
            rendered_content = _truncate_text(rendered_content, max_chars=max_chars)
            filtering_info = {
                "enabled": True,
                "focus_query": effective_focus or None,
                "original_chars": len(markdown_content),
                "returned_chars": len(rendered_content),
                "snippet_count": len(filtered_snippets),
            }

        return {
            "url": str(response.url),
            "markdown_content": rendered_content,
            "status_code": response.status_code,
            "content_length": len(rendered_content),
            "dynamic_filtering": filtering_info,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}


def web_research(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    fetch_top_n: int = 3,
    timeout: int = 30,
    *,
    dynamic_filter: bool = True,
    snippet_size: int = 420,
    max_chars_per_page: int = 5000,
) -> dict[str, Any]:
    """Run a programmatic web workflow in one tool call.

    Executes search + fetch + local filtering in one step to reduce model
    round-trips and return only high-signal evidence.

    Args:
        query: The research question to investigate.
        max_results: Number of search results to keep after filtering.
        topic: Search topic type.
        fetch_top_n: Number of top search hits to fetch and summarize.
        timeout: Request timeout for page fetches.
        dynamic_filter: Whether to filter snippets before returning.
        snippet_size: Character window used for relevance snippets.
        max_chars_per_page: Maximum characters to return per fetched page.

    Returns:
        Combined search and fetch output with workflow metadata.
    """
    search_response = web_search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=False,
        dynamic_filter=dynamic_filter,
        focus_query=query,
        snippet_size=snippet_size,
    )
    if "error" in search_response:
        return search_response

    result_list = search_response.get("results")
    if not isinstance(result_list, list):
        return {
            "query": query,
            "results": [],
            "fetched_pages": [],
            "workflow": {"error": "Unexpected search result format"},
        }

    fetch_count = max(0, min(fetch_top_n, len(result_list)))
    fetched_pages: list[dict[str, Any]] = []
    for item in result_list[:fetch_count]:
        if not isinstance(item, dict):
            continue
        page_url = str(item.get("url") or "")
        if not page_url:
            continue
        fetched = fetch_url(
            page_url,
            timeout=timeout,
            dynamic_filter=dynamic_filter,
            focus_query=query,
            max_chars=max_chars_per_page,
            snippet_size=max(300, snippet_size),
        )
        fetched_pages.append(
            {
                "title": str(item.get("title", "")),
                "url": page_url,
                "content": fetched.get("markdown_content", ""),
                "error": fetched.get("error"),
                "dynamic_filtering": fetched.get("dynamic_filtering"),
            }
        )

    return {
        "query": query,
        "results": result_list,
        "fetched_pages": fetched_pages,
        "workflow": {
            "mode": "programmatic_web_research",
            "dynamic_filtering": dynamic_filter,
            "search_results": len(result_list),
            "pages_fetched": len(fetched_pages),
        },
    }
