"""Lightweight MCP client for calling CAST AI MCP server tools.

The CAST AI MCP server exposes tools like analyze_snapshot_with_code,
loki_query, and get_cluster_snapshot_summary via Streamable HTTP
transport (JSON-RPC 2.0 over HTTP).

Authentication:
  - JWT token: passed as Authorization: Bearer header
  - IAP token: passed as GCP_IAP_TOKEN cookie (for IAP-protected endpoints)
  - Tokens loaded from ~/.castai/credentials.json and ~/.castai/iap_token.json

Usage:
    client = MCPClient("https://castai-mcp.prod-master.cast.ai/mcp")
    result = await client.call_tool("analyze_snapshot_with_code", {
        "cluster_id_or_name": "my-cluster",
        "analysis_code": "result = len(snapshot.get_pods())",
        "description": "count pods",
    })
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx

# Transient HTTP status codes worth retrying
_RETRYABLE_STATUS = {502, 503, 504, 429}

logger = logging.getLogger("watchdog.mcp_client")

# Token file locations (created by browser_login.py)
CASTAI_DIR = Path.home() / ".castai"
JWT_FILE = CASTAI_DIR / "credentials.json"
IAP_FILE = CASTAI_DIR / "iap_token.json"


def _load_jwt() -> str | None:
    """Load JWT token from ~/.castai/credentials.json."""
    if not JWT_FILE.exists():
        return None
    try:
        data = json.loads(JWT_FILE.read_text())
        return data.get("token") or data.get("jwt") or data.get("access_token")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load JWT from %s: %s", JWT_FILE, e)
        return None


def _load_iap_token() -> tuple[str, str] | None:
    """Load IAP cookie from ~/.castai/iap_token.json.

    Returns (cookie_name, cookie_value) or None.
    The file format from browser_login.py uses cookie_name/cookie_value keys.
    """
    if not IAP_FILE.exists():
        return None
    try:
        data = json.loads(IAP_FILE.read_text())
        # browser_login.py format: cookie_name + cookie_value
        name = data.get("cookie_name")
        value = data.get("cookie_value")
        if name and value:
            return (name, value)
        # Fallback: flat token field
        token = data.get("token") or data.get("iap_token")
        if token:
            return ("GCP_IAP_AUTH_TOKEN", token)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load IAP token from %s: %s", IAP_FILE, e)
        return None


class MCPClient:
    """Streamable HTTP client for the CAST AI MCP server."""

    def __init__(
        self,
        mcp_url: str,
        jwt_token: str | None = None,
        iap_token: str | None = None,
        organization_id: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.mcp_url = mcp_url.rstrip("/")
        self.jwt_token = jwt_token or _load_jwt()
        self.organization_id = organization_id
        self.timeout = timeout
        self._request_id = 0
        self._session_id: str | None = None

        # IAP cookie: (name, value) tuple
        loaded_iap = _load_iap_token()
        if iap_token:
            # Env var override — use generic cookie name
            self._iap_cookie: tuple[str, str] | None = ("GCP_IAP_AUTH_TOKEN", iap_token)
        elif loaded_iap:
            self._iap_cookie = loaded_iap
        else:
            self._iap_cookie = None

        if not self.jwt_token:
            logger.warning(
                "No JWT token found. Run browser_login.py or set CASTAI_JWT_TOKEN. "
                "MCP calls will likely fail."
            )
        if not self._iap_cookie:
            logger.warning(
                "No IAP token found. Run browser_login.py to generate ~/.castai/iap_token.json. "
                "MCP calls to IAP-protected endpoints will fail."
            )

    def _build_headers(self) -> dict[str, str]:
        """Build auth headers for MCP requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        if self._iap_cookie:
            cookie_name, cookie_value = self._iap_cookie
            headers["Cookie"] = f"{cookie_name}={cookie_value}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def initialize(self) -> bool:
        """Send MCP initialize handshake. Returns True on success."""
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "grip-watchdog",
                    "version": "0.1.0",
                },
            },
            "id": self._next_id(),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.mcp_url,
                    headers=self._build_headers(),
                    json=payload,
                )
                resp.raise_for_status()

                # Capture session ID from response header
                session_id = resp.headers.get("Mcp-Session-Id")
                if session_id:
                    self._session_id = session_id

                content_type = resp.headers.get("content-type", "")

                # Parse response — may be JSON or SSE
                if "text/event-stream" in content_type:
                    result = self._parse_sse_raw(resp.text)
                else:
                    body = resp.text.strip()
                    if not body:
                        # Empty 200 — some servers ack initialize with no body
                        logger.info("MCP initialize returned empty 200 (session=%s)", self._session_id or "none")
                        result = {}
                    else:
                        result = json.loads(body)

                if isinstance(result, dict) and "error" in result:
                    logger.error("MCP initialize error: %s", result["error"])
                    return False

                logger.info("MCP session initialized (session=%s)", self._session_id or "none")

                # Send initialized notification
                await client.post(
                    self.mcp_url,
                    headers=self._build_headers(),
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                    },
                )
                return True

        except Exception as e:
            logger.error("MCP initialize failed: %s", e)
            return False

    def _parse_sse_raw(self, body: str) -> dict:
        """Parse SSE stream and return the last JSON-RPC message."""
        last_data = None
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    last_data = json.loads(line[6:])
                except json.JSONDecodeError:
                    pass
        return last_data or {}

    def _parse_jsonrpc_body(self, body: str, tool_name: str = "") -> dict:
        """Parse a JSON-RPC response body that may contain concatenated JSON objects.

        The MCP server sometimes returns multiple JSON objects concatenated
        without separators, e.g. `{}{"jsonrpc":"2.0",...}`. We want the last
        complete JSON-RPC result (the one with "result" or "error").
        """
        # Fast path: single valid JSON
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

        # Try newline-separated (one JSON per line)
        lines = body.split("\n")
        if len(lines) > 1:
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                            return obj
                    except json.JSONDecodeError:
                        continue

        # Concatenated JSON objects on one line — use decoder to walk through
        objects = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(body):
            # Skip whitespace
            while idx < len(body) and body[idx] in " \t\r\n":
                idx += 1
            if idx >= len(body):
                break
            try:
                obj, end_idx = decoder.raw_decode(body, idx)
                objects.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                idx += 1  # skip undecodable char

        if not objects:
            raise json.JSONDecodeError("No JSON objects found", body, 0)

        # Prefer the object that has a "result" key (the actual JSON-RPC response)
        for obj in reversed(objects):
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                logger.debug(
                    "MCP %s: parsed %d concatenated JSON objects, using JSON-RPC result",
                    tool_name, len(objects),
                )
                return obj

        # Fallback to last object
        logger.debug(
            "MCP %s: parsed %d concatenated JSON objects, using last",
            tool_name, len(objects),
        )
        return objects[-1] if isinstance(objects[-1], dict) else {"result": objects[-1]}

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        max_retries: int = 3,
    ) -> Any:
        """Call an MCP tool and return the result content.

        Returns the parsed result on success, or None on failure.
        Retries transient errors (502/503/504/429/timeouts) with exponential backoff.
        """
        # Auto-inject organization_id for tools that accept it.
        _NO_ORG_ID_TOOLS = {
            "analyze_snapshot_with_code",
            "get_cluster_snapshot_summary",
            "extract_fields_from_all",
            "loki_query",
        }
        if (self.organization_id
                and "organization_id" not in arguments
                and tool_name not in _NO_ORG_ID_TOOLS):
            arguments = {**arguments, "organization_id": self.organization_id}

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": self._next_id(),
            }

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.mcp_url,
                        headers=self._build_headers(),
                        json=payload,
                    )

                    # Retry on transient HTTP errors
                    if resp.status_code in _RETRYABLE_STATUS:
                        wait = 2 ** attempt  # 2, 4, 8 seconds
                        logger.warning(
                            "MCP %s → %d (attempt %d/%d), retrying in %ds",
                            tool_name, resp.status_code, attempt, max_retries, wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    body_preview = resp.text[:500].replace("\n", "\\n")
                    logger.debug(
                        "MCP tool %s → %d %s | body[:%d]: %s",
                        tool_name, resp.status_code, content_type,
                        min(500, len(resp.text)), body_preview,
                    )

                    # Handle SSE response (streaming)
                    if "text/event-stream" in content_type:
                        extracted = self._parse_sse_response(resp.text)
                        logger.debug(
                            "MCP %s SSE → type=%s",
                            tool_name, type(extracted).__name__,
                        )
                        return extracted

                    # Handle direct JSON response
                    body = resp.text.strip()
                    if not body:
                        logger.warning("MCP tool %s returned empty body", tool_name)
                        return None

                    result = self._parse_jsonrpc_body(body, tool_name)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(
                            "MCP tool %s error: %s",
                            tool_name, result["error"],
                        )
                        return None

                    extracted = self._extract_content(result)
                    logger.debug(
                        "MCP %s JSON → type=%s",
                        tool_name, type(extracted).__name__,
                    )
                    return extracted

            except httpx.TimeoutException:
                last_error = Exception(f"timeout after {self.timeout}s")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "MCP %s timed out (attempt %d/%d), retrying in %ds",
                        tool_name, attempt, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
            except httpx.HTTPStatusError as e:
                # Non-retryable HTTP error (4xx etc)
                logger.error("MCP tool %s failed: HTTP %d", tool_name, e.response.status_code)
                return None
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "MCP %s failed (attempt %d/%d): %s, retrying in %ds",
                        tool_name, attempt, max_retries, e, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

        logger.error("MCP tool %s failed after %d attempts: %s", tool_name, max_retries, last_error)
        return None

    def _extract_content(self, result: dict) -> Any:
        """Extract tool result content from JSON-RPC response."""
        result_obj = result.get("result", {})
        if not isinstance(result_obj, dict):
            return result_obj

        # Check MCP isError flag
        if result_obj.get("isError"):
            texts = [
                c.get("text", "") for c in result_obj.get("content", [])
                if c.get("type") == "text"
            ]
            error_msg = "\n".join(texts) or "unknown MCP tool error"
            raise RuntimeError(f"MCP tool error: {error_msg}")

        content_list = result_obj.get("content", [])
        if not content_list:
            return result_obj

        # Collect text blocks
        texts = [c.get("text", "") for c in content_list if c.get("type") == "text"]
        if not texts:
            return result.get("result")

        # Parse each text block
        parsed = []
        for t in texts:
            parsed.append(self._safe_parse_json(t))

        # Single result — return directly
        if len(parsed) == 1:
            return parsed[0]

        # Multiple parsed dicts — merge them
        if all(isinstance(p, dict) for p in parsed):
            merged = {}
            for p in parsed:
                merged.update(p)
            return merged

        # Mixed — return largest dict, or first non-empty
        dicts = [p for p in parsed if isinstance(p, dict)]
        if dicts:
            return max(dicts, key=len)

        return parsed[0]

    def _safe_parse_json(self, text: str) -> Any:
        """Parse JSON text, handling concatenated objects like `{}{...}`.

        Returns the largest/most-useful parsed object.
        """
        text = text.strip()
        if not text:
            return {}

        # Fast path
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Concatenated JSON — walk with raw_decode, keep the biggest dict
        decoder = json.JSONDecoder()
        objects = []
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] in " \t\r\n":
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end_idx = decoder.raw_decode(text, idx)
                objects.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                idx += 1

        if not objects:
            return text  # give up, return raw string

        # Prefer the largest dict (the actual data, not empty `{}`)
        dicts = [o for o in objects if isinstance(o, dict)]
        if dicts:
            return max(dicts, key=len)
        return objects[-1]

    def _parse_sse_response(self, body: str) -> Any:
        """Parse SSE event stream to extract the final tool result."""
        last_data = None
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    last_data = json.loads(line[6:])
                except json.JSONDecodeError:
                    pass

        if last_data:
            return self._extract_content(last_data)
        return None

    async def analyze_snapshot(
        self,
        cluster_id: str,
        code: str,
        description: str,
        timeout: int = 30,
        snapshot_time: str | None = None,
    ) -> Any:
        """Convenience: call analyze_snapshot_with_code.

        Args:
            snapshot_time: ISO 8601 timestamp to analyze a historical snapshot
                           instead of the latest. e.g. "2026-05-21T05:00:00Z"
        """
        args = {
            "cluster_id_or_name": cluster_id,
            "analysis_code": code,
            "description": description,
            "timeout": timeout,
        }
        if snapshot_time:
            args["snapshot_time"] = snapshot_time
        return await self.call_tool("analyze_snapshot_with_code", args)

    async def loki_query(
        self,
        query: str,
        start: str | None = None,
        limit: int = 100,
    ) -> Any:
        """Convenience: call loki_query."""
        args: dict[str, Any] = {"query": query, "limit": limit}
        if start:
            args["start"] = start
        return await self.call_tool("loki_query", args)

    async def get_snapshot_summary(self, cluster_id: str) -> Any:
        """Convenience: call get_cluster_snapshot_summary."""
        return await self.call_tool("get_cluster_snapshot_summary", {
            "cluster_id_or_name": cluster_id,
        })
