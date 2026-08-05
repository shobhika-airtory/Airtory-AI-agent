"""
mcp_client.py
"""

import json
import uuid
from typing import Any, Optional

import httpx


def _parse_response(resp: httpx.Response) -> dict:
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str:
                    return json.loads(data_str)
        raise RuntimeError(f"No data line found in SSE response: {resp.text[:300]}")
    return resp.json()


class MCPClient:
    def __init__(self, base_url: str, args_key: str = "input"):
        self.base_url = base_url.rstrip("/")
        self.args_key = args_key
        self.session_id: Optional[str] = None
        self.client = httpx.Client(timeout=30.0)
        self._tools_cache: Optional[list] = None

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method}
        if params is not None:
            payload["params"] = params

        resp = self.client.post(self.base_url, headers=self._headers(), json=payload)
        resp.raise_for_status()

        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

        data = _parse_response(resp)
        if "error" in data:
            raise RuntimeError(f"MCP error on '{method}': {data['error']}")
        return data.get("result", {})

    def initialize(self) -> dict:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "airtory-creative-agent", "version": "0.1.0"},
            },
        )
        try:
            self.client.post(
                self.base_url,
                headers=self._headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
        except Exception:
            pass
        self._tools_cache = None
        return result

    def list_tools(self, force_refresh: bool = False) -> list:
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
        result = self._rpc("tools/list")
        self._tools_cache = result.get("tools", [])
        return self._tools_cache

    def call_tool(self, name: str, arguments: dict) -> Any:
        params = {"name": name, "arguments": {self.args_key: arguments}}
        result = self._rpc("tools/call", params)

        content = result.get("content")
        if content:
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
        return result