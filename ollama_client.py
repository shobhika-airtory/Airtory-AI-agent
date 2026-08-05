"""
ollama_client.py
"""

from typing import Optional

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=180.0)

    def chat(self, messages: list, tools: Optional[list] = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": 400},
        }
        if tools:
            payload["tools"] = tools

        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()