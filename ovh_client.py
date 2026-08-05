"""
ovh_client.py
"""

import requests


class OVHClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        # base_url should be the full .../v1/chat/completions endpoint.
        # IMPORTANT: verify this exact URL works for your account before
        # wiring it in for real -- see the curl check in the deployment
        # notes. OVH's per-model subdomain (.../api/openai_compat/) vs.
        # their shared router (oai.endpoints.kepler.ai.cloud.ovh.net/v1)
        # may need slightly different suffixes depending on your account
        # setup.
        self.url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages, tools=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        message = data["choices"][0]["message"]
        
        return {"message": message}