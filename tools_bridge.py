"""
tools_bridge.py

"""

MAX_DESCRIPTION_CHARS = 220


def _shorten(text: str) -> str:
    if not text:
        return ""
    first_chunk = text.split("\n\n")[0].strip()
    if len(first_chunk) > MAX_DESCRIPTION_CHARS:
        first_chunk = first_chunk[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "..."
    return first_chunk


def mcp_tools_to_ollama(mcp_tools: list) -> list:
    ollama_tools = []
    for t in mcp_tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": _shorten(t.get("description", "")),
                    "parameters": t.get(
                        "inputSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
        )
    return ollama_tools