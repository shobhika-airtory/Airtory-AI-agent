# Airtory-AI-agent
Summer internship project
# Airtory Creative Agent

An AI chat agent that lets you build ad campaigns and creatives through plain-language conversation, instead of clicking through a multi-step web UI. Built during a software engineering internship at [Airtory](https://www.airtory.com), an AdTech platform for interactive ad creative.

> **Note on this repo:** this is a showcase version. One file (`template_catalog.py`) has been intentionally left out — it contains Airtory's real internal catalog of ad template names and IDs, which is business data belonging to the company, not something that reflects the actual engineering work here. Its role in the architecture is described in detail below. Everything else in this repo is the real, working implementation.

## The problem

Airtory Studio's own UI requires several manual steps to build an ad creative: pick a campaign, pick an ad type, pick a size, pick a specific template out of hundreds, then fill in a template-specific set of fields correctly. This project replaces that with a chat interface — "create a Quiz-n-Win ad for the Domino's campaign" — while still producing exactly the same result a person clicking through the UI would.

## Why it's not "just call an LLM with some tools"

The obvious approach — give a language model a list of API tools and let it figure out the sequence of calls — turned out to be unreliable with a small, locally-hosted model. It would hallucinate IDs, skip required steps, or get confused across multi-turn conversations. Rather than throw a bigger (and more expensive) model at the problem, the system is built as three layers with decreasing trust in the model:

1. **Fast path.** Simple, predictable requests ("list campaigns," "how many creatives are there") go straight to a tool call. No model involved at all — always fast, always correct.
2. **Deterministic creative wizard.** Building a new creative follows a fixed, code-defined flow (Campaign → Ad Type → Ad Size → Ad Format → required fields) that mirrors Airtory Studio's own creation UI exactly. Every ad format is resolved to its precise template ID upfront, rather than left for the model to guess or fuzzy-match from a name — eliminating a whole class of "which one did you mean" ambiguity that a name-based lookup alone couldn't solve, since Airtory's catalog has dozens of templates that legitimately share the same display name across different categories.
3. **AI fallback.** Anything that doesn't fit the above — edits, open-ended questions, less common actions — falls through to the model, which is given real tools and asked to reason about what to call. This is the only layer where the model has real autonomy, and it's scoped narrowly on purpose.

The result: the parts of the system that get used constantly are 100% reliable because they're not asking an LLM to get anything right, and the model's unreliability is contained to the long tail of requests where a wrong guess is lower-stakes.

## Notable engineering problems solved along the way

- **The platform's own API had undocumented bugs.** List/search endpoints silently capped at 100 results and had a broken name-search filter, both discovered through direct testing rather than documentation. Worked around by fetching individual records by ID instead of relying on search — reliable in every case, at the cost of needing to know or discover the ID first.
- **A "polished" embeddable ad tag wasn't actually returned by any API endpoint.** After tracing through several endpoints that seemed like they should return it, it turned out the platform's own UI *constructs* that tag client-side from data already in hand, rather than fetching it from the server. Once identified, the agent builds it directly instead of chasing a response field that was never going to exist.
- **Silent data corruption from a subtle Python gotcha.** During a large data-entry pass, a Python dictionary literal ended up with duplicate keys after an editing mistake — which doesn't raise a syntax error (the last key silently wins), so a naive syntax check wouldn't have caught it. Caught by actually loading and querying the data structure at runtime, not just checking that the file parsed — which became standard practice for every subsequent change to that file.
- **Ambiguity that only showed up with real data.** Requesting a template by name occasionally matched more than one entry with genuinely different dimensions. Rather than silently picking one (which could hand back a functionally wrong creative) or always blocking with a question (which gets tedious for the 95% of cases with no ambiguity), the agent only asks when there's a *genuine* collision within the specific list the user is actually looking at — and even then, shows the real IDs and dimensions rather than a generic prompt.

## Architecture

```
User message
     |
     v
Fast path? --yes--> direct tool call --> reply
     |no
     v
In an active multi-step flow (building a creative, etc.)? --yes--> advance that flow
     |no
     v
AI fallback loop --> model picks a tool --> tool result --> model responds
```

## Tech stack

- **Backend:** FastAPI (Python)
- **Model:** configurable -- built and tested against both a locally-hosted Ollama model and a cloud-hosted OpenAI-compatible endpoint, swappable via a small adapter interface
- **Tool access:** [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) client, calling the platform's own tool-based API
- **Frontend:** plain HTML/CSS/JS, no framework or build step -- deliberately minimal since the interesting engineering is on the agent side

## Project structure

```
backend/
  main.py              FastAPI app -- chat endpoint, session handling
  agent.py             Core agent logic: fast path, wizard, AI fallback loop
  ovh_client.py        Adapter for a cloud-hosted OpenAI-compatible model endpoint
  ollama_client.py      Adapter for a local Ollama model
  tools_bridge.py       Converts MCP tool schemas into the model-facing tool format
  requirements.txt

frontend/
  index.html
  chat.js
  style.css
```

## What this repo doesn't include, and why

- `template_catalog.py` -- Airtory's actual template catalog data. The agent's wizard logic references this module for a static Ad Type -> Ad Size -> Ad Format -> template ID mapping; without it, the deterministic wizard falls back gracefully to asking the user for a template name or ID directly rather than crashing.
- Any `.env` file, MCP server URL, or model API key -- all real infrastructure, none of it needed to evaluate the code itself.

## Running this yourself

Since the platform-specific API (`mcp_client.py`) and the real template catalog aren't included, this repo is meant to be read rather than run end-to-end. The architecture and agent logic in `agent.py` are the parts worth reviewing.
