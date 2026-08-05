"""
agent.py
----------
Three layers, in order of how much control we hand to the AI:

1. FAST PATH: simple, predictable "list X" requests -> direct tool calls,
   no AI involved. Always fast, always correct.
2. CREATE-CREATIVE WIZARD: a fixed, step-by-step flow (in code, not left to
   the AI to sequence) for building a new creative -- including the
   top-level "fallback" image, and a follow-up offer for a preview link or
   a full embeddable ad tag (which needs a placement).
3. AI LOOP: for everything else (editing, other lookups), the model thinks
   and calls tools as needed.
"""

import json
import re
import difflib
import concurrent.futures
from datetime import datetime, timedelta, timezone

from mcp_client import MCPClient
from ollama_client import OllamaClient
from tools_bridge import mcp_tools_to_ollama
from template_catalog import AD_TYPES, type_options, size_options, format_names

SYSTEM_PROMPT = """You are the Airtory Studio Creative Agent.

You help users manage campaigns, creatives, templates, and placements using
the tools available to you.

CRITICAL RULES:
- If the user asks you to find, get, check, update, or delete something, you
  MUST call the matching tool. Never answer with just words, never invent data.
- NEVER invent or guess an ID (campaignid, creativeid, formatid, etc.). IDs
  are always 32-character strings from a real tool result. If you don't
  already have the real ID in this conversation, call a lookup tool first.
- NEVER call any create/update/delete tool using placeholder, example, or
  made-up values. If you don't have real values, ask the user.
- Keep replies short and conversational. NEVER write Python code, JSON
  parsing code, or raw JSON in your replies -- always plain English or a
  clean bullet list.
- Never fetch more than 20 items unless the user explicitly asks for more.
"""

MAX_TOOL_ROUNDS = 6


_OLLAMA_TIMEOUT_SECONDS = 30
_ollama_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _call_ollama_with_timeout(ollama, messages, tools, timeout=_OLLAMA_TIMEOUT_SECONDS):
    """Runs ollama.chat() on a worker thread and bails out after `timeout`
    seconds instead of blocking forever. Returns None on timeout (the
    underlying call keeps running in the background thread and its result
    is simply discarded -- harmless for a chat demo, just not awaited)."""
    future = _ollama_executor.submit(ollama.chat, messages, tools=tools)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return None

PENDING_CREATIVES_QUESTION = (
    "Which campaign's creatives would you like to see? (Or say \"all\" to "
    "see the most recent 20 across every campaign.)"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _trim_tool_result(raw: str, max_items: int = 20, max_chars: int = 6000) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:max_chars]

    if isinstance(data, dict):
        for key in list(data.keys()):
            if isinstance(data[key], list) and len(data[key]) > max_items:
                data[key] = data[key][:max_items]
                data[f"{key}_truncated"] = True
        trimmed = json.dumps(data)
    else:
        trimmed = raw

    return trimmed[:max_chars]


def _safe_json(raw) -> dict:
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        text = str(raw)
        if "validation error" in text.lower():
            print(f"[agent] WARNING: tool returned a validation error instead of data: {text[:300]}")
        return {}


def _format_list_reply(items: list, empty_msg: str = "I didn't find any results for that.") -> str:
    if not items:
        return empty_msg
    lines = []
    for item in items[:20]:
        name = item.get("cname") or item.get("name") or item.get("id") or "Unnamed"
        extra = ""
        if "advertiser" in item:
            extra = f" — {item['advertiser']}"
        elif "campaign" in item and isinstance(item["campaign"], dict):
            extra = f" — campaign: {item['campaign'].get('cname', '')}"
        elif "type" in item:
            extra = f" — {item['type']}"
        lines.append(f"- {name}{extra}")
    return f"Here are the top {len(lines)}:\n" + "\n".join(lines)


def _clean_name_answer(text: str) -> str:
    answer = text.strip()
    for filler in (" campaign", " campaigns", "campaign ", "the "):
        answer = re.sub(re.escape(filler), " ", answer, flags=re.I).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _campaign_match_text(m) -> str:
    """_CAMPAIGN_NAME_PATTERN has two alternatives (with/without a trailing
    'campaign' keyword) -- whichever one matched, pull out its group."""
    if not m:
        return None
    return (m.group(1) or m.group(2) or "").strip() or None


def _decode_ad_tag(tag: str) -> str:
    """Airtory's placement/generate_preview tools return the ad tag as a
    base64-encoded string (their tool description says so explicitly). If
    we hand that raw base64 to the user, it looks like broken gibberish --
    decode it so they get the real, usable <script>/<ins> HTML directly."""
    import base64
    try:
        padded = tag + "=" * (-len(tag) % 4)
        decoded = base64.b64decode(padded).decode("utf-8")
        if "<script" in decoded or "<ins" in decoded or "<a " in decoded:
            return decoded
    except Exception:
        pass
    return tag  # wasn't valid base64 / didn't decode to HTML -- return as-is


def _find_ins_tag(obj):
    """Recursively search a tool result for the polished <ins>...</ins> tag
    (the one Airtory Studio's UI shows as the "Ins Tag") -- as opposed to
    the raw <script>var adTag=... snippet from get_placement_preview. We
    search by content, not by a specific key name, since different
    endpoints/response shapes may nest this under different keys."""
    if isinstance(obj, str):
        if "<ins" in obj and "airtory" in obj.lower():
            return obj
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_ins_tag(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_ins_tag(item)
            if found:
                return found
    return None


def _build_ins_tag(creativeid: str) -> str:
    """Airtory Studio's polished 'Ins Tag' repeats the CREATIVE ID in both
    the data-airtory.placement and data-airtory.creative attributes --
    confirmed directly against Studio's own UI output, which showed the
    identical ID in both spots (matching the Creative ID field, not a
    separate placement ID). It's a fixed template built client-side, not
    something any tool call returns."""
    return (
        f'<ins id="airtory" class="airtory-ads" '
        f'data-airtory.placement="{creativeid}" '
        f'data-airtory.creative="{creativeid}"></ins>'
        f'<script src="https://cdn.airtory.com/js/ins/display.min.js"></script>'
    )


def _find_first_id(obj, keys=("creativeid", "id")) -> str:
    """Dig through a tool result looking for a creative/placement ID under
    a few common key names, since exact response shapes vary."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        for v in obj.values():
            found = _find_first_id(v, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_id(item, keys)
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# FAST PATH: simple "list X" requests
# ---------------------------------------------------------------------------

_LIST_PATTERNS = [
    (re.compile(r"\b(list|show|see|get|give)\b.*\bcampaigns?\b", re.I), "list_campaigns"),
    (re.compile(r"\b(list|show|see|get|give)\b.*\btemplates?\b", re.I), "get_templates"),
    (re.compile(r"\b(list|show|see|get|give)\b.*\bformats?\b", re.I), "get_templates"),
    (re.compile(r"\b(list|show|see|get|give)\b.*\bplacements?\b", re.I), "list_placements"),
    (re.compile(r"\b(list|show|see|get|give)\b.*\bassets?\b", re.I), "get_assets"),
]

_CREATIVES_PATTERN = re.compile(r"\b(list|show|see|get|give)\b.*\bcreatives?\b", re.I)


# ---------------------------------------------------------------------------
# CREATE-CREATIVE WIZARD
# ---------------------------------------------------------------------------

_CREATE_INTENT_PATTERN = re.compile(r"\b(create|make|build|generate|set up|setup)\b.*\b(ad|creative)\b", re.I)

_CREATE_PATTERN = re.compile(
    r"\b(?:create|make|build|generate)\b.*?\b(?:a|an)?\s*(.+?)\s*(?:ad|creative)s?\b.*?\b(?:for|under|in|inside)\b\s*(?:the\s+)?(.+?)(?:\s*campaign)?$",
    re.I,
)

_TEMPLATE_ID_PATTERN = re.compile(r"\b(?:template|format)(?:\s*id)?\s*#?\s*(\d+)\b", re.I)

_EXISTING_CREATIVE_LOOKUP_PATTERN = re.compile(
    r"\b(preview|tag)\b.*?\b(?:of|for)\b\s+(?:the\s+)?(?:creative\s+)?([A-Za-z0-9_\- ]+?)"
    r"\s+(?:under|in|from)\s+(?:the\s+)?([A-Za-z0-9_\- ]+?)(?:\s*campaign)?\s*$",
    re.I,
)

_DUPLICATE_CREATIVE_PATTERN = re.compile(
    r"\bduplicate\b\s+(?:the\s+)?(?:creative\s+)?([A-Za-z0-9_\- ]+?)"
    r"\s+(?:under|in|from)\s+(?:the\s+)?([A-Za-z0-9_\- ]+?)(?:\s*campaign)?\s*$",
    re.I,
)

_EDIT_CREATIVE_PATTERN = re.compile(
    r"\b(?:edit|update|modify|change)\b\s+(?:the\s+)?(?:creative\s+)?([A-Za-z0-9_\- ]+?)"
    r"\s+(?:under|in|from)\s+(?:the\s+)?([A-Za-z0-9_\- ]+?)(?:\s*campaign)?\s*$",
    re.I,
)

_EDIT_INTENT_PATTERN = re.compile(r"\b(edit|update|modify|change)\b.*\bcreative\b", re.I)

_SUGGEST_ME_PATTERN = re.compile(
    r"\b(suggest|recommend|surprise me|you choose|you pick|your choice|"
    r"not sure|don'?t know|no idea|whatever|any(?:thing)? (?:is|works)|pick (?:one )?for me)\b",
    re.I,
)


def _extract_creative_name_flexible(text: str):
    patterns = [
        r"\b(?:of|for)\s+(?:the\s+)?(?:creative\s+)?([A-Za-z0-9_\-]+)\s+creative\b",
        r"\b(?:of|for)\s+(?:the\s+)?creative\s+([A-Za-z0-9_\-]+)\b",
        r"\bcreative\s+([A-Za-z0-9_\-]+)\b",
        r"\bduplicate\s+([A-Za-z0-9_\-]+)\b",
        r"\b(?:edit|update|modify|change)\s+([A-Za-z0-9_\-]+)\b",
        # last-resort fallback: no "creative" keyword at all, e.g.
        # "give preview link of puppy" -- just grab whatever follows
        # "of"/"for" at the end of the message.
        r"\b(?:of|for)\s+(?:the\s+)?([A-Za-z0-9_\-]+)\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            candidate = m.group(1)
            if candidate.lower() not in ("the", "a", "an", "this", "that"):
                return candidate
    return None


def _parse_creative_action_request(text: str):
    """Detects a request to act on an EXISTING creative (preview/tag,
    duplicate, or edit) and pulls out whatever creative name / campaign
    name it can. Missing pieces are left as None -- the caller asks for
    them rather than guessing, so this never needs a perfectly-phrased
    single sentence to work."""
    text_l = text.lower()
    action = None
    wants = None

    if "duplicate" in text_l and re.search(r"\bcreative\b", text_l):
        action = "duplicate"
    elif _EDIT_INTENT_PATTERN.search(text):
        action = "edit"
    elif (
        ("preview" in text_l or "tag" in text_l)
        and not _CREATE_INTENT_PATTERN.search(text)
    ):
        action = "preview_tag"
        has_preview = "preview" in text_l
        has_tag = "tag" in text_l
        wants = "preview and tag" if has_preview and has_tag else ("preview" if has_preview else "tag")

    if not action:
        return None

    creative_name = _extract_creative_name_flexible(text)
    campaign_match = _CAMPAIGN_NAME_PATTERN.search(text)
    campaign_name = _campaign_match_text(campaign_match)

    # "for X" is genuinely ambiguous between "campaign named X" and
    # "creative named X". If both extractors land on the exact same
    # phrase, the sentence was actually about the creative -- treating it
    # as ALSO a campaign filter sends us looking for a nonexistent
    # campaign instead of the real creative.
    if campaign_name and creative_name and campaign_name.strip().lower() == creative_name.strip().lower():
        campaign_name = None

    return {"action": action, "creative_name": creative_name, "campaign_name": campaign_name, "wants": wants}

_COUNT_PATTERN = re.compile(
    r"how many\s+(campaigns?|creatives?|placements?|templates?|formats?|assets?)\b",
    re.I,
)

_NEW_CAMPAIGN_INTENT_PATTERN = re.compile(r"\b(new|create|make|set ?up)\b.*\bcampaign\b|^\s*new\s*$", re.I)

_STANDALONE_CAMPAIGN_INTENT_PATTERN = re.compile(r"\b(create|make|build|set ?up)\b.*\bcampaign\b", re.I)


def _fuzzy_word_match(text: str, target: str, cutoff: float = 0.78) -> bool:
    """Typo-tolerant check for a single keyword -- e.g. catches 'campign',
    'campaing', 'campain' all resolving to 'campaign' -- using stdlib
    difflib so no new dependency is needed. Deliberately only used for
    longer/distinctive words (campaign, creative); short words like "ad"
    are too easy to false-positive on and aren't fuzzy-matched."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    return any(difflib.SequenceMatcher(None, w, target).ratio() >= cutoff for w in words)


def _looks_like_campaign_start_intent(text: str) -> bool:
    """Typo-tolerant version of _STANDALONE_CAMPAIGN_INTENT_PATTERN --
    catches misspellings like 'campign'/'campaing' without also matching
    on an unrelated 'creative' typo (which should start a different flow)."""
    if _STANDALONE_CAMPAIGN_INTENT_PATTERN.search(text):
        return True
    has_action_verb = re.search(r"\b(create|make|build|set ?up|setup)\b", text, re.I)
    return bool(has_action_verb and _fuzzy_word_match(text, "campaign"))


def _looks_like_fresh_start_intent(text: str) -> bool:
    """Broad, typo-tolerant detector for 'the user clearly wants to start
    something brand new' (a campaign, or an ad/creative). Used in two
    places: (1) as a typo-tolerant fallback for the main creation
    triggers, and (2) to let an unambiguous fresh request override stale
    wizard state instead of being silently swallowed by whatever
    unrelated question the agent was stuck on."""
    if _CREATE_INTENT_PATTERN.search(text) or _STANDALONE_CAMPAIGN_INTENT_PATTERN.search(text):
        return True
    has_action_verb = re.search(r"\b(create|make|build|generate|set ?up|setup)\b", text, re.I)
    if has_action_verb and (_fuzzy_word_match(text, "campaign") or _fuzzy_word_match(text, "creative")):
        return True
    return False

_WHOAMI_PATTERN = re.compile(
    r"\bwho\s*am\s*i\b|\bmy\s+(?:profile|account|user)\b|\bcurrent\s+user\b|\blogged\s+in\s+as\b",
    re.I,
)

_DELETE_CAMPAIGN_PATTERN = re.compile(
    r"\b(?:delete|remove)\b\s+(?:the\s+)?campaign\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*$"
    r"|\b(?:delete|remove)\b\s+(?:the\s+)?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s+campaign\s*$",
    re.I,
)

_DELETE_CREATIVE_PATTERN = re.compile(
    r"\b(?:delete|remove)\b\s+(?:the\s+)?creative\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*$"
    r"|\b(?:delete|remove)\b\s+(?:the\s+)?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s+creative\s*$",
    re.I,
)

_RENAME_CAMPAIGN_PATTERN = re.compile(
    r"\b(?:rename|update)\b\s+(?:the\s+)?campaign\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s+to\s+"
    r"([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*$",
    re.I,
)

_CAMPAIGN_DETAIL_PATTERN = re.compile(
    r"\b(?:details?|info(?:rmation)?)\b.*?\bcampaign\b|\bcampaign\b.*?\b(?:details?|info(?:rmation)?)\b",
    re.I,
)

_CREATIVE_DETAIL_PATTERN = re.compile(
    r"\b(?:details?|info(?:rmation)?)\b.*?\bcreative\b|\bcreative\b.*?\b(?:details?|info(?:rmation)?)\b",
    re.I,
)

_TEMPLATE_ID_LOOKUP_PATTERN = re.compile(
    r"\btemplate\b.{0,20}?\b(\d+)\b|\b(\d+)\b.{0,20}?\btemplate\b",
    re.I,
)

_TEMPLATE_NAME_TO_ID_PATTERN = re.compile(
    r"\b(?:id|format\s*id|formatid)\b.*?\btemplate\b|\btemplate\b.*?\b(?:id|format\s*id|formatid)\b",
    re.I,
)

_DETAIL_LOOKUP_STOPWORDS = {
    "the", "a", "an", "of", "about", "regarding", "on", "for", "me", "give",
    "show", "see", "get", "list", "details", "detail", "info", "information",
    "my", "please", "tell", "what", "is", "can", "you", "would", "like",
}


def _extract_name_before_keyword(text: str, keyword: str):
    """Grabs whatever comes right before a literal keyword like 'campaign'
    or 'creative' -- e.g. 'give me details of the ShobhikaTest campaign'
    or 'details of the dawg creative' -- regardless of how the rest of
    the sentence is phrased, by capturing the whole prefix up to the
    keyword and stripping known filler words from the front."""
    m = re.search(
        r"\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,8})\s+" + re.escape(keyword) + r"\b",
        text, re.I,
    )
    if not m:
        return None
    words = m.group(1).split()
    while words and words[0].lower() in _DETAIL_LOOKUP_STOPWORDS:
        words.pop(0)
    return " ".join(words) if words else None


def _extract_name_after_keyword(text: str, keyword: str):
    """Grabs whatever comes right after a literal keyword -- e.g. 'give
    details about the creative dawg' phrases it as 'creative NAME'
    instead of 'NAME creative'. Strips filler from both ends of the
    captured span."""
    m = re.search(
        r"\b" + re.escape(keyword) + r"\b\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,8})",
        text, re.I,
    )
    if not m:
        return None
    words = m.group(1).split()
    while words and words[0].lower() in _DETAIL_LOOKUP_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _DETAIL_LOOKUP_STOPWORDS:
        words.pop()
    return " ".join(words) if words else None


def _extract_name_near_keyword(text: str, keyword: str):
    """People phrase this both ways -- 'the dawg creative' AND 'the
    creative dawg' -- so try both directions instead of assuming one."""
    return _extract_name_before_keyword(text, keyword) or _extract_name_after_keyword(text, keyword)


_USE_ID_CORRECTION_PATTERN = re.compile(
    r"\b(?:use|switch|change|go with|try)\b.*?\b(?:id\s+|format\s+)?(\d+)\b",
    re.I,
)
_ORDINAL_INDEX_PATTERN = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b.*\bcreatives?\b", re.I)


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

_CAMPAIGN_NAME_FROM_TRIGGER_PATTERN = re.compile(
    r"\b(?:called|named|call it|name it)\s+[\"']?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)[\"']?\s*$",
    re.I,
)


def _extract_campaign_name_from_trigger(text: str):
    """Pull a campaign name straight out of the initial request (e.g. "make
    a campaign called arigato") instead of discarding it and re-asking for
    a name the user already gave."""
    m = _CAMPAIGN_NAME_FROM_TRIGGER_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None

_TEMPLATE_MENTION_PATTERN = re.compile(r"\b(?:fid|template|format)(?:\s*id)?\s*#?\s*(\d+)\b", re.I)

_STOPWORDS = {
    "tell", "me", "more", "about", "the", "a", "an", "property", "field",
    "properties", "fields", "in", "of", "for", "can", "you", "please",
    "what", "is", "explain", "using", "fid", "template", "format", "id",
    "creative", "build", "trying", "hey", "hi", "im",
}

_CAMPAIGN_NAME_PATTERN = re.compile(
    r"\b(?:under|for|in|inside)\s+(?:the\s+)?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*campaign\b"
    r"|\b(?:under|for|in|inside)\s+(?:the\s+)?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*$",
    re.I,
)

# used specifically for parsing a brand-new "make an ad ..." request, where
# a bare "for X" (no "campaign" keyword) is more often just describing what
# the ad is for -- "for my pizza shop" -- than actually naming a campaign.
# the bare fallback above is still needed elsewhere (e.g. "details for
# ShobhikaTest"), just not here.
_CAMPAIGN_NAME_STRICT_PATTERN = re.compile(
    r"\b(?:under|for|in|inside)\s+(?:the\s+)?([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*campaign\b",
    re.I,
)

_CREATIVE_NAME_PATTERN = re.compile(
    r"\b(?:name it|named|call it|call the creative)\s*[:\-]?\s*\"?([A-Za-z0-9_\- ]+?)\"?\s*$",
    re.I,
)

_FORMAT_KEYWORD_PATTERN = re.compile(
    r"\busing\s+(?:the\s+)?(.+?)\s*template\b"                        # "using X template"
    r"|\busing\s+(?:the\s+)?(.+?)\s+(?:under|for|in|inside)\b"        # "using X under/for/in Y"
    r"|\busing\s+(?:the\s+)?(.+)$"                                    # "using X" (rest of message)
    r"|\btemplate\s*[:\-]?\s*(.+)$",
    re.I,
)

FALLBACK_FIELD = {
    "id": "fallback",
    "data_key": "fallback",
    "label": "Upload Fallback Image [ .png / .jpg / .jpeg ]",
    "type": "image",
    "top_level": True,
}


def _parse_create_request(text: str) -> dict:
    result = {}

    m = _TEMPLATE_ID_PATTERN.search(text)
    if m:
        result["format_id"] = m.group(1)

    m = _CAMPAIGN_NAME_STRICT_PATTERN.search(text)
    if m:
        result["campaign_name"] = m.group(1).strip()

    m = _CREATIVE_NAME_PATTERN.search(text)
    if m:
        result["creative_name"] = m.group(1).strip()

    if "format_id" not in result and "format_name" not in result:
        m = _FORMAT_KEYWORD_PATTERN.search(text)
        if m:
            candidate = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip()
            if candidate:
                result["format_name"] = candidate

    if "format_id" not in result and "format_name" not in result:
        m2 = _CREATE_PATTERN.search(text)
        if m2:
            format_name = re.sub(r"^(a|an)\s+", "", m2.group(1).strip(), flags=re.I)
            if format_name and format_name.lower() not in ("a", "an"):
                result["format_name"] = format_name
            if "campaign_name" not in result and re.search(r"\bcampaign\b", text, re.I):
                campaign_name = re.sub(r"^(a|an)\s+", "", m2.group(2).strip(), flags=re.I)
                result["campaign_name"] = campaign_name

    return result


class CreativeAgent:
    def __init__(self, mcp: MCPClient, ollama: OllamaClient):
        self.mcp = mcp
        self.ollama = ollama
        self._ollama_tools = mcp_tools_to_ollama(mcp.list_tools())

    def refresh_tools(self):
        self._ollama_tools = mcp_tools_to_ollama(self.mcp.list_tools(force_refresh=True))

    def _call_tool_retrying(self, name: str, args: dict, retries: int = 1):
        """Wraps self.mcp.call_tool with one automatic retry on failure.
        Network blips against the MCP tunnel are common and usually
        transient -- a single retry costs a fraction of a second and
        saves the user from having to manually resend the same message
        every time there's a momentary hiccup."""
        last_error = None
        for attempt in range(retries + 1):
            try:
                return self.mcp.call_tool(name, args)
            except Exception as e:
                last_error = e
        raise last_error

    def _last_assistant_text(self, history: list) -> str:
        for m in reversed(history[:-1]):
            if m.get("role") == "assistant":
                return m.get("content", "") or ""
        return ""

    # ---- creatives list follow-up + simple list fast path ----

    def _list_creatives_for_campaign_name(self, campaign_name: str):
        """Returns (reply_text, raw_items_list_or_None)."""
        answer = _clean_name_answer(campaign_name)
        if answer.lower() in ("all", "everything", "any", "all of them"):
            try:
                raw = self._call_tool_retrying("list_creatives", {"limit": 20})
                data = _safe_json(raw)
                items = data.get("data", [])
                return _format_list_reply(items), items
            except Exception as e:
                return f"Couldn't fetch creatives — the tool failed with: {e}", None

        try:
            raw = self.mcp.call_tool("list_campaigns", {"name": answer, "limit": 5})
            matches = _safe_json(raw).get("data", [])
        except Exception as e:
            return f"Couldn't look up that campaign — the tool failed with: {e}", None

        if not matches:
            return f"I couldn't find a campaign matching \"{answer}\". Could you check the name and try again?", None

        if len(matches) > 1:
            names = ", ".join(m.get("cname", "Unnamed") for m in matches)
            return f"I found a few campaigns matching that: {names}. Which one did you mean?", None

        campaign = matches[0]
        campaign_id = campaign.get("campaignid")
        try:
            raw = self.mcp.call_tool("list_creatives", {"campaignid": campaign_id, "limit": 20})
            data = _safe_json(raw)
            items = data.get("data", [])
            reply = _format_list_reply(
                items,
                empty_msg=f"\"{campaign.get('cname')}\" doesn't have any creatives yet.",
            )
            return reply, items
        except Exception as e:
            return f"Couldn't fetch creatives for that campaign — the tool failed with: {e}", None

    def _try_fast_path(self, user_text: str, history: list):
        if self._last_assistant_text(history) == PENDING_CREATIVES_QUESTION:
            reply, items = self._list_creatives_for_campaign_name(user_text)
            history.append({"role": "assistant", "content": reply})
            wizard = {"last_creatives": items} if items else {}
            return reply, history, wizard

        # --- who am I / current user (get_user) ---
        if _WHOAMI_PATTERN.search(user_text):
            try:
                raw = self.mcp.call_tool("get_user", {})
                data = _safe_json(raw)
                u = data.get("data", data) if isinstance(data, dict) else {}
                name = " ".join(filter(None, [u.get("first_name"), u.get("last_name")])).strip() or "unknown"
                email = u.get("email", "unknown")
                company = u.get("company", "unknown")
                admin = "yes" if u.get("admin") else "no"
                reply = f"You're logged in as {name} ({email}) — company: {company}, org admin: {admin}."
            except Exception as e:
                reply = f"I couldn't fetch that — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- rename/update campaign (check before delete-campaign, since
        # both mention "campaign" and could otherwise collide) ---
        rename_match = _RENAME_CAMPAIGN_PATTERN.search(user_text)
        if rename_match:
            old_name, new_name = rename_match.group(1).strip(), rename_match.group(2).strip()
            campaign, err = self._find_campaign(old_name)
            if err:
                reply = err
            else:
                try:
                    raw = self.mcp.call_tool(
                        "update_campaign", {"id": campaign["campaignid"], "cname": new_name}
                    )
                    _safe_json(raw)
                    reply = f"Done — renamed campaign \"{old_name}\" to \"{new_name}\"."
                except Exception as e:
                    reply = f"Couldn't rename that campaign — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- delete campaign ---
        delete_match = _DELETE_CAMPAIGN_PATTERN.search(user_text)
        if delete_match and not _CREATE_INTENT_PATTERN.search(user_text):
            name = (delete_match.group(1) or delete_match.group(2) or "").strip()
            campaign, err = self._find_campaign(name)
            if err:
                reply = err
            else:
                try:
                    raw = self.mcp.call_tool("delete_campaign", {"id": campaign["campaignid"]})
                    data = _safe_json(raw)
                    status = str(data.get("status", (data.get("data") or {}).get("status", "")))
                    if status in ("1", "True", "true"):
                        reply = f"Deleted campaign \"{campaign.get('cname', name)}\"."
                    else:
                        reply = (
                            f"I tried to delete \"{campaign.get('cname', name)}\" but the API didn't "
                            f"confirm success. Raw response: {json.dumps(data)[:200]}"
                        )
                except Exception as e:
                    reply = f"Couldn't delete that campaign — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- delete creative ---
        delete_creative_match = _DELETE_CREATIVE_PATTERN.search(user_text)
        if delete_creative_match and not _CREATE_INTENT_PATTERN.search(user_text):
            name = (delete_creative_match.group(1) or delete_creative_match.group(2) or "").strip()
            campaign_match = _CAMPAIGN_NAME_PATTERN.search(user_text)
            campaign_name = _campaign_match_text(campaign_match)
            payload, campaign, err, needs_campaign = self._find_existing_creative(name, campaign_name)
            if err:
                reply = err
            else:
                creativeid = payload.get("creativeid")
                try:
                    raw = self.mcp.call_tool("delete_creative", {"id": creativeid})
                    data = _safe_json(raw)
                    status = str(data.get("status", (data.get("data") or {}).get("status", "")))
                    if status in ("1", "True", "true"):
                        reply = f"Deleted creative \"{payload.get('cname', name)}\"."
                    else:
                        reply = (
                            f"I tried to delete \"{payload.get('cname', name)}\" but the API didn't "
                            f"confirm success. Raw response: {json.dumps(data)[:200]}"
                        )
                except Exception as e:
                    reply = f"Couldn't delete that creative — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- campaign details lookup (checked before the generic
        # creatives/list patterns below, since phrasing like "give me
        # details of the ShobhikaTest campaign" would otherwise get
        # swallowed by the generic "give...campaign" list pattern) ---
        if _CAMPAIGN_DETAIL_PATTERN.search(user_text) and not _CREATE_INTENT_PATTERN.search(user_text):
            name = _extract_name_near_keyword(user_text, "campaign")
            if name:
                campaign, err = self._find_campaign(name)
                if err:
                    reply = err
                else:
                    reply = (
                        f"Campaign: {campaign.get('cname', '?')}\n"
                        f"Advertiser: {campaign.get('advertiser', '?')}\n"
                        f"Type: {campaign.get('ctype', '?')}\n"
                        f"Campaign ID: {campaign.get('campaignid', '?')}\n"
                        f"Start: {campaign.get('start', 'n/a')}  End: {campaign.get('end', 'n/a')}"
                    )
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}
            # couldn't pull a specific name out -- fall through rather
            # than guessing

        # --- creative details lookup (same reasoning as campaign details
        # above -- must run before the generic creatives-list pattern) ---
        if _CREATIVE_DETAIL_PATTERN.search(user_text) and not _CREATE_INTENT_PATTERN.search(user_text):
            name = _extract_name_near_keyword(user_text, "creative")
            if name:
                campaign_match = _CAMPAIGN_NAME_PATTERN.search(user_text)
                campaign_name = _campaign_match_text(campaign_match)
                payload, campaign, err, needs_campaign = self._find_existing_creative(name, campaign_name)
                if err:
                    reply = err
                else:
                    data_obj = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
                    reply = (
                        f"Creative: {payload.get('cname', '?')}\n"
                        f"Campaign: {campaign.get('cname', '?') if campaign else '?'}\n"
                        f"Creative ID: {payload.get('creativeid', '?')}\n"
                        f"Format: {payload.get('formatname', payload.get('formatid', '?'))}\n"
                        f"Dimensions: {payload.get('width', '?')}x{payload.get('height', '?')}\n"
                        f"Last updated: {payload.get('updated_at', payload.get('modified', 'n/a'))}"
                    )
                    if data_obj:
                        reply += "\nFields: " + ", ".join(f"{k}={v}" for k, v in list(data_obj.items())[:8])
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}
            # couldn't pull a specific name out -- fall through rather
            # than guessing

        # --- "the Nth creative (in campaign X)" -- ordinal position
        # lookup. If no campaign is named in THIS message, we ask rather
        # than guess or silently fall into the AI loop (which has no
        # reliable way to count positions either). ---
        nth_match = _ORDINAL_INDEX_PATTERN.search(user_text)
        if nth_match and not _CREATE_INTENT_PATTERN.search(user_text):
            n = int(nth_match.group(1))
            campaign_match = _CAMPAIGN_NAME_PATTERN.search(user_text)
            campaign_name = _campaign_match_text(campaign_match)
            if not campaign_name:
                reply = (
                    f"Which campaign's creative list should I count #{n} from? "
                    f"(e.g. \"the {n}{_ordinal_suffix(n)} creative in ShobhikaTest\")"
                )
            else:
                campaign, err = self._find_campaign(campaign_name)
                if err:
                    reply = err
                else:
                    try:
                        raw = self.mcp.call_tool(
                            "list_creatives", {"campaignid": campaign["campaignid"], "limit": max(n, 20)}
                        )
                        items = _safe_json(raw).get("data", [])
                        if len(items) < n:
                            reply = f"\"{campaign.get('cname')}\" only has {len(items)} creatives, so there's no #{n}."
                        else:
                            target = items[n - 1]
                            reply = f"#{n} in \"{campaign.get('cname')}\" is \"{target.get('cname', 'Unnamed')}\"."
                    except Exception as e:
                        reply = f"Couldn't fetch that — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- template ID -> name lookup (e.g. "which template is id 208") ---
        template_id_match = _TEMPLATE_ID_LOOKUP_PATTERN.search(user_text)
        if template_id_match and not _CREATE_INTENT_PATTERN.search(user_text):
            format_id = template_id_match.group(1) or template_id_match.group(2)
            template, err = self._find_template(format_id=format_id)
            if err:
                reply = err
            else:
                name = template.get("name") or template.get("formatname", "Unknown")
                ttype = template.get("type", template.get("formattype", "?"))
                width = template.get("width", "?")
                height = template.get("height", "?")
                reply = f"Template ID {format_id} is \"{name}\" ({ttype}), {width}x{height}."
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        # --- template name -> ID lookup (e.g. "what's the id for the
        # Quiz-n-Win template") -- only reached if no digit-based lookup
        # matched above, so this and the block above don't collide ---
        if _TEMPLATE_NAME_TO_ID_PATTERN.search(user_text) and not _CREATE_INTENT_PATTERN.search(user_text):
            name = _extract_name_near_keyword(user_text, "template")
            if name:
                template, err = self._find_template(format_name=name)
                if err:
                    reply = err  # already lists name+ID options if ambiguous
                else:
                    formatid = template.get("formatid") or template.get("id", "?")
                    tname = template.get("name", name)
                    reply = f"\"{tname}\" has template ID {formatid}."
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}

        if _CREATIVES_PATTERN.search(user_text) and not _CREATE_INTENT_PATTERN.search(user_text):
            # if the user already named a campaign right in this message
            # (e.g. "list creatives under ShobhikaTest campaign"), don't
            # ask again -- just look it up directly.
            campaign_match = _CAMPAIGN_NAME_PATTERN.search(user_text)
            if campaign_match:
                reply, items = self._list_creatives_for_campaign_name(_campaign_match_text(campaign_match))
                history.append({"role": "assistant", "content": reply})
                wizard = {"last_creatives": items} if items else {}
                return reply, history, wizard
            history.append({"role": "assistant", "content": PENDING_CREATIVES_QUESTION})
            return PENDING_CREATIVES_QUESTION, history, {}

        for pattern, tool_name in _LIST_PATTERNS:
            if pattern.search(user_text) and not _CREATE_INTENT_PATTERN.search(user_text):
                try:
                    raw = self.mcp.call_tool(tool_name, {"limit": 20})
                    data = _safe_json(raw)
                    reply = _format_list_reply(data.get("data", []))
                except Exception as e:
                    reply = f"I couldn't fetch that — the tool failed with: {e}"
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}

        return None

    # ---- creative-creation wizard ----

    def _extract_required_fields(self, template_result: dict) -> list:
        raw_fields = (
            template_result.get("template")
            or template_result.get("fields")
            or (template_result.get("data") or {}).get("template")
            or []
        )
        fields = []
        if isinstance(raw_fields, list):
            for f in raw_fields:
                if not isinstance(f, dict):
                    continue
                field_id = f.get("id") or f.get("key") or f.get("name")
                if not field_id:
                    continue
                if field_id.lower() == "cname":
                    continue
                if "enable" in f:
                    continue

                is_required = None
                for key_name in ("required", "isRequired", "mandatory"):
                    if key_name in f:
                        is_required = bool(f[key_name])
                        break
                if is_required is None and "optional" in f:
                    is_required = not bool(f["optional"])
                if is_required is None:
                    is_required = True
                if not is_required:
                    continue

                model = f.get("model", "")
                top_level = False
                if model.startswith("obj.data."):
                    data_key = model[len("obj.data."):]
                elif model == "obj.cname":
                    continue
                elif model.startswith("obj."):
                    # anything under "obj." but not "obj.data." is a
                    # top-level create_creative argument (like "fallback"),
                    # not something nested inside "data".
                    data_key = model[len("obj."):]
                    top_level = True
                else:
                    data_key = field_id

                label = f.get("title") or f.get("label") or f.get("info") or field_id
                field_type = f.get("type", "text")
                fields.append({
                    "id": field_id,
                    "data_key": data_key,
                    "label": label,
                    "type": field_type,
                    "top_level": top_level,
                })

        # Airtory Studio always requires a fallback image for interactive
        # creatives. Some templates already declare this themselves under
        # whatever internal key name they use -- check by LABEL TEXT
        # ("fallback" appears in the title), not by exact key name, since
        # different templates name this differently internally.
        already_has_fallback = any("fallback" in _normalize(f["label"]) for f in fields)
        if not already_has_fallback:
            fields.append(dict(FALLBACK_FIELD))
        return fields

    def _fetch_all_templates(self) -> list:
        """Airtory's get_templates caps at 100 results per call, but there
        are 577+ templates total -- paginate through all pages so a search
        can find ANY template, not just ones in the first batch."""
        all_templates = []
        offset = 0
        page_size = 100
        while True:
            try:
                raw = self.mcp.call_tool("get_templates", {"limit": page_size, "offset": offset})
                data = _safe_json(raw)
            except Exception:
                break
            page = data.get("data", [])
            all_templates.extend(page)
            total_count = data.get("count")
            offset += page_size
            if len(page) < page_size:
                break
            if total_count is not None and offset >= total_count:
                break
            if offset > 2000:  # safety cap against runaway pagination
                break
        return all_templates

    def _find_template(self, format_id: str = None, format_name: str = None):
        # ID-based lookup bypasses Airtory's broken pagination entirely --
        # fetch that ONE template directly via get_template, no list needed.
        if format_id:
            try:
                raw = self.mcp.call_tool("get_template", {"id": str(format_id)})
                detail = _safe_json(raw)
            except Exception as e:
                return None, f"Couldn't look up template ID {format_id} — the tool failed with: {e}"
            payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
            if payload and (payload.get("name") or payload.get("formatname")):
                if "formatid" not in payload:
                    payload["formatid"] = format_id
                return payload, None
            return None, f"I couldn't find a template with ID {format_id}. Try \"list templates\" to see what's available."

        try:
            all_templates = self._fetch_all_templates()
        except Exception as e:
            return None, f"Couldn't look up ad formats — the tool failed with: {e}"

        needle = _normalize(format_name)
        matches = [t for t in all_templates if needle in _normalize(t.get("name", ""))]

        if not matches:
            return None, f"I couldn't find an ad format matching \"{format_name}\". Try \"list templates\" to see what's available."

        if len(matches) > 1:
            # Multiple templates share this exact name -- rather than
            # blocking the flow with a "which ID?" question, auto-pick
            # the first match but say so plainly, since duplicates can
            # have genuinely different dimensions/behavior and a silent
            # guess risks handing back a functionally wrong creative.
            chosen = dict(matches[0])
            alts = matches[1:]
            if alts:
                alt_list = ", ".join(
                    f"ID {t.get('formatid') or t.get('id')} ({t.get('width')}x{t.get('height')})"
                    for t in alts
                )
                chosen["_auto_picked_note"] = (
                    f"Heads up — a few formats are named \"{format_name}\"; I went with "
                    f"ID {chosen.get('formatid') or chosen.get('id')} "
                    f"({chosen.get('width')}x{chosen.get('height')}). Other sizes that also "
                    f"go by this name: {alt_list}. If that's not the one you meant, let me know "
                    f"and I'll switch to a specific ID."
                )
            return chosen, None

        return matches[0], None

    def _find_campaign(self, campaign_name: str):
        clean = _clean_name_answer(campaign_name)
        try:
            raw = self.mcp.call_tool("list_campaigns", {"name": clean, "limit": 5})
            campaigns = _safe_json(raw).get("data", [])
        except Exception as e:
            return None, f"Couldn't look up that campaign — the tool failed with: {e}"

        if not campaigns:
            return None, f"I couldn't find a campaign matching \"{clean}\". Could you check the name, or should I create a new campaign for it?"
        if len(campaigns) > 1:
            names = ", ".join(c.get("cname", "Unnamed") for c in campaigns)
            return None, f"I found a few campaigns matching that: {names}. Which one did you mean?"
        return campaigns[0], None

    def _start_new_campaign_flow(self, parsed: dict, purpose: str = "creative_flow", trigger_text: str = ""):
        partial_data = {}
        pre_filled_name = _extract_campaign_name_from_trigger(trigger_text) if trigger_text else None
        if pre_filled_name:
            partial_data["cname"] = pre_filled_name

        wizard_state = {
            "active": True,
            "stage": "awaiting_new_campaign_details",
            "parsed": parsed,
            "partial_data": partial_data,
            "purpose": purpose,
        }
        lines = ["Sure, let's set up a new campaign. Reply with one per line:"]
        if not pre_filled_name:
            lines.append("- Campaign Name: <value>")
        else:
            lines[0] = f"Sure, let's set up a new campaign called \"{pre_filled_name}\". Just need:"
        lines.append("- Advertiser Name: <value>")
        lines.append("(Optional) Campaign Type (video or display, defaults to display): <value>")
        return "\n".join(lines), wizard_state

    def _finish_new_campaign_flow(self, user_text: str, wizard: dict):
        answers = self._parse_field_answers(user_text)
        data = dict(wizard.get("partial_data", {}))
        for key, value in answers.items():
            key_norm = _normalize(key)
            if "advertiser" in key_norm:
                data["advertiser"] = value
            elif "type" in key_norm:
                data["ctype"] = value.lower()
            elif "campaign" in key_norm or "name" in key_norm:
                data["cname"] = value

        missing = []
        if not data.get("cname"):
            missing.append("Campaign Name")
        if not data.get("advertiser"):
            missing.append("Advertiser Name")

        if missing:
            new_wizard = dict(wizard)
            new_wizard["partial_data"] = data
            lines = ["Got it, saved. I still need:"] + [f"- {m}: <value>" for m in missing]
            return "\n".join(lines), new_wizard

        ctype = data.get("ctype") if data.get("ctype") in ("video", "display") else "display"

        try:
            raw = self.mcp.call_tool(
                "create_campaign",
                {"advertiser": data["advertiser"], "cname": data["cname"], "ctype": ctype},
            )
            result = _safe_json(raw)
        except Exception as e:
            return f"Couldn't create the campaign — the tool failed with: {e}", {}

        if not (result.get("status") == 1 or result.get("data")):
            return f"Something went wrong creating the campaign: {json.dumps(result)[:300]}", {}

        if wizard.get("purpose") == "standalone":
            return (
                f"Done! Created campaign \"{data['cname']}\" (advertiser: {data['advertiser']}). "
                "Want me to build a creative in it now, or is that all for now?",
                {},
            )

        # resume the original creative-creation flow, now that the
        # campaign exists -- look it up fresh so we get its real ID
        parsed = dict(wizard.get("parsed", {}))
        parsed["campaign_name"] = data["cname"]
        reply, new_wizard = self._start_creative_wizard(parsed)
        reply = f"Created campaign \"{data['cname']}\"! {reply}"
        return reply, new_wizard

    def _match_option(self, text: str, options: list):
        """Match user input against a list of (key, label) options, by
        number (1-indexed) or by fuzzy label text. Returns the matched
        (key, label) tuple, or None."""
        text = text.strip()
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return None
        needle = _normalize(text)
        if not needle:
            return None
        for key, label in options:
            norm_label = _normalize(label)
            if needle in norm_label or norm_label in needle:
                return (key, label)
        best, best_ratio = None, 0.0
        for key, label in options:
            ratio = difflib.SequenceMatcher(None, needle, _normalize(label)).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (key, label)
        return best if best_ratio >= 0.6 else None

    def _show_format_list(self, type_key, type_label, size_key, size_label, wizard):
        """Shared by both the normal ad-size flow and the auto-skip path
        (when a type has only one size option) -- builds the Ad Format
        list reply and the wizard state for it."""
        formats = format_names(type_key, size_key)
        new_wizard = dict(wizard)
        new_wizard["ad_type_key"] = type_key
        new_wizard["ad_type_label"] = type_label
        new_wizard["ad_size_key"] = size_key
        new_wizard["ad_size_label"] = size_label

        if not formats:
            new_wizard["stage"] = "awaiting_format"
            return (
                f"I don't have the exact format list for {type_label} mapped yet — "
                f"no worries though, just tell me the template name or ID directly, "
                f"or ask me to \"list templates\" and we'll find it together.",
                new_wizard,
            )

        new_wizard["stage"] = "awaiting_catalog_format"
        lines = [f"Nice choice! Here's what's available under {type_label}:"]
        for i, (name, _fid) in enumerate(formats, start=1):
            lines.append(f"{i}. {name}")
        lines.append("Which one catches your eye?")
        return "\n".join(lines), new_wizard

    def _handle_ad_type_selection(self, text: str, wizard: dict):
        options = type_options()
        matched = self._match_option(text, options)
        if not matched:
            lines = ["Hmm, I didn't quite catch that — could you pick one of these?"]
            for i, (_, label) in enumerate(options, start=1):
                lines.append(f"{i}. {label}")
            return "\n".join(lines), wizard

        type_key, type_label = matched
        sizes = size_options(type_key)
        new_wizard = dict(wizard)
        new_wizard["ad_type_key"] = type_key
        new_wizard["ad_type_label"] = type_label

        if not sizes:
            # No catalog data for this type yet -- fall back to the
            # legacy direct name/ID entry instead of blocking the user.
            new_wizard["stage"] = "awaiting_format"
            return (
                f"{type_label} sounds great! I don't have the exact Studio breakdown for "
                f"it mapped yet, though — just tell me the template name or ID directly, "
                f"or ask me to \"list templates\" and we'll figure it out together.",
                new_wizard,
            )

        if len(sizes) == 1:
            # Only one real option -- Studio likely doesn't even show a
            # size-selection step for this type, so don't make the user
            # answer a question with a single possible answer.
            size_key, size_label = sizes[0]
            return self._show_format_list(type_key, type_label, size_key, size_label, new_wizard)

        new_wizard["stage"] = "awaiting_ad_size"
        lines = [f"{type_label} it is! What size are we thinking?"]
        for i, (_, label) in enumerate(sizes, start=1):
            lines.append(f"{i}. {label}")
        return "\n".join(lines), new_wizard

    def _handle_ad_size_selection(self, text: str, wizard: dict):
        type_key = wizard.get("ad_type_key")
        sizes = size_options(type_key)
        matched = self._match_option(text, sizes)
        if not matched:
            lines = ["Hmm, I didn't quite catch that — could you pick one of these?"]
            for i, (_, label) in enumerate(sizes, start=1):
                lines.append(f"{i}. {label}")
            return "\n".join(lines), wizard

        size_key, size_label = matched
        return self._show_format_list(
            type_key, wizard.get("ad_type_label"), size_key, size_label, wizard
        )

    def _lookup_format_candidates_by_name(self, format_name: str):
        """Returns every catalog template matching this name, without
        picking one -- used to build an accurate 'which one' question
        when Studio's OWN displayed list contains the same name twice."""
        try:
            all_templates = self._fetch_all_templates()
        except Exception:
            return []
        needle = _normalize(format_name)
        return [t for t in all_templates if needle in _normalize(t.get("name", ""))]

    def _handle_ambiguous_format_id(self, text: str, wizard: dict):
        id_match = re.search(r"\b(\d+)\b", text)
        if not id_match:
            return "Please reply with just the ID number.", wizard
        parsed = dict(wizard.get("parsed", {}))
        parsed["format_id"] = id_match.group(1)
        return self._start_creative_wizard(parsed)

    def _handle_catalog_format_selection(self, text: str, wizard: dict):
        type_key = wizard.get("ad_type_key")
        size_key = wizard.get("ad_size_key")
        formats = format_names(type_key, size_key)

        stripped = text.strip()
        # options for _match_option need a (key, label) shape -- use the
        # display name for both, matching against position index below
        options = [(name, name) for name, _fid in formats]
        matched = self._match_option(stripped, options)
        if not matched:
            if _SUGGEST_ME_PATTERN.search(stripped) and formats:
                # go with whatever Studio lists first for this bucket -- a
                # reasonable, deterministic default rather than leaving the
                # user stuck re-reading the same list
                suggested_name, suggested_id = formats[0]
                parsed = dict(wizard.get("parsed", {}))
                if suggested_id:
                    parsed["format_id"] = suggested_id
                else:
                    parsed["format_name"] = suggested_name
                    parsed["_suppress_auto_pick_note"] = True
                reply, new_wizard = self._start_creative_wizard(parsed)
                return f"Sure — {suggested_name} is a solid pick, let's go with that!\n\n{reply}", new_wizard

            lines = ["I didn't catch that — please pick one:"]
            for i, (name, _fid) in enumerate(formats, start=1):
                lines.append(f"{i}. {name}")
            return "\n".join(lines), wizard

        picked_by_number = stripped.isdigit()

        if picked_by_number:
            # A number always refers to one specific position -- that's
            # inherently unambiguous even if its name repeats elsewhere
            # in this same list, so resolve directly, no question.
            idx = int(stripped) - 1
            chosen_name, chosen_formatid = formats[idx] if 0 <= idx < len(formats) else (matched[0], None)
        else:
            # Picked by name/text -- if Studio's OWN displayed list (the
            # one the user is actually looking at right now) contains
            # this exact name more than once, that's a genuine
            # ambiguity we can't silently resolve by just taking the
            # first match, regardless of whether we happen to know both
            # entries' exact IDs. Ask, using the real per-position IDs
            # we have for this bucket instead of guessing across the
            # wider catalog.
            chosen_name = matched[0]
            same_name_entries = [
                (i, name, fid) for i, (name, fid) in enumerate(formats, start=1)
                if _normalize(name) == _normalize(chosen_name)
            ]
            if len(same_name_entries) > 1:
                lines = [f"Studio lists \"{chosen_name}\" more than once here — which one did you mean?"]
                candidate_ids = []
                for i, name, fid in same_name_entries:
                    if fid:
                        detail, err = self._find_template(format_id=fid)
                        dims = f" ({detail.get('width')}x{detail.get('height')})" if detail else ""
                        lines.append(f"- ID {fid}{dims}")
                        candidate_ids.append(fid)
                    else:
                        lines.append(f"- position {i} in the list above (ID not yet known)")
                lines.append("Reply with the ID.")
                new_wizard = dict(wizard)
                new_wizard["stage"] = "awaiting_ambiguous_format_id"
                return "\n".join(lines), new_wizard

            chosen_formatid = same_name_entries[0][2] if same_name_entries else None

        parsed = dict(wizard.get("parsed", {}))
        if chosen_formatid:
            # Exact ID known and unambiguous -- skip name-based lookup
            # entirely.
            parsed["format_id"] = chosen_formatid
        else:
            # Unique within Studio's displayed list but no exact ID
            # captured yet -- resolve via name lookup, even if
            # _find_template's own broader catalog search still finds
            # same-named entries elsewhere. Suppress its auto-pick note
            # here since, from the user's perspective in THIS flow,
            # there was nothing ambiguous to begin with.
            parsed["format_name"] = chosen_name
            parsed["_suppress_auto_pick_note"] = True
        # Continue the normal creative-creation flow -- reuses all the
        # existing lookup/field logic unchanged.
        return self._start_creative_wizard(parsed)

    def _start_creative_wizard(self, parsed: dict):
        format_id = parsed.get("format_id")
        format_name = parsed.get("format_name")
        campaign_name = parsed.get("campaign_name")
        creative_name = parsed.get("creative_name")

        if not campaign_name:
            wizard_state = {"active": True, "stage": "awaiting_campaign", "parsed": parsed}
            return (
                "Heyy, that's exciting — let's build this together! If you want to add it to "
                "an existing campaign, just tell me its name. Otherwise, let me know if you'd "
                "like to set up a new one.",
                wizard_state,
            )

        if not format_id and not format_name:
            wizard_state = {"active": True, "stage": "awaiting_ad_type", "parsed": parsed}
            lines = ["What kind of ad would you like to build?"]
            for i, (key, label) in enumerate(type_options(), start=1):
                lines.append(f"{i}. {label}")
            return "\n".join(lines), wizard_state

        template_summary, err = self._find_template(format_id=format_id, format_name=format_name)
        if err:
            wizard_state = {
                "active": True,
                "stage": "awaiting_format",
                "parsed": {k: v for k, v in parsed.items() if k not in ("format_id", "format_name")},
            }
            return err, wizard_state

        formatid = str(template_summary.get("formatid") or template_summary.get("id"))

        campaign, err = self._find_campaign(campaign_name)
        if err:
            wizard_state = {
                "active": True,
                "stage": "awaiting_campaign",
                "parsed": {k: v for k, v in parsed.items() if k != "campaign_name"},
            }
            return err, wizard_state
        campaignid = campaign.get("campaignid")

        try:
            raw = self.mcp.call_tool("get_template", {"id": formatid})
            template_detail = _safe_json(raw)
        except Exception as e:
            return f"Couldn't load that template's details — the tool failed with: {e}", {}

        required_fields = self._extract_required_fields(template_detail)

        wizard_state = {
            "active": True,
            "stage": "awaiting_fields",
            "formatid": formatid,
            "format_name": template_summary.get("name", format_name or formatid),
            "campaignid": campaignid,
            "campaign_name": campaign.get("cname", campaign_name),
            "required_fields": required_fields,
            "cname": creative_name,
            "partial_data": {},
            "auto_picked": bool(template_summary.get("_auto_picked_note")) and not parsed.get("_suppress_auto_pick_note"),
        }

        lines = [f"Love it — a {wizard_state['format_name']} creative for {wizard_state['campaign_name']}, coming right up!"]
        auto_pick_note = template_summary.get("_auto_picked_note")
        if auto_pick_note and not parsed.get("_suppress_auto_pick_note"):
            lines.append(auto_pick_note)
        lines.append("Just need a few details from you — reply with one per line, like \"Field: value\" "
                      "(Shift+Enter for a new line, then send when you're ready):")
        if not creative_name:
            lines.append("- Creative Name: <a name for this creative>")
        for f in required_fields:
            hint = " <image URL>" if f["type"] == "image" else " <value>"
            lines.append(f"- {f['label']}:{hint}")

        return "\n".join(lines), wizard_state

    def _parse_field_answers(self, text: str, known_labels: list = None) -> dict:
        """Parse 'Field: value' lines. If known_labels is given, each line
        is matched against those EXACT labels first (as a prefix) -- this
        avoids misparsing lines where the label itself happens to contain
        a colon somewhere in its own wording (which broke naive first-colon
        splitting and corrupted saved data)."""
        known_labels = known_labels or []
        # try longer labels first so a short label can't accidentally
        # prefix-match before a longer, more specific one does
        sorted_labels = sorted(known_labels, key=len, reverse=True)

        answers = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line_wo_dash = line.lstrip("-").strip()

            matched_label = None
            for label in sorted_labels:
                if line_wo_dash.lower().startswith(label.lower()):
                    matched_label = label
                    break

            if matched_label:
                remainder = line_wo_dash[len(matched_label):].lstrip()
                if remainder.startswith(":"):
                    remainder = remainder[1:].lstrip()
                if remainder:
                    answers[matched_label] = remainder
                continue

            # fallback for anything that didn't match a known label
            m = re.match(r"^([^:]+?):\s+(.+)$", line_wo_dash)
            if m:
                key = m.group(1).strip()
                value = m.group(2).strip()
                if key and value:
                    answers[key] = value

        return answers

    def _match_field(self, key: str, required_fields: list) -> dict:
        key_norm = _normalize(key)
        MIN_LABEL_LEN = 4
        MIN_ID_LEN = 6  # short ids like "back" can accidentally collide as
                        # substrings of unrelated words (e.g. "fallback"),
                        # so id-based matching needs a longer minimum and
                        # only runs as an absolute last resort.

        # Pass 0: EXACT label equality always wins outright, before any
        # substring/containment logic runs. This is what stops something
        # like "Button Text" from losing to "Button Text Color" just
        # because the shorter name happens to be a prefix of the longer
        # one -- an exact match is never ambiguous.
        for f in required_fields:
            if _normalize(f["label"]) == key_norm:
                return f

        # Pass 1: match using the FULL label, brackets included. This is
        # what correctly tells apart fields that only differ by bracketed
        # info, like "Overlay Image [520 X 780]" vs "[780 X 520]" -- if we
        # strip the brackets too early, both look identical.
        exact_candidates = []
        for f in required_fields:
            full_label_norm = _normalize(f["label"])
            if full_label_norm and (full_label_norm in key_norm or key_norm in full_label_norm):
                exact_candidates.append(f)
        if len(exact_candidates) == 1:
            return exact_candidates[0]
        if len(exact_candidates) > 1:
            # more than one full-label match is unusual; prefer the most
            # specific (longest) label as the best guess.
            return max(exact_candidates, key=lambda f: len(_normalize(f["label"])))

        # Pass 2: fuzzy match with brackets stripped -- fine for fields
        # whose bracket content doesn't matter for disambiguation.
        stripped_candidates = []
        for f in required_fields:
            label_core = _normalize(re.sub(r"\[.*?\]", "", f["label"]))
            if len(label_core) >= MIN_LABEL_LEN and (label_core in key_norm or key_norm in label_core):
                stripped_candidates.append(f)
        if len(stripped_candidates) == 1:
            return stripped_candidates[0]
        if len(stripped_candidates) > 1:
            # multiple fields share the same generic label (e.g. two
            # "Overlay Image" fields at different sizes) -- try to
            # disambiguate using whatever's inside the brackets on both
            # sides (e.g. "520 X 780") before giving up.
            key_brackets = _normalize("".join(re.findall(r"\[(.*?)\]", key)))
            if key_brackets:
                for f in stripped_candidates:
                    field_brackets = _normalize("".join(re.findall(r"\[(.*?)\]", f["label"])))
                    if field_brackets and field_brackets == key_brackets:
                        return f
            # still ambiguous -- don't guess wrong, leave it unmatched so
            # it stays in the "still need" list instead of silently
            # overwriting a different field's answer.
            return None

        # Pass 3: id-based fallback, only if nothing matched by label at all.
        for f in required_fields:
            id_norm = _normalize(f["id"])
            if len(id_norm) >= MIN_ID_LEN and (id_norm in key_norm or key_norm in id_norm):
                return f

        return None

    def _find_existing_creative(self, creative_name: str, campaign_name: str = None):
        """Returns (creative_detail_payload, campaign_dict, error_message,
        needs_campaign: bool). If campaign_name isn't given, searches by
        name across the whole org first -- Airtory's list_creatives
        supports that directly. Only asks for a campaign if the name is
        genuinely ambiguous (found in more than one campaign)."""
        campaign = None
        args = {"name": _clean_name_answer(creative_name), "limit": 10}

        if campaign_name:
            campaign, err = self._find_campaign(campaign_name)
            if err:
                return None, None, err, False
            args = {"campaignid": campaign.get("campaignid"), "name": _clean_name_answer(creative_name), "limit": 5}

        try:
            raw = self.mcp.call_tool("list_creatives", args)
            matches = _safe_json(raw).get("data", [])
        except Exception as e:
            return None, None, f"Couldn't look up that creative — the tool failed with: {e}", False

        if not matches:
            scope = f" in \"{campaign.get('cname')}\"" if campaign else ""
            return None, None, (
                f"I couldn't find a creative matching \"{creative_name}\"{scope}. "
                f"Try \"list creatives\" to check the name."
            ), False

        if len(matches) > 1 and not campaign_name:
            campaign_names = {
                m["campaign"]["cname"]
                for m in matches
                if isinstance(m.get("campaign"), dict) and m["campaign"].get("cname")
            }
            if len(campaign_names) > 1:
                names = ", ".join(sorted(campaign_names))
                return None, None, (
                    f"I found \"{creative_name}\" in a few different campaigns: {names}. "
                    f"Which one did you mean?"
                ), True
            # all matches are actually in the same campaign (duplicates by
            # name within it) -- just proceed with the first.
        elif len(matches) > 1:
            names = ", ".join(m.get("cname", "Unnamed") for m in matches)
            return None, None, f"I found a few creatives matching that: {names}. Which one did you mean?", False

        creative_summary = matches[0]
        creativeid = creative_summary.get("creativeid") or creative_summary.get("id")

        try:
            raw = self.mcp.call_tool("get_creative", {"id": creativeid})
            detail = _safe_json(raw)
        except Exception as e:
            return None, None, f"Found the creative, but couldn't load its details — the tool failed with: {e}", False

        payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
        if "creativeid" not in payload:
            payload["creativeid"] = creativeid

        if campaign is None:
            embedded = payload.get("campaign", {})
            campaign = {
                "campaignid": embedded.get("campaignid"),
                "cname": embedded.get("cname", "Unknown campaign"),
            }

        return payload, campaign, None, False

    def _lookup_existing_creative_and_offer(self, creative_name: str, campaign_name: str, wants: str):
        payload, campaign, err, needs_campaign = self._find_existing_creative(creative_name, campaign_name)
        if needs_campaign:
            wizard_state = {
                "active": True,
                "stage": "awaiting_action_campaign_name",
                "parsed_action": {"action": "preview_tag", "creative_name": creative_name, "campaign_name": None, "wants": wants},
            }
            return err, wizard_state
        if err:
            return err, {}

        fake_wizard = {
            "formatid": str(payload.get("formatid")),
            "format_name": payload.get("formatname", "this format"),
            "campaignid": campaign.get("campaignid"),
            "campaign_name": campaign.get("cname", campaign_name),
            "cname": payload.get("cname", creative_name),
            "data": payload.get("data", {}),
            "creativeid": payload.get("creativeid"),
        }
        return self._handle_post_create_offer(wants, fake_wizard)

    def _duplicate_existing_creative(self, creative_name: str, campaign_name: str):
        payload, campaign, err, needs_campaign = self._find_existing_creative(creative_name, campaign_name)
        if needs_campaign:
            wizard_state = {
                "active": True,
                "stage": "awaiting_action_campaign_name",
                "parsed_action": {"action": "duplicate", "creative_name": creative_name, "campaign_name": None},
            }
            return err, wizard_state
        if err:
            return err, {}

        try:
            raw = self.mcp.call_tool(
                "duplicate_creative",
                {"id": payload.get("creativeid"), "campaign": {"campaignid": campaign.get("campaignid")}},
            )
            result = _safe_json(raw)
        except Exception as e:
            return f"Couldn't duplicate that creative — the tool failed with: {e}", {}

        if result.get("status") == 1 or result.get("data"):
            new_data = result.get("data", result)
            new_name = new_data.get("cname", f"Copy of {creative_name}")
            return (
                f"Done! Duplicated \"{creative_name}\" as \"{new_name}\" in "
                f"\"{campaign.get('cname')}\".",
                {},
            )
        return f"Something went wrong duplicating the creative: {json.dumps(result)[:300]}", {}

    def _start_edit_creative_flow(self, creative_name: str, campaign_name: str):
        payload, campaign, err, needs_campaign = self._find_existing_creative(creative_name, campaign_name)
        if needs_campaign:
            wizard_state = {
                "active": True,
                "stage": "awaiting_action_campaign_name",
                "parsed_action": {"action": "edit", "creative_name": creative_name, "campaign_name": None},
            }
            return err, wizard_state
        if err:
            return err, {}

        formatid = str(payload.get("formatid"))
        try:
            raw = self.mcp.call_tool("get_template", {"id": formatid})
            template_detail = _safe_json(raw)
        except Exception as e:
            return f"Found the creative, but couldn't load its editable fields — the tool failed with: {e}", {}

        all_fields = self._extract_all_fields(template_detail)
        current_data = payload.get("data", {})

        wizard_state = {
            "active": True,
            "stage": "awaiting_edit_fields",
            "creativeid": payload.get("creativeid"),
            "campaignid": campaign.get("campaignid"),
            "cname": payload.get("cname", creative_name),
            "all_fields": all_fields,
            "current_data": current_data,
        }

        lines = [
            f"Editing \"{wizard_state['cname']}\". Here are its current values -- "
            "tell me which field(s) to change and to what, one per line (\"Field: new value\"):",
        ]
        for f in all_fields[:20]:
            current_val = current_data.get(f["data_key"], "<not set>")
            lines.append(f"- {f['label']}: currently \"{current_val}\"")
        if len(all_fields) > 20:
            lines.append(f"...and {len(all_fields) - 20} more fields not shown.")

        return "\n".join(lines), wizard_state

    def _finish_edit_creative_flow(self, user_text: str, wizard: dict):
        all_fields = wizard.get("all_fields", [])
        known_labels = ["Creative Name"] + [f["label"] for f in all_fields]
        answers = self._parse_field_answers(user_text, known_labels=known_labels)

        if not answers:
            return (
                "I couldn't read any changes from that — reply with \"Field: new value\", "
                "one per line.",
                wizard,
            )

        new_cname = None
        updates = {}
        for key, value in answers.items():
            if key.lower() in ("creative name", "name", "cname"):
                new_cname = value
                continue
            matched = self._match_field(key, all_fields)
            if matched:
                updates[matched["data_key"]] = value

        if not updates and not new_cname:
            return (
                "I couldn't match those to any real field on this creative — "
                "please use the exact field names shown above.",
                wizard,
            )

        merged_data = {**wizard.get("current_data", {}), **updates}

        try:
            call_args = {"id": wizard["creativeid"], "data": merged_data}
            if new_cname:
                call_args["cname"] = new_cname
            raw = self.mcp.call_tool("update_creative", call_args)
            result = _safe_json(raw)
        except Exception as e:
            return f"Couldn't update the creative — the tool failed with: {e}", {}

        if result.get("status") == 1 or result.get("data"):
            changed = ", ".join(updates.keys()) or "the creative name"
            return f"Updated! Changed: {changed}.", {}
        return f"Something went wrong updating the creative: {json.dumps(result)[:300]}", {}

    def _extract_all_fields(self, template_result: dict) -> list:
        """Like _extract_required_fields but keeps every field, not just
        required ones -- used for editing, where any field is fair game."""
        raw_fields = (
            template_result.get("template")
            or template_result.get("fields")
            or (template_result.get("data") or {}).get("template")
            or []
        )
        fields = []
        if isinstance(raw_fields, list):
            for f in raw_fields:
                if not isinstance(f, dict):
                    continue
                field_id = f.get("id") or f.get("key") or f.get("name")
                if not field_id or field_id.lower() == "cname":
                    continue

                model = f.get("model", "")
                top_level = False
                if model.startswith("obj.data."):
                    data_key = model[len("obj.data."):]
                elif model == "obj.cname":
                    continue
                elif model.startswith("obj."):
                    data_key = model[len("obj."):]
                    top_level = True
                else:
                    data_key = field_id

                label = f.get("title") or f.get("label") or f.get("info") or field_id
                field_type = f.get("type", "text")
                fields.append({
                    "id": field_id,
                    "data_key": data_key,
                    "label": label,
                    "type": field_type,
                    "top_level": top_level,
                })
        return fields

    def _finish_creative_wizard(self, user_text: str, wizard: dict):
        # If an earlier ambiguous-name auto-pick offered "let me know and
        # I'll switch to a specific ID", this is where that promise
        # actually gets honored -- only recognized when this wizard came
        # from an auto-pick, so a legitimately-typed field value can
        # never be misread as a correction request.
        if wizard.get("auto_picked"):
            correction_match = _USE_ID_CORRECTION_PATTERN.search(user_text)
            if correction_match:
                try:
                    parsed = {
                        "campaign_name": wizard.get("campaign_name"),
                        "creative_name": wizard.get("cname"),
                        "format_id": correction_match.group(1),
                    }
                    return self._start_creative_wizard(parsed)
                except Exception as e:
                    return (
                        f"Something went wrong switching to ID {correction_match.group(1)} — "
                        f"{e}. You can try again, or just tell me the ID once more.",
                        wizard,
                    )

        known_labels = ["Creative Name"] + [f["label"] for f in wizard.get("required_fields", [])]
        answers = self._parse_field_answers(user_text, known_labels=known_labels)

        cname = wizard.get("cname")
        new_data = {}
        new_top_level = {}
        unmatched_keys = []
        for key, value in answers.items():
            if key.lower() in ("creative name", "name", "cname"):
                cname = value
                continue
            matched = self._match_field(key, wizard.get("required_fields", []))
            if matched and matched.get("top_level"):
                new_top_level[matched["data_key"]] = value
            elif matched:
                new_data[matched["data_key"]] = value
            else:
                # don't silently store this under the raw typed key -- it's
                # not a real field, so keep it out of the payload and flag
                # it to the user instead of quietly discarding their answer.
                unmatched_keys.append(key)

        data = {**wizard.get("partial_data", {}), **new_data}
        top_level = {**wizard.get("partial_top_level", {}), **new_top_level}

        if not answers and not cname:
            return (
                "I couldn't read any field values from that — please reply with one "
                "per line, like \"Field: value\" (make sure there's a space after the colon).",
                {**wizard, "partial_data": data, "partial_top_level": top_level},
            )

        if not cname:
            return (
                "I still need a Creative Name — what should this creative be called?",
                {**wizard, "partial_data": data, "partial_top_level": top_level},
            )

        required_fields = wizard.get("required_fields", [])
        required_keys = {
            (f["data_key"], f.get("top_level", False)) for f in required_fields
        }
        have_keys = {(k, False) for k in data.keys()} | {(k, True) for k in top_level.keys()}
        missing_fields = [
            f for f in required_fields
            if (f["data_key"], f.get("top_level", False)) not in have_keys
        ]

        if missing_fields or unmatched_keys:
            new_wizard = dict(wizard)
            new_wizard["cname"] = cname
            new_wizard["partial_data"] = data
            new_wizard["partial_top_level"] = top_level
            lines = ["Got it, saved."]
            if unmatched_keys:
                bad = ", ".join(f"\"{k}\"" for k in unmatched_keys)
                lines.append(
                    f"Heads up — I couldn't match {bad} to any real field on this "
                    "template, so that value wasn't saved. Please use the exact "
                    "field label shown below."
                )
            if missing_fields:
                lines.append("I still need:")
                for f in missing_fields:
                    lines.append(f"- {f['label']}: <value>")
            return "\n".join(lines), new_wizard

        try:
            call_args = {
                "formatid": wizard["formatid"],
                "cname": cname,
                "campaign": {"campaignid": wizard["campaignid"]},
                "data": data,
            }
            call_args.update(top_level)  # e.g. "fallback"
            raw = self.mcp.call_tool("create_creative", call_args)
            result = _safe_json(raw)
        except Exception as e:
            return f"The creative wasn't created — the tool failed with: {e}", {}

        if result.get("status") == 1 or result.get("data"):
            creativeid = _find_first_id(result, keys=("creativeid", "id"))
            new_wizard = {
                "active": True,
                "stage": "post_create_offer",
                "formatid": wizard["formatid"],
                "format_name": wizard["format_name"],
                "campaignid": wizard["campaignid"],
                "campaign_name": wizard["campaign_name"],
                "cname": cname,
                "data": data,
                "creativeid": creativeid,
            }
            reply = (
                f"Done! Created \"{cname}\" ({wizard['format_name']}) in "
                f"{wizard['campaign_name']}.\n\n"
                "Would you like:\n"
                "1. A quick preview link, or\n"
                "2. The full embeddable ad tag (I'll need a few placement details for that)\n"
                "Reply \"preview\", \"tag\", or \"no thanks\"."
            )
            return reply, new_wizard
        else:
            reply = f"Something went wrong creating the creative: {json.dumps(result)[:300]}"
            return reply, {}

    # ---- post-creation: preview link or full ad tag ----

    def _handle_post_create_offer(self, user_text: str, wizard: dict):
        choice = user_text.strip().lower()
        wants_preview = "preview" in choice
        wants_tag = "tag" in choice or "embed" in choice or "ins" in choice

        if not wants_preview and not wants_tag:
            if choice in ("no", "no thanks", "nope", "nah", "no thank you"):
                return "No problem! Let me know if you need anything else.", {}
            return (
                "Sorry, I didn't catch that — reply \"preview\", \"tag\", or \"no thanks\".",
                wizard,
            )

        if not wizard.get("creativeid") or not wizard.get("campaignid"):
            return (
                "I don't have a real creative/campaign ID to work with here, so I can't "
                "generate a working preview or tag. Try \"list creatives\" to find the "
                "real one and ask again.",
                {},
            )

 
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=30)
        try:
            raw = self.mcp.call_tool(
                "create_placement",
                {
                    "name": f"{wizard['cname']} Placement",
                    "publisher": "Airtory Studio",
                    "start": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "end": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "campaign": {"campaignid": wizard["campaignid"]},
                    "creativeid": wizard["creativeid"],
                },
            )
            placement_result = _safe_json(raw)
        except Exception as e:
            return f"Couldn't set up a placement — the tool failed with: {e}", {}

        placementid = _find_first_id(placement_result, keys=("placementid", "id"))
        if not placementid:
            return (
                f"Placement may have been created, but I couldn't find its ID in the "
                f"response: {json.dumps(placement_result)[:300]}",
                {},
            )

        try:
            raw = self.mcp.call_tool("get_placement_preview", {"id": placementid})
            preview_result = _safe_json(raw)
        except Exception as e:
            return f"Placement created, but couldn't generate its preview/tag — the tool failed with: {e}", {}

        payload = preview_result.get("data") if isinstance(preview_result.get("data"), dict) else preview_result
        preview_url = payload.get("preview")
        preview_tag = payload.get("tag") or payload.get("embed")

        # Airtory Studio's "Ins Tag" repeats the CREATIVE ID in both
        # attributes -- confirmed directly against Studio's own UI output.
        # It isn't something get_placement or get_placement_preview
        # actually returns, which is why hunting for it in their responses
        # kept falling back to the messy raw <script>var adTag=... snippet
        # instead. Build it directly.
        production_tag = _build_ins_tag(wizard["creativeid"])

        # Still check get_placement in case some creative type genuinely
        # returns a different/updated tag format -- prefer that if present,
        # but never fail back to the messy snippet just because this call
        # errors or comes back empty; we already have a correct tag above.
        try:
            raw2 = self.mcp.call_tool("get_placement", {"id": placementid})
            placement_detail = _safe_json(raw2)
            api_tag = _find_ins_tag(placement_detail)
            if api_tag:
                production_tag = api_tag
        except Exception:
            pass

        lines = []
        if wants_preview:
            lines.append(
                f"Preview link: {preview_url}" if preview_url else "Didn't get a preview URL back."
            )
        if wants_tag:
            if production_tag:
                # production_tag was already decoded/verified above.
                lines.append(f"Ad tag:\n\n{production_tag}")
            elif preview_tag:
                decoded = _decode_ad_tag(preview_tag)
                lines.append(f"Embed tag:\n\n{decoded}")
            else:
                lines.append("Didn't get an embed tag back.")

        if not lines:
            lines.append(f"Didn't get anything usable back. Raw response: {json.dumps(preview_result)[:300]}")

        return "\n\n".join(lines), {}

    def _finish_placement_wizard(self, user_text: str, wizard: dict):
        defaults = wizard.get("defaults", {})
        overrides = {}

        confirm_words = ("ok", "okay", "yes", "proceed", "sure", "go ahead", "sounds good", "yep", "yeah", "confirm")
        if user_text.strip().lower() not in confirm_words:
            answers = self._parse_field_answers(user_text)
            for key, value in answers.items():
                key_norm = _normalize(key)
                if "publisher" in key_norm:
                    overrides["publisher"] = value
                elif "start" in key_norm:
                    overrides["start"] = value
                elif "end" in key_norm:
                    overrides["end"] = value
                elif "name" in key_norm:
                    overrides["name"] = value

        final = {**defaults, **overrides}

        try:
            raw = self.mcp.call_tool(
                "create_placement",
                {
                    "name": final["name"],
                    "publisher": final["publisher"],
                    "start": final["start"],
                    "end": final["end"],
                    "campaign": {"campaignid": wizard["campaignid"]},
                    "creativeid": wizard["creativeid"],
                },
            )
            result = _safe_json(raw)
        except Exception as e:
            return f"Couldn't create the placement — the tool failed with: {e}", {}

        placementid = _find_first_id(result, keys=("placementid", "id"))
        if not placementid:
            return (
                f"Placement created, but I couldn't find its ID in the response: "
                f"{json.dumps(result)[:300]}",
                {},
            )
        production_tag = _build_ins_tag(wizard["creativeid"])
        return f"Placement created! Here's your ad tag (ready to paste into a page):\n\n{production_tag}", {}

    # ---- grounded template/field info lookups (no AI guessing) ----

    def _answer_template_field_question(self, format_id: str, user_text: str) -> str:
        try:
            raw = self.mcp.call_tool("get_template", {"id": format_id})
            detail = _safe_json(raw)
        except Exception as e:
            return f"Couldn't look up format {format_id} — the tool failed with: {e}"

        payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
        template_name = payload.get("name", f"format {format_id}")
        raw_fields = payload.get("template", [])
        if not isinstance(raw_fields, list) or not raw_fields:
            return f"I looked up {template_name} (ID {format_id}) but couldn't read its field list."

        words = [
            w for w in re.findall(r"[a-z]+", user_text.lower())
            if w not in _STOPWORDS and len(w) >= 4
        ]

        best_field, best_score = None, 0
        for f in raw_fields:
            if not isinstance(f, dict):
                continue
            title_norm = _normalize(f.get("title") or f.get("label") or f.get("id", ""))
            score = sum(1 for w in words if w in title_norm)
            if score > best_score:
                best_field, best_score = f, score

        if not best_field or best_score == 0:
            names = ", ".join(f.get("title", f.get("id", "?")) for f in raw_fields if isinstance(f, dict))
            return f"I found {template_name} (ID {format_id}), but couldn't tell which field you meant. Its fields are: {names}."

        lines = [f"In {template_name} (ID {format_id}), \"{best_field.get('title', best_field.get('id'))}\":"]
        if "info" in best_field:
            lines.append(f"- {best_field['info']}")
        if "type" in best_field:
            lines.append(f"- Type: {best_field['type']}")
        if "min" in best_field or "max" in best_field:
            lines.append(f"- Range: {best_field.get('min', '?')} to {best_field.get('max', '?')}")
        if "step" in best_field:
            lines.append(f"- Step: {best_field['step']}")
        if "required" in best_field:
            lines.append(f"- Required: {'yes' if best_field['required'] else 'no'}")

        return "\n".join(lines)

    # ---- main entry point ----

    def _answer_count_question(self, entity: str, campaign_name: str = None):
        entity = entity.lower().rstrip("s")  

        if entity == "campaign":
            try:
                raw = self.mcp.call_tool("list_campaigns", {"limit": 1})
                data = _safe_json(raw)
            except Exception as e:
                return f"Couldn't check that — the tool failed with: {e}"
            count = data.get("count")
            return f"There are {count} campaigns in total." if count is not None else "I couldn't determine the total campaign count."

        if entity == "creative":
            args = {"limit": 1}
            scope = "in total"
            if campaign_name:
                campaign, err = self._find_campaign(campaign_name)
                if err:
                    return err
                args["campaignid"] = campaign.get("campaignid")
                scope = f"in \"{campaign.get('cname')}\""
            try:
                raw = self.mcp.call_tool("list_creatives", args)
                data = _safe_json(raw)
            except Exception as e:
                return f"Couldn't check that — the tool failed with: {e}"
            count = data.get("count")
            return f"There are {count} creatives {scope}." if count is not None else f"I couldn't determine the creative count {scope}."

        if entity == "placement":
            args = {"limit": 1}
            scope = "in total"
            if campaign_name:
                campaign, err = self._find_campaign(campaign_name)
                if err:
                    return err
                args["campaignid"] = campaign.get("campaignid")
                scope = f"in \"{campaign.get('cname')}\""
            try:
                raw = self.mcp.call_tool("list_placements", args)
                data = _safe_json(raw)
            except Exception as e:
                return f"Couldn't check that — the tool failed with: {e}"
            count = data.get("count")
            return f"There are {count} placements {scope}." if count is not None else f"I couldn't determine the placement count {scope}."

        if entity in ("template", "format"):
            try:
                raw = self.mcp.call_tool("get_templates", {"limit": 1})
                data = _safe_json(raw)
            except Exception as e:
                return f"Couldn't check that — the tool failed with: {e}"
            count = data.get("count")
            return f"There are {count} ad templates available." if count is not None else "I couldn't determine the template count."

        if entity == "asset":
            try:
                raw = self.mcp.call_tool("get_assets", {"limit": 1})
                data = _safe_json(raw)
            except Exception as e:
                return f"Couldn't check that — the tool failed with: {e}"
            count = data.get("count")
            return f"There are {count} assets in total." if count is not None else "I couldn't determine the asset count."

        return None

    def _advance_creative_action(self, parsed_action: dict):
        action = parsed_action.get("action")
        creative_name = parsed_action.get("creative_name")
        campaign_name = parsed_action.get("campaign_name")

        if not creative_name:
            wizard_state = {
                "active": True,
                "stage": "awaiting_action_creative_name",
                "parsed_action": parsed_action,
            }
            return "Which creative would you like to work with?", wizard_state

        # campaign_name may be None here -- that's fine. Each lookup tries
        # searching by creative name alone first, and only asks for a
        # campaign if the name turns out to be ambiguous.
        if action == "preview_tag":
            return self._lookup_existing_creative_and_offer(
                creative_name, campaign_name, parsed_action.get("wants", "preview and tag")
            )
        if action == "duplicate":
            return self._duplicate_existing_creative(creative_name, campaign_name)
        if action == "edit":
            return self._start_edit_creative_flow(creative_name, campaign_name)

        return "Sorry, I'm not sure what to do with that — could you rephrase?", {}

    _ORDINAL_WORD_MAP = {
        "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
    }

    def _resolve_ordinal_index(self, text: str):
        text_l = text.lower()
        for word, idx in self._ORDINAL_WORD_MAP.items():
            if re.search(rf"\b{word}\b", text_l):
                return idx
        m = re.search(r"\b(?:number\s*|#)(\d+)\b", text_l)
        if m:
            return int(m.group(1)) - 1
        return None

    def _handle_ordinal_reference(self, last_user_msg: str, wizard: dict, history: list):
        """If the user references a position in a creatives list we just
        showed them (e.g. \"the first one\"), resolve it directly and act
        on it -- no name-based lookup needed since we already have the
        real ID from that list."""
        items = wizard.get("last_creatives")
        if not items:
            return None
        ordinal_idx = self._resolve_ordinal_index(last_user_msg)
        wants_action = re.search(r"\b(preview|tag|duplicate|edit|update|modify|change)\b", last_user_msg, re.I)
        if ordinal_idx is None or not wants_action:
            return None

        try:
            item = items[ordinal_idx]
        except IndexError:
            reply = f"I only have {len(items)} in that list — which position did you mean?"
            history.append({"role": "assistant", "content": reply})
            return reply, history, wizard

        creativeid = item.get("creativeid") or item.get("id")
        try:
            raw = self.mcp.call_tool("get_creative", {"id": creativeid})
            detail = _safe_json(raw)
        except Exception as e:
            reply = f"Couldn't load that creative's details — the tool failed with: {e}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
        if "creativeid" not in payload:
            payload["creativeid"] = creativeid
        embedded_campaign = payload.get("campaign", {}) or {}
        campaign = {
            "campaignid": embedded_campaign.get("campaignid"),
            "cname": embedded_campaign.get("cname", "Unknown campaign"),
        }

        msg_l = last_user_msg.lower()

        if "duplicate" in msg_l:
            try:
                raw2 = self.mcp.call_tool(
                    "duplicate_creative",
                    {"id": payload["creativeid"], "campaign": {"campaignid": campaign["campaignid"]}},
                )
                result = _safe_json(raw2)
            except Exception as e:
                reply = f"Couldn't duplicate that creative — the tool failed with: {e}"
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}
            if result.get("status") == 1 or result.get("data"):
                new_name = result.get("data", result).get("cname", f"Copy of {payload.get('cname')}")
                reply = f"Done! Duplicated \"{payload.get('cname')}\" as \"{new_name}\" in \"{campaign['cname']}\"."
            else:
                reply = f"Something went wrong duplicating the creative: {json.dumps(result)[:300]}"
            history.append({"role": "assistant", "content": reply})
            return reply, history, {}

        if any(w in msg_l for w in ("edit", "update", "modify", "change")):
            formatid = str(payload.get("formatid"))
            try:
                raw2 = self.mcp.call_tool("get_template", {"id": formatid})
                template_detail = _safe_json(raw2)
            except Exception as e:
                reply = f"Couldn't load that template's fields — the tool failed with: {e}"
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}
            all_fields = self._extract_all_fields(template_detail)
            current_data = payload.get("data", {})
            new_wizard = {
                "active": True,
                "stage": "awaiting_edit_fields",
                "creativeid": payload["creativeid"],
                "campaignid": campaign["campaignid"],
                "cname": payload.get("cname"),
                "all_fields": all_fields,
                "current_data": current_data,
            }
            lines = [f"Editing \"{payload.get('cname')}\". Reply with \"Field: new value\", one per line:"]
            for f in all_fields[:20]:
                lines.append(f"- {f['label']}: currently \"{current_data.get(f['data_key'], '<not set>')}\"")
            reply = "\n".join(lines)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        wants = "preview and tag" if "preview" in msg_l and "tag" in msg_l else ("preview" if "preview" in msg_l else "tag")
        fake_wizard = {
            "formatid": str(payload.get("formatid")),
            "format_name": payload.get("formatname", "this format"),
            "campaignid": campaign["campaignid"],
            "campaign_name": campaign["cname"],
            "cname": payload.get("cname"),
            "data": payload.get("data", {}),
            "creativeid": payload["creativeid"],
        }
        reply, new_wizard = self._handle_post_create_offer(wants, fake_wizard)
        history.append({"role": "assistant", "content": reply})
        return reply, history, new_wizard

    def run(self, history: list, wizard: dict = None):
        wizard = wizard or {}

        last_user_msg = ""
        for m in reversed(history):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        ordinal_result = self._handle_ordinal_reference(last_user_msg, wizard, history)
        if ordinal_result is not None:
            return ordinal_result

     
        _STALE_OVERRIDABLE_STAGES = {
            "post_create_offer",
            "awaiting_placement_fields",
            "awaiting_new_campaign_details",
            "awaiting_campaign",
            "awaiting_action_campaign_name",
            "awaiting_action_creative_name",
            "awaiting_format",
            "awaiting_ad_type",
            "awaiting_ad_size",
            "awaiting_catalog_format",
            "awaiting_ambiguous_format_id",
        }
        if wizard.get("active") and wizard.get("stage") in _STALE_OVERRIDABLE_STAGES:
            if _looks_like_fresh_start_intent(last_user_msg):
                wizard = {}

        if wizard.get("active") and wizard.get("stage") == "awaiting_fields":
            reply, new_wizard = self._finish_creative_wizard(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "post_create_offer":
            reply, new_wizard = self._handle_post_create_offer(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_placement_fields":
            reply, new_wizard = self._finish_placement_wizard(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_new_campaign_details":
            reply, new_wizard = self._finish_new_campaign_flow(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_campaign":
            if _NEW_CAMPAIGN_INTENT_PATTERN.search(last_user_msg):
                parsed = dict(wizard.get("parsed", {}))
                reply, new_wizard = self._start_new_campaign_flow(parsed, trigger_text=last_user_msg)
                history.append({"role": "assistant", "content": reply})
                return reply, history, new_wizard
            parsed = dict(wizard.get("parsed", {}))
            parsed["campaign_name"] = last_user_msg.strip()
            reply, new_wizard = self._start_creative_wizard(parsed)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_ad_type":
            reply, new_wizard = self._handle_ad_type_selection(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_ad_size":
            reply, new_wizard = self._handle_ad_size_selection(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_catalog_format":
            reply, new_wizard = self._handle_catalog_format_selection(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_ambiguous_format_id":
            reply, new_wizard = self._handle_ambiguous_format_id(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_format":
            # if they're asking to SEE the list rather than naming a format
            # (e.g. "list templates"), show it and keep waiting right here.
            if re.search(r"\b(list|show|see)\b.*\b(templates?|formats?)\b", last_user_msg, re.I):
                fast_result = self._try_fast_path(last_user_msg, history)
                if fast_result is not None:
                    reply, history, _fast_wizard = fast_result
                    reply += "\n\nWhich one would you like to use?"
                    history[-1]["content"] = reply
                    return reply, history, wizard

            parsed = dict(wizard.get("parsed", {}))
            id_match = re.search(r"\b(\d+)\b", last_user_msg)
            if id_match:
                parsed["format_id"] = id_match.group(1)
            else:
                
                kw_match = _FORMAT_KEYWORD_PATTERN.search(last_user_msg)
                if kw_match:
                    parsed["format_name"] = (kw_match.group(1) or kw_match.group(2) or "").strip()
                elif len(last_user_msg.split()) <= 5 and "campaign" not in last_user_msg.lower():
                    parsed["format_name"] = last_user_msg.strip()
                else:
                    reply = (
                        "I just need the ad format/template name (e.g. \"Quiz-n-Win\" or "
                        "\"Peel to Reveal\") — what would you like to use?"
                    )
                    history.append({"role": "assistant", "content": reply})
                    return reply, history, wizard
            reply, new_wizard = self._start_creative_wizard(parsed)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if _looks_like_campaign_start_intent(last_user_msg) and not _CREATE_INTENT_PATTERN.search(last_user_msg):
            reply, new_wizard = self._start_new_campaign_flow({}, purpose="standalone", trigger_text=last_user_msg)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_edit_fields":
            reply, new_wizard = self._finish_edit_creative_flow(last_user_msg, wizard)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_action_creative_name":
            parsed_action = dict(wizard.get("parsed_action", {}))
            parsed_action["creative_name"] = last_user_msg.strip()
            reply, new_wizard = self._advance_creative_action(parsed_action)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if wizard.get("active") and wizard.get("stage") == "awaiting_action_campaign_name":
            parsed_action = dict(wizard.get("parsed_action", {}))
            parsed_action["campaign_name"] = last_user_msg.strip()
            reply, new_wizard = self._advance_creative_action(parsed_action)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        parsed_action = _parse_creative_action_request(last_user_msg)
        if parsed_action:
            reply, new_wizard = self._advance_creative_action(parsed_action)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        if _CREATE_INTENT_PATTERN.search(last_user_msg):
            parsed = _parse_create_request(last_user_msg)
            reply, new_wizard = self._start_creative_wizard(parsed)
            history.append({"role": "assistant", "content": reply})
            return reply, history, new_wizard

        count_match = _COUNT_PATTERN.search(last_user_msg)
        if count_match:
            entity = count_match.group(1)
            campaign_match = _CAMPAIGN_NAME_PATTERN.search(last_user_msg)
            campaign_name = _campaign_match_text(campaign_match)
            answer = self._answer_count_question(entity, campaign_name)
            if answer:
                history.append({"role": "assistant", "content": answer})
                return answer, history, {}

        fast_result = self._try_fast_path(last_user_msg, history)
        if fast_result is not None:
            reply, history, fast_wizard = fast_result
            return reply, history, fast_wizard

        if not _CREATE_INTENT_PATTERN.search(last_user_msg):
            id_mention = _TEMPLATE_MENTION_PATTERN.search(last_user_msg)
            if id_mention:
                info_reply = self._answer_template_field_question(id_mention.group(1), last_user_msg)
                history.append({"role": "assistant", "content": info_reply})
                return info_reply, history, {}

        recent_history = history[-10:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent_history

        for _ in range(MAX_TOOL_ROUNDS):
            result = _call_ollama_with_timeout(self.ollama, messages, self._ollama_tools)

            if result is None:
                reply = (
                    "That's taking longer than it should, and I'd rather tell you that "
                    "than leave you waiting — this might be an action I can't handle "
                    "reliably yet. I can currently create/edit creatives and campaigns, "
                    "list campaigns/creatives/templates/placements/assets, generate "
                    "preview links and ad tags, and answer template field questions. "
                    "Could you try one of those, or rephrase the request?"
                )
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}

            msg = result.get("message", {})
            tool_calls = msg.get("tool_calls")

            print(f"[agent] model replied. tool_calls={bool(tool_calls)}")

            if not tool_calls:
                reply = msg.get("content", "")
                history.append({"role": "assistant", "content": reply})
                return reply, history, {}

            messages.append(msg)
            history.append(msg)

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                print(f"[agent] calling tool: {name} with {args}")

                try:
                    tool_result = self.mcp.call_tool(name, args)
                    tool_result = _trim_tool_result(str(tool_result))
                except Exception as e:
                    tool_result = f"Tool '{name}' failed: {e}"

                print(f"[agent] tool result: {str(tool_result)[:500]}")

                tool_msg = {"role": "tool", "content": str(tool_result), "name": name}
                messages.append(tool_msg)
                history.append(tool_msg)

        fallback = (
            "I'm having trouble finishing that with the tools available — "
            "could you rephrase or simplify the request?"
        )
        history.append({"role": "assistant", "content": fallback})
        return fallback, history, {}
