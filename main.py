"""
main.py
---------
Run this with:  uvicorn main:app --reload --port 8000
Then open:       http://localhost:8000
"""

import os
import uuid

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import CreativeAgent
from mcp_client import MCPClient
from ovh_client import OVHClient

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "https://your-tunnel-url.trycloudflare.com/mcp")
MCP_ARGS_KEY = os.getenv("MCP_ARGS_KEY", "input")
OVH_ENDPOINT_URL = os.getenv(
    "OVH_ENDPOINT_URL",
    "https://gpt-oss-20b.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions",
)
OVH_API_KEY = os.getenv("OVH_API_KEY", "")
OVH_MODEL = os.getenv("OVH_MODEL", "gpt-oss-20b")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

if CLOUDINARY_CLOUD_NAME:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )

app = FastAPI(title="Airtory Creative Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mcp = MCPClient(MCP_URL, args_key=MCP_ARGS_KEY)
mcp.initialize()

ollama = OVHClient(OVH_ENDPOINT_URL, OVH_API_KEY, OVH_MODEL)
agent = CreativeAgent(mcp, ollama)

SESSIONS: dict[str, list] = {}
WIZARDS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(session_id, [])
    wizard = WIZARDS.get(session_id, {})

    history.append({"role": "user", "content": req.message})

    reply, updated_history, updated_wizard = agent.run(history, wizard)

    SESSIONS[session_id] = updated_history
    WIZARDS[session_id] = updated_wizard

    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts an image file from the browser, uploads it to Cloudinary,
    and returns the public URL. The frontend inserts this URL into the
    chat box for the user to attach to whichever field they're filling in.
    """
    if not CLOUDINARY_CLOUD_NAME:
        raise HTTPException(
            status_code=500,
            detail="Cloudinary isn't configured yet. Add CLOUDINARY_CLOUD_NAME, "
                   "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET to your .env file.",
        )

    try:
        contents = await file.read()
        result = cloudinary.uploader.upload(contents, folder="airtory-creative-agent")
        return {"url": result["secure_url"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/reset")
def reset(session_id: str):
    SESSIONS.pop(session_id, None)
    WIZARDS.pop(session_id, None)
    return {"ok": True}


@app.post("/reconnect-mcp")
def reconnect_mcp():
    mcp.initialize()
    agent.refresh_tools()
    return {"ok": True, "tool_count": len(agent._ollama_tools)}


@app.get("/tools")
def tools():
    return mcp.list_tools(force_refresh=True)


@app.get("/debug-creative")
def debug_creative(creativeid: str):
    return mcp.call_tool("get_creative", {"id": creativeid})


@app.get("/debug-templates")
def debug_templates(name: str = ""):
    import json, re
    raw = mcp.call_tool("get_templates", {"limit": 100})
    data = json.loads(str(raw))
    templates = data.get("data", [])
    if name:
        needle = re.sub(r"[^a-z0-9]", "", name.lower())
        templates = [t for t in templates if needle in re.sub(r"[^a-z0-9]", "", t.get("name", "").lower())]
    return {"count": len(templates), "matches": [t.get("name") for t in templates]}


@app.get("/debug-templates-by-type")
def debug_templates_by_type(type: str = "", adsize: str = ""):
    """Tests whether get_templates' documented type/size filters actually
    narrow results server-side (as opposed to name search/pagination,
    which are known broken). Uses the REAL field name "adsize" (not
    "size") as discovered from a raw template object. On a parse
    failure, returns the raw response instead of hiding it, since that's
    exactly the thing we need to see to diagnose what's going on."""
    import json
    args = {"limit": 200}
    if type:
        args["type"] = type
    if adsize:
        args["adsize"] = adsize
    raw = mcp.call_tool("get_templates", args)
    raw_text = str(raw)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "parse_error": str(e),
            "raw_response_first_2000_chars": raw_text[:2000],
        }

    templates = data.get("data", [])
    distinct_types = sorted({t.get("type") for t in templates if t.get("type")})
    distinct_adsizes = sorted({t.get("adsize") for t in templates if t.get("adsize")})
    return {
        "count": len(templates),
        "distinct_types_in_results": distinct_types,
        "distinct_adsizes_in_results": distinct_adsizes,
        "matches": [f"{t.get('name')} [{t.get('type')}/{t.get('adsize')}]" for t in templates],
    }



@app.get("/debug-template-fields")
def debug_template_fields(formatid: str):
    return mcp.call_tool("get_template", {"id": formatid})


@app.get("/debug-raw-name-search")
def debug_raw_name_search(name: str):
    return mcp.call_tool("get_templates", {"name": name})


@app.get("/debug-format-candidates")
def debug_format_candidates(name: str):
    """For a given (possibly ambiguous) format name, returns full details
    -- description, dimensions, use cases -- for every candidate
    template sharing that name. Compare each candidate's description
    against what Studio's UI shows in the right-hand panel when you
    click a specific tile, to identify which formatid corresponds to
    which position in Studio's grid -- no DevTools/Network tab needed."""
    import json, re as _re
    all_templates = agent._fetch_all_templates()
    needle = _re.sub(r"[^a-z0-9]", "", name.lower())
    matches = [
        t for t in all_templates
        if needle in _re.sub(r"[^a-z0-9]", "", t.get("name", "").lower())
    ]

    results = []
    for t in matches:
        fid = t.get("formatid") or t.get("id")
        try:
            raw = mcp.call_tool("get_template", {"id": str(fid)})
            detail = json.loads(str(raw))
            payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
        except Exception as e:
            payload = {"error": str(e)}
        results.append({
            "formatid": fid,
            "name": t.get("name"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "description": payload.get("description"),
            "usecase": payload.get("usecase"),
        })
    return {"count": len(results), "candidates": results}


@app.get("/debug-dump-all-templates")
def debug_dump_all_templates(start: int = 1, end: int = 700):
    """Brute-force sweep of get_template by ID across a range, since
    get_templates' own listing/pagination/name-search are all confirmed
    broken or incomplete. This bypasses all of that by fetching each
    template directly by ID (the one reliably correct path), skipping
    any ID that doesn't resolve to a real template. Writes the full
    result to template_dump.json on disk (next to main.py) so it only
    needs to be run once and can be reused/inspected afterward.

    NOTE: this may take a while -- it's making up to (end - start + 1)
    sequential network calls. That's expected for a one-time sweep.
    """
    import json
    results = []
    misses = 0
    for i in range(start, end + 1):
        try:
            raw = mcp.call_tool("get_template", {"id": str(i)})
            detail = json.loads(str(raw))
            payload = detail.get("data") if isinstance(detail.get("data"), dict) else detail
            name = payload.get("name") or payload.get("formatname")
            if not name:
                misses += 1
                continue
            results.append({
                "formatid": str(i),
                "name": name,
                "width": payload.get("width"),
                "height": payload.get("height"),
                "description": payload.get("description"),
            })
        except Exception:
            misses += 1
            continue

    with open("template_dump.json", "w") as f:
        json.dump(results, f, indent=2)

    return {
        "swept_range": [start, end],
        "found": len(results),
        "missed_or_invalid_ids": misses,
        "saved_to": "template_dump.json (in this backend folder)",
    }


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")