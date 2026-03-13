"""
Equity Research Agent — FastAPI web server.

Endpoints
---------
GET  /                  → serves static/index.html
GET  /api/health        → health check
POST /api/research      → SSE stream of research events
POST /api/reset         → reset conversation history
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Equity Research Agent")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount /static for any extra assets (JS, CSS files if added later)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Agent (one instance per server process — stateful conversation)
# ---------------------------------------------------------------------------

from equity_agent.agent import EquityResearchAgent  # noqa: E402

_agent = EquityResearchAgent(
    api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    model=os.environ.get("AGENT_MODEL", "claude-opus-4-6"),
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model": _agent.model}


class ResearchRequest(BaseModel):
    query: str


@app.post("/api/research")
def research(req: ResearchRequest):
    """Stream research events as newline-delimited JSON (NDJSON) via SSE."""

    def event_stream():
        for event in _agent.research_events(req.query):
            # SSE format: "data: <json>\n\n"
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/api/reset")
def reset():
    _agent.reset()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)
