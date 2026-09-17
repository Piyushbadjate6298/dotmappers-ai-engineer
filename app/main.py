from __future__ import annotations

import os
from typing import Literal

# Ensure Gradio runs cleanly in read-only serverless environments (Vercel)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")

import gradio as gr
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import config
from .services import SupportTicketService
from .ui import APP_CSS, APP_THEME, build_ui

app = FastAPI(
    title="SupportScope API",
    version="1.1.0",
    description="Natural-language support-ticket analytics with explainable anomaly detection.",
)
service = SupportTicketService()


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "healthy",
        "tickets_loaded": int(len(service.df)),
        "data_reference_time": service.reference_time.isoformat(),
        "configured_llm_provider": service.planner.provider,
        "configured_llm_model": service.planner.model,
        "heuristic_fallback_enabled": config.ALLOW_HEURISTIC_FALLBACK,
    }


@app.get("/stats", tags=["analytics"])
def stats() -> dict:
    return service.stats()


@app.post("/query", tags=["ai"])
def query_tickets(payload: QueryRequest) -> dict:
    try:
        return service.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The query could not be processed.") from exc


@app.get("/anomalies", tags=["analytics"])
def anomalies(
    scope: Literal["all", "this_week", "this_month"] = Query(default="all"),
    unresolved_hours: float = Query(default=24.0, gt=0, le=720),
) -> dict:
    try:
        return service.anomalies(scope=scope, unresolved_hours=unresolved_hours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


try:
    app = gr.mount_gradio_app(
        app,
        build_ui(service),
        path="/ui",
        theme=APP_THEME,
        css=APP_CSS,
        footer_links=[],
    )
except Exception:
    from fastapi.responses import HTMLResponse

    @app.get("/ui", response_class=HTMLResponse, tags=["ui"], include_in_schema=False)
    @app.get("/ui/", response_class=HTMLResponse, tags=["ui"], include_in_schema=False)
    def ui_fallback() -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <title>SupportScope — Support Ticket Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; line-height: 1.5; }
        .container { max-width: 900px; margin: 0 auto; }
        .hero { background: linear-gradient(135deg, #1e293b, #312e81); padding: 28px; border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.1); }
        h1 { margin: 0 0 8px; font-size: 32px; color: #fff; }
        p { margin: 0; color: #cbd5e1; font-size: 15px; }
        .card { background: #1e293b; border: 1px solid #334155; padding: 24px; border-radius: 14px; margin-bottom: 20px; }
        input[type="text"] { width: 100%; padding: 14px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #fff; font-size: 15px; box-sizing: border-box; margin-top: 8px; }
        button { background: #4f46e5; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 12px; font-size: 15px; }
        button:hover { background: #4338ca; }
        pre { background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto; color: #38bdf8; font-size: 13px; border: 1px solid #334155; }
        .badge { display: inline-block; background: rgba(99,102,241,0.2); color: #c7d2fe; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-bottom: 12px; }
        a { color: #818cf8; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <span class="badge">DOTMappers AI Engineer Assessment</span>
            <h1>SupportScope Intelligence</h1>
            <p>Natural-language support-ticket analytics & explainable anomaly detection.</p>
        </div>
        <div class="card">
            <h3 style="margin-top:0">Ask the Data</h3>
            <p style="font-size:13px;color:#94a3b8">Ask plain-English questions about the 500-ticket dataset:</p>
            <input type="text" id="q" placeholder="e.g. How many tickets are currently open?" value="How many tickets are currently open?">
            <button onclick="ask()">Run analysis</button>
            <div id="res" style="margin-top:16px;"></div>
        </div>
        <div class="card">
            <h3 style="margin-top:0">Quick Links & Endpoints</h3>
            <p>
                <a href="/docs" target="_blank">Swagger API Docs (/docs)</a> &nbsp;·&nbsp; 
                <a href="/health" target="_blank">Health Status (/health)</a> &nbsp;·&nbsp; 
                <a href="/stats" target="_blank">Dataset Stats (/stats)</a> &nbsp;·&nbsp; 
                <a href="/anomalies" target="_blank">Anomaly Scan (/anomalies)</a>
            </p>
        </div>
    </div>
    <script>
        async function ask() {
            const q = document.getElementById('q').value;
            const resDiv = document.getElementById('res');
            resDiv.innerHTML = '<p style="color:#94a3b8">Processing query...</p>';
            try {
                const res = await fetch('/query', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q}) });
                const data = await res.json();
                resDiv.innerHTML = '<h4 style="color:#34d399;margin:12px 0 8px;">Result: ' + (data.answer || 'No answer') + '</h4><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            } catch(e) {
                resDiv.innerHTML = '<p style="color:#ef4444">Error executing request: ' + e.message + '</p>';
            }
        }
    </script>
</body>
</html>"""

