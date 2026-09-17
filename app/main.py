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


app = gr.mount_gradio_app(
    app,
    build_ui(service),
    path="/ui",
    theme=APP_THEME,
    css=APP_CSS,
    footer_links=[],
)
