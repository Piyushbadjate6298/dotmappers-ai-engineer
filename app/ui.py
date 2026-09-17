from __future__ import annotations

from html import escape
from typing import Any

import gradio as gr
import pandas as pd

from . import config
from .services import SupportTicketService


SAMPLE_QUESTIONS = [
    ["How many tickets are currently open?"],
    ["Which agent resolved the most tickets this month?"],
    ["Show me all Critical tickets not resolved within 12 hours."],
    ["What is the average customer rating for Technical category tickets?"],
    ["Are there any anomalies in resolution times this week?"],
]

APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
)

APP_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding-bottom: 36px !important;
}
.hero {
    padding: 28px 30px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #312e81 100%);
    color: #ffffff;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.22);
    margin-bottom: 18px;
}
.hero .eyebrow {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #c7d2fe;
    margin-bottom: 10px;
}
.hero h1 {
    margin: 0;
    font-size: clamp(28px, 4vw, 44px);
    line-height: 1.08;
    color: #ffffff;
}
.hero p {
    margin: 12px 0 0;
    max-width: 840px;
    color: #dbeafe;
    font-size: 16px;
    line-height: 1.6;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.16);
    color: #f8fafc;
    font-size: 12px;
    font-weight: 600;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 18px;
}
.kpi-card {
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 16px 17px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .055);
}
.kpi-label {
    color: var(--body-text-color-subdued);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.kpi-value {
    margin-top: 5px;
    color: var(--body-text-color);
    font-size: 28px;
    line-height: 1;
    font-weight: 800;
}
.section-intro {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 14px 16px;
    background: var(--block-background-fill);
    margin-bottom: 8px;
}
.section-intro h3 { margin: 0 0 4px; font-size: 17px; }
.section-intro p { margin: 0; color: var(--body-text-color-subdued); font-size: 13px; line-height: 1.55; }
.answer-card {
    border-radius: 18px;
    border: 1px solid var(--border-color-primary);
    padding: 18px 20px;
    background: var(--block-background-fill);
    min-height: 126px;
}
.answer-card .answer-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    color: #6366f1;
    text-transform: uppercase;
}
.answer-card .answer-text {
    margin-top: 8px;
    font-size: 20px;
    line-height: 1.45;
    font-weight: 700;
    color: var(--body-text-color);
}
.meta-line {
    margin-top: 12px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.status-ok, .status-warn {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 700;
}
.status-ok { background: rgba(16,185,129,.12); color: #059669; }
.status-warn { background: rgba(245,158,11,.14); color: #b45309; }
.anomaly-summary {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 16px 18px;
    background: var(--block-background-fill);
}
.anomaly-summary strong { font-size: 24px; }
.arch-flow {
    display: grid;
    grid-template-columns: repeat(7, auto);
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 22px 10px;
}
.arch-node {
    padding: 12px 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: var(--block-background-fill);
    font-weight: 700;
    font-size: 13px;
    text-align: center;
}
.arch-arrow { color: var(--body-text-color-subdued); font-weight: 800; }
.small-note {
    color: var(--body-text-color-subdued);
    font-size: 12px;
    line-height: 1.55;
}
.footer-note {
    margin-top: 14px;
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .arch-flow { grid-template-columns: 1fr; }
    .arch-arrow { transform: rotate(90deg); text-align: center; }
}
@media (max-width: 560px) {
    .hero { padding: 22px 18px; border-radius: 18px; }
    .kpi-grid { grid-template-columns: 1fr 1fr; }
    .kpi-value { font-size: 23px; }
}
"""


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _kpi_html(stats: dict[str, Any]) -> str:
    cards = [
        ("Total tickets", stats["total_tickets"]),
        ("Open", stats["open"]),
        ("Escalated", stats["escalated"]),
        ("Resolved", stats["resolved"]),
        ("Agents", stats["agents"]),
    ]
    body = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{escape(str(label))}</div>'
        f'<div class="kpi-value">{escape(str(value))}</div></div>'
        for label, value in cards
    )
    return f'<div class="kpi-grid">{body}</div>'


def _answer_html(result: dict[str, Any]) -> str:
    planner = result.get("meta", {}).get("planner", {})
    answer = escape(str(result.get("answer", "No answer returned.")))
    if planner.get("used_llm"):
        provider = escape(str(planner.get("provider", "LLM"))).title()
        model = escape(str(planner.get("model", "")))
        status = f'<span class="status-ok">LLM planner active</span> &nbsp; {provider} · {model}'
    else:
        reason = escape(str(planner.get("fallback_reason") or "LLM unavailable"))
        status = (
            '<span class="status-warn">Fallback parser used</span> &nbsp; '
            f'{reason[:140]}'
        )
    return (
        '<div class="answer-card">'
        '<div class="answer-label">Result</div>'
        f'<div class="answer-text">{answer}</div>'
        f'<div class="meta-line">{status}</div>'
        '</div>'
    )


def _anomaly_html(result: dict[str, Any]) -> str:
    by_type = result.get("by_type", {})
    urgent = int(by_type.get("urgent_unresolved", 0))
    outliers = int(by_type.get("resolution_time_outlier", 0))
    upper = result.get("iqr_upper_bound_hours")
    upper_display = "n/a" if upper is None else f"{float(upper):.2f}h"
    return (
        '<div class="anomaly-summary">'
        f'<strong>{int(result.get("total_anomalies", 0))}</strong> anomalies detected'
        '<div class="meta-line">'
        f'Urgent unresolved: <b>{urgent}</b> &nbsp; · &nbsp; '
        f'Resolution-time outliers: <b>{outliers}</b> &nbsp; · &nbsp; '
        f'IQR upper bound: <b>{upper_display}</b>'
        '</div></div>'
    )


def build_ui(service: SupportTicketService) -> gr.Blocks:
    stats = service.stats()
    provider = service.planner.provider.title()
    provider_note = "configured" if (
        service.planner.provider == "groq" and bool(config.GROQ_API_KEY)
    ) else "selected"

    status_df = pd.DataFrame(
        [{"status": name, "tickets": count} for name, count in service.df["status"].value_counts().items()]
    )
    priority_df = pd.DataFrame(
        [{"priority": name, "tickets": count} for name, count in service.df["priority"].value_counts().items()]
    )
    category_df = pd.DataFrame(
        [{"category": name, "tickets": count} for name, count in service.df["category"].value_counts().items()]
    )
    preview = service.df.sort_values("created_at", ascending=False).head(12).copy()

    def ask(question: str):
        try:
            result = service.ask(question)
            return (
                _answer_html(result),
                _rows_to_frame(result["rows"]),
                result["plan"],
                result["meta"],
            )
        except Exception as exc:
            message = escape(str(exc))
            error = (
                '<div class="answer-card"><div class="answer-label">Request could not be completed</div>'
                f'<div class="answer-text" style="font-size:16px">{message}</div></div>'
            )
            return error, pd.DataFrame(), {}, {}

    def detect(scope: str, threshold: float):
        scope_map = {"All data": "all", "This week": "this_week", "This month": "this_month"}
        try:
            result = service.anomalies(
                scope=scope_map.get(scope, "all"),
                unresolved_hours=float(threshold or 24),
            )
            return _anomaly_html(result), _rows_to_frame(result["anomalies"])
        except Exception as exc:
            return (
                f'<div class="anomaly-summary">Could not run anomaly detection: {escape(str(exc))}</div>',
                pd.DataFrame(),
            )

    with gr.Blocks(title="SupportScope — Ticket Intelligence") as demo:
        gr.HTML(
            '<div class="hero">'
            '<div class="eyebrow">DOTMappers AI Engineer Assessment</div>'
            '<h1>SupportScope</h1>'
            '<p>A lightweight support-ticket intelligence workspace. Ask questions in plain English, '
            'inspect the underlying result rows, and review explainable operational anomalies.</p>'
            '<div class="hero-chips">'
            '<span class="hero-chip">FastAPI</span>'
            '<span class="hero-chip">Gradio</span>'
            '<span class="hero-chip">Pandas</span>'
            '<span class="hero-chip">Structured LLM planning</span>'
            '</div></div>'
        )
        gr.HTML(_kpi_html(stats))

        with gr.Tabs():
            with gr.Tab("Ask the data"):
                gr.HTML(
                    '<div class="section-intro"><h3>Natural-language analysis</h3>'
                    '<p>The LLM translates a question into a restricted query plan. '
                    'Pandas performs the actual filtering and calculation.</p></div>'
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        question = gr.Textbox(
                            label="Question",
                            lines=4,
                            placeholder="Example: Which agent resolved the most tickets this month?",
                        )
                        ask_button = gr.Button("Run analysis", variant="primary", size="lg")
                        gr.Examples(
                            examples=SAMPLE_QUESTIONS,
                            inputs=question,
                            label="Try an assessment-style question",
                        )
                    with gr.Column(scale=6):
                        answer = gr.HTML(
                            '<div class="answer-card"><div class="answer-label">Result</div>'
                            '<div class="answer-text" style="font-size:16px;font-weight:600">'
                            'Enter a question to analyze the ticket dataset.</div>'
                            f'<div class="meta-line">Planner: {escape(provider)} ({provider_note})</div></div>'
                        )

                rows = gr.Dataframe(
                    label="Supporting rows",
                    interactive=False,
                    wrap=True,
                    max_height=430,
                )
                with gr.Accordion("How this query was interpreted", open=False):
                    plan = gr.JSON(label="Validated query plan")
                    meta = gr.JSON(label="Execution metadata")

                ask_button.click(ask, inputs=question, outputs=[answer, rows, plan, meta])
                question.submit(ask, inputs=question, outputs=[answer, rows, plan, meta])

            with gr.Tab("Anomaly monitor"):
                gr.HTML(
                    '<div class="section-intro"><h3>Explainable anomaly detection</h3>'
                    '<p>Combines an IQR rule for unusually long resolution times with a business rule '
                    'for High/Critical unresolved tickets.</p></div>'
                )
                with gr.Row():
                    scope = gr.Dropdown(
                        choices=["All data", "This week", "This month"],
                        value="All data",
                        label="Time window",
                    )
                    threshold = gr.Slider(
                        minimum=1,
                        maximum=168,
                        step=1,
                        value=24,
                        label="Urgent unresolved threshold (hours)",
                    )
                anomaly_button = gr.Button("Scan for anomalies", variant="primary")
                anomaly_summary = gr.HTML(
                    '<div class="anomaly-summary">Choose a time window and run the scan.</div>'
                )
                anomaly_rows = gr.Dataframe(
                    label="Flagged tickets",
                    interactive=False,
                    wrap=True,
                    max_height=500,
                )
                anomaly_button.click(
                    detect,
                    inputs=[scope, threshold],
                    outputs=[anomaly_summary, anomaly_rows],
                )

            with gr.Tab("Dataset overview"):
                gr.HTML(
                    '<div class="section-intro"><h3>Dataset profile</h3>'
                    f'<p>500 support tickets · reference timestamp: '
                    f'<b>{escape(str(stats["data_reference_time"]))}</b>. '
                    'Relative phrases such as “this week” use this dataset timestamp for reproducibility.</p></div>'
                )
                with gr.Row():
                    gr.BarPlot(
                        value=status_df,
                        x="status",
                        y="tickets",
                        title="Tickets by status",
                        x_title="Status",
                        y_title="Tickets",
                        height=300,
                    )
                    gr.BarPlot(
                        value=priority_df,
                        x="priority",
                        y="tickets",
                        title="Tickets by priority",
                        x_title="Priority",
                        y_title="Tickets",
                        height=300,
                    )
                gr.BarPlot(
                    value=category_df,
                    x="category",
                    y="tickets",
                    title="Tickets by category",
                    x_title="Category",
                    y_title="Tickets",
                    height=300,
                )
                gr.Dataframe(
                    value=preview,
                    label="Latest 12 tickets in the supplied dataset",
                    interactive=False,
                    wrap=True,
                    max_height=430,
                )

            with gr.Tab("System design"):
                gr.HTML(
                    '<div class="section-intro"><h3>Execution path</h3>'
                    '<p>The model never performs the numeric calculation itself. It only produces a '
                    'validated plan that the query engine is allowed to execute.</p></div>'
                    '<div class="arch-flow">'
                    '<div class="arch-node">User question</div><div class="arch-arrow">→</div>'
                    '<div class="arch-node">LLM planner</div><div class="arch-arrow">→</div>'
                    '<div class="arch-node">Pydantic validation</div><div class="arch-arrow">→</div>'
                    '<div class="arch-node">Pandas engine</div>'
                    '</div>'
                    '<div class="small-note">Anomaly requests use the same service layer but route to the '
                    'anomaly detector, which combines statistical and business rules. REST endpoints are '
                    'available at <code>/health</code>, <code>/stats</code>, <code>/query</code>, and '
                    '<code>/anomalies</code>. Swagger documentation is available at <code>/docs</code>.</div>'
                )

        gr.HTML(
            '<div class="footer-note">SupportScope · built for the DOTMappers AI Engineer assessment · '
            'UI and API run from the same Python application.</div>'
        )

    return demo
