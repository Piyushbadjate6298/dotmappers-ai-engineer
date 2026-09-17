from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from . import config
from .query_schema import DateScope, FilterCondition, QueryPlan


SYSTEM_PROMPT = """You are a query planner for a customer-support ticket dataset.
Convert the user's question into exactly one JSON query plan. Return JSON only, with no markdown.

Dataset columns:
- ticket_id: string
- created_at: datetime
- category: Billing | Technical | General
- priority: Low | Medium | High | Critical
- status: Open | Resolved | Escalated
- response_time_hrs: float
- resolution_time_hrs: float, null if unresolved
- agent_id: AGT-01 ... AGT-12
- customer_rating: 1..5, null if unresolved
- issue_summary: text

Allowed JSON shape:
{
  "operation": "count|list|aggregate|group_aggregate|anomalies|unsupported",
  "filters": [{"column":"...", "operator":"eq|ne|in|not_in|gt|gte|lt|lte|contains|is_null|not_null", "value": ...}],
  "metric": null | "column",
  "aggregation": null | "count|mean|median|min|max|sum",
  "group_by": null | "column",
  "sort": "asc|desc",
  "limit": 1..500,
  "date_scope": {"kind":"all|this_week|this_month|last_n_days|between", "days":null, "start":null, "end":null},
  "resolution_sla_hours": null | number,
  "reason": null | "short explanation for unsupported questions"
}

Rules:
1. 'currently open' means status == Open.
2. 'unresolved' normally means status in [Open, Escalated].
3. For 'not resolved within N hours', set resolution_sla_hours=N. Do not add an unresolved status filter; the executor includes both tickets resolved after N hours and tickets still unresolved beyond N hours.
4. 'this month' and 'this week' refer to the dataset reference time, not today's wall-clock date.
5. 'which agent resolved the most tickets this month' => group_aggregate, status Resolved, date_scope this_month, group_by agent_id, aggregation count, metric ticket_id, sort desc, limit 1.
6. Average customer rating => metric customer_rating, aggregation mean.
7. Resolution-time anomaly/outlier questions => operation anomalies and preserve any date scope.
8. Use operation=list when the user asks to show/list tickets. limit=100 if they say 'all'; otherwise 25.
9. Use only the listed columns and allowed values. Do not invent data.
10. If the question cannot be answered from this dataset, use operation=unsupported and put a short explanation in reason.
"""


@dataclass
class PlannerMetadata:
    provider: str
    model: str | None
    used_llm: bool
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "used_llm": self.used_llm,
            "fallback_reason": self.fallback_reason,
        }


class LLMPlanner:
    def __init__(self) -> None:
        self.provider = self._resolve_provider()

    @staticmethod
    def _resolve_provider() -> str:
        if config.LLM_PROVIDER in {"groq", "ollama"}:
            return config.LLM_PROVIDER
        if config.LLM_PROVIDER != "auto":
            raise ValueError("LLM_PROVIDER must be auto, groq, or ollama")
        return "groq" if config.GROQ_API_KEY else "ollama"

    @property
    def model(self) -> str:
        return config.GROQ_MODEL if self.provider == "groq" else config.OLLAMA_MODEL

    def plan(self, question: str) -> tuple[QueryPlan, PlannerMetadata]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            raw = self._call_llm(question)
            payload = self._extract_json(raw)
            plan = QueryPlan.model_validate(payload)
            return plan, PlannerMetadata(self.provider, self.model, True)
        except Exception as exc:
            if not config.ALLOW_HEURISTIC_FALLBACK:
                raise RuntimeError(f"LLM query planning failed: {exc}") from exc
            plan = self._heuristic_plan(question)
            return plan, PlannerMetadata(
                "heuristic-fallback",
                None,
                False,
                fallback_reason=str(exc)[:300],
            )

    def _call_llm(self, question: str) -> str:
        if self.provider == "groq":
            if not config.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not configured")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.GROQ_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "response_format": {"type": "json_object"},
            }
            response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
            # Some free-tier models may not accept response_format; retry once without it.
            if response.status_code == 400:
                payload.pop("response_format", None)
                response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "options": {"temperature": 0},
        }
        response = requests.post(url, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError("LLM did not return a JSON object")
            return json.loads(match.group(0))

    def _heuristic_plan(self, question: str) -> QueryPlan:
        """Resilience fallback for common analytics questions if the configured LLM is unavailable."""
        q = question.casefold()
        filters: list[FilterCondition] = []
        date_scope = DateScope()

        if "this month" in q:
            date_scope = DateScope(kind="this_month")
        elif "this week" in q:
            date_scope = DateScope(kind="this_week")
        else:
            days_match = re.search(r"(?:last|past)\s+(\d+)\s+days?", q)
            if days_match:
                date_scope = DateScope(kind="last_n_days", days=int(days_match.group(1)))

        for value in ["critical", "high", "medium", "low"]:
            if re.search(rf"\b{value}\b", q):
                filters.append(FilterCondition(column="priority", operator="eq", value=value.title()))
                break

        for value in ["technical", "billing", "general"]:
            if re.search(rf"\b{value}\b", q):
                filters.append(FilterCondition(column="category", operator="eq", value=value.title()))
                break

        agent_match = re.search(r"agt[-\s]?(\d{1,2})", q, re.IGNORECASE)
        if agent_match:
            filters.append(
                FilterCondition(column="agent_id", operator="eq", value=f"AGT-{int(agent_match.group(1)):02d}")
            )

        sla_match = re.search(r"not\s+resolved\s+within\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", q)
        resolution_sla = float(sla_match.group(1)) if sla_match else None

        if "currently open" in q or re.search(r"\bopen tickets?\b", q):
            filters.append(FilterCondition(column="status", operator="eq", value="Open"))
        elif "escalated" in q and "unresolved" not in q:
            filters.append(FilterCondition(column="status", operator="eq", value="Escalated"))
        elif "resolved" in q and "unresolved" not in q and "not resolved" not in q:
            filters.append(FilterCondition(column="status", operator="eq", value="Resolved"))
        elif ("unresolved" in q or "not resolved" in q) and resolution_sla is None:
            filters.append(FilterCondition(column="status", operator="in", value=["Open", "Escalated"]))

        if "anomal" in q or "outlier" in q:
            return QueryPlan(operation="anomalies", filters=filters, date_scope=date_scope)

        dataset_terms = [
            "ticket", "agent", "rating", "resolution", "response", "technical",
            "billing", "general", "priority", "critical", "escalated", "open",
            "resolved", "issue", "support", "category"
        ]
        if not any(term in q for term in dataset_terms):
            return QueryPlan(
                operation="unsupported",
                reason="The question asks for information that is not present in the support-ticket dataset.",
            )

        metric = None
        if "customer rating" in q or "rating" in q:
            metric = "customer_rating"
        elif "resolution time" in q:
            metric = "resolution_time_hrs"
        elif "response time" in q:
            metric = "response_time_hrs"

        aggregation = None
        if "average" in q or "mean" in q:
            aggregation = "mean"
        elif "median" in q:
            aggregation = "median"
        elif re.search(r"\b(sum|total)\b", q) and metric:
            aggregation = "sum"
        elif "maximum" in q or "highest" in q:
            aggregation = "max"
        elif "minimum" in q or "lowest" in q:
            aggregation = "min"

        if "which agent" in q or "by agent" in q or "per agent" in q:
            group_agg = aggregation or "count"
            sort = "asc" if any(term in q for term in ["lowest", "least", "minimum"]) else "desc"
            if "most tickets" in q or "most" in q and metric is None:
                group_agg = "count"
            return QueryPlan(
                operation="group_aggregate",
                filters=filters,
                metric=metric or "ticket_id",
                aggregation=group_agg,
                group_by="agent_id",
                sort=sort,
                limit=1 if any(term in q for term in ["which agent", "most", "lowest", "highest", "least"]) else 12,
                date_scope=date_scope,
                resolution_sla_hours=resolution_sla,
            )

        if aggregation and metric:
            return QueryPlan(
                operation="aggregate",
                filters=filters,
                metric=metric,
                aggregation=aggregation,
                date_scope=date_scope,
                resolution_sla_hours=resolution_sla,
            )

        if any(term in q for term in ["how many", "count", "number of"]):
            return QueryPlan(
                operation="count",
                filters=filters,
                date_scope=date_scope,
                resolution_sla_hours=resolution_sla,
            )

        limit = 100 if re.search(r"\ball\b", q) else 25
        return QueryPlan(
            operation="list",
            filters=filters,
            limit=limit,
            date_scope=date_scope,
            resolution_sla_hours=resolution_sla,
        )
