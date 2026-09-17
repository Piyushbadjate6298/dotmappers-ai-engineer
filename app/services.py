from __future__ import annotations

from typing import Any

from . import config
from .anomaly_detector import AnomalyDetector
from .data_loader import load_ticket_data
from .llm_service import LLMPlanner
from .query_engine import QueryEngine
from .query_schema import DateScope


class SupportTicketService:
    def __init__(self) -> None:
        snapshot = load_ticket_data(config.DATA_PATH)
        self.df = snapshot.dataframe
        self.reference_time = snapshot.reference_time
        self.engine = QueryEngine(self.df, self.reference_time)
        self.anomaly_detector = AnomalyDetector(self.df, self.reference_time)
        self.planner = LLMPlanner()

    def ask(self, question: str) -> dict[str, Any]:
        plan, planner_meta = self.planner.plan(question)

        if plan.operation == "unsupported":
            return {
                "question": question,
                "answer": plan.reason or "This question cannot be answered from the support-ticket dataset.",
                "plan": plan.model_dump(),
                "rows": [],
                "meta": {
                    "matched_rows": 0,
                    "data_reference_time": self.reference_time.isoformat(),
                    "planner": planner_meta.as_dict(),
                },
            }

        if plan.operation == "anomalies":
            anomaly_data = self.anomaly_detector.detect(scope=plan.date_scope)
            count = anomaly_data["total_anomalies"]
            answer = f"Found {count} anomal{'y' if count == 1 else 'ies'} for the selected period."
            rows = anomaly_data["anomalies"][: plan.limit]
            result_meta = {
                "matched_rows": count,
                "data_reference_time": self.reference_time.isoformat(),
                "anomaly_summary": {
                    "by_type": anomaly_data["by_type"],
                    "iqr_upper_bound_hours": anomaly_data["iqr_upper_bound_hours"],
                    "unresolved_urgent_threshold_hours": anomaly_data["unresolved_urgent_threshold_hours"],
                },
            }
        else:
            result = self.engine.execute(plan)
            answer, rows, result_meta = result.answer, result.rows, result.meta

        return {
            "question": question,
            "answer": answer,
            "plan": plan.model_dump(),
            "rows": rows,
            "meta": {
                **result_meta,
                "planner": planner_meta.as_dict(),
            },
        }

    def stats(self) -> dict[str, Any]:
        status_counts = self.df["status"].value_counts().to_dict()
        return {
            "total_tickets": int(len(self.df)),
            "open": int(status_counts.get("Open", 0)),
            "resolved": int(status_counts.get("Resolved", 0)),
            "escalated": int(status_counts.get("Escalated", 0)),
            "agents": int(self.df["agent_id"].nunique()),
            "categories": {k: int(v) for k, v in self.df["category"].value_counts().to_dict().items()},
            "priorities": {k: int(v) for k, v in self.df["priority"].value_counts().to_dict().items()},
            "data_reference_time": self.reference_time.isoformat(),
        }

    def anomalies(self, scope: str = "all", unresolved_hours: float = 24.0) -> dict[str, Any]:
        return self.anomaly_detector.detect(DateScope(kind=scope), unresolved_hours=unresolved_hours)
