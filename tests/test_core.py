from pathlib import Path

import pytest

from app.anomaly_detector import AnomalyDetector
from app.data_loader import load_ticket_data
from app.llm_service import LLMPlanner
from app.query_engine import QueryEngine
from app.query_schema import DateScope, FilterCondition, QueryPlan

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = load_ticket_data(ROOT / "data" / "support_tickets.csv")
ENGINE = QueryEngine(SNAPSHOT.dataframe, SNAPSHOT.reference_time)


def test_dataset_loaded():
    assert len(SNAPSHOT.dataframe) == 500
    assert SNAPSHOT.dataframe["ticket_id"].is_unique


def test_open_ticket_count():
    plan = QueryPlan(
        operation="count",
        filters=[FilterCondition(column="status", operator="eq", value="Open")],
    )
    result = ENGINE.execute(plan)
    assert result.rows[0]["count"] == 111


def test_technical_average_rating():
    plan = QueryPlan(
        operation="aggregate",
        filters=[FilterCondition(column="category", operator="eq", value="Technical")],
        metric="customer_rating",
        aggregation="mean",
    )
    result = ENGINE.execute(plan)
    expected = SNAPSHOT.dataframe.loc[
        SNAPSHOT.dataframe["category"] == "Technical", "customer_rating"
    ].mean()
    assert result.rows[0]["value"] == pytest.approx(expected)


def test_this_month_uses_dataset_reference_time():
    plan = QueryPlan(operation="count", date_scope=DateScope(kind="this_month"))
    result = ENGINE.execute(plan)
    expected = SNAPSHOT.dataframe[
        SNAPSHOT.dataframe["created_at"].dt.month == SNAPSHOT.reference_time.month
    ]
    assert result.rows[0]["count"] == len(expected)


def test_resolution_sla_includes_late_resolved_and_old_unresolved():
    plan = QueryPlan(
        operation="list",
        filters=[FilterCondition(column="priority", operator="eq", value="Critical")],
        resolution_sla_hours=12,
        limit=100,
    )
    filtered = ENGINE.filter_dataframe(plan)
    assert not filtered.empty
    assert (filtered["priority"] == "Critical").all()
    resolved = filtered["resolution_time_hrs"].notna()
    assert (filtered.loc[resolved, "resolution_time_hrs"] > 12).all()


def test_group_agent_resolved_most_this_month():
    plan = QueryPlan(
        operation="group_aggregate",
        filters=[FilterCondition(column="status", operator="eq", value="Resolved")],
        metric="ticket_id",
        aggregation="count",
        group_by="agent_id",
        sort="desc",
        limit=1,
        date_scope=DateScope(kind="this_month"),
    )
    result = ENGINE.execute(plan)
    assert len(result.rows) == 1
    assert result.rows[0]["value"] > 0


def test_anomaly_detector_returns_explainable_records():
    detector = AnomalyDetector(SNAPSHOT.dataframe, SNAPSHOT.reference_time)
    result = detector.detect(DateScope(kind="all"))
    assert "total_anomalies" in result
    assert result["total_anomalies"] == len(result["anomalies"])
    if result["anomalies"]:
        first = result["anomalies"][0]
        assert first["reason"]
        assert first["anomaly_type"] in {"resolution_time_outlier", "urgent_unresolved"}


def test_heuristic_planner_handles_sample_questions(monkeypatch):
    planner = LLMPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda question: (_ for _ in ()).throw(RuntimeError("offline")))

    plan, meta = planner.plan("How many tickets are currently open?")
    assert plan.operation == "count"
    assert any(f.column == "status" and f.value == "Open" for f in plan.filters)
    assert meta.used_llm is False

    plan, _ = planner.plan("Which agent resolved the most tickets this month?")
    assert plan.operation == "group_aggregate"
    assert plan.group_by == "agent_id"
    assert plan.date_scope.kind == "this_month"

    plan, _ = planner.plan("Show me all Critical tickets not resolved within 12 hours.")
    assert plan.operation == "list"
    assert plan.resolution_sla_hours == 12
    assert plan.limit == 100

    plan, _ = planner.plan("What is the average customer rating for Technical category tickets?")
    assert plan.operation == "aggregate"
    assert plan.metric == "customer_rating"
    assert plan.aggregation == "mean"

    plan, _ = planner.plan("Are there any anomalies in resolution times this week?")
    assert plan.operation == "anomalies"
    assert plan.date_scope.kind == "this_week"


def test_out_of_domain_question_is_rejected_by_fallback(monkeypatch):
    planner = LLMPlanner()
    monkeypatch.setattr(planner, "_call_llm", lambda question: (_ for _ in ()).throw(RuntimeError("offline")))
    plan, _ = planner.plan("What is the salary of the CEO?")
    assert plan.operation == "unsupported"
    assert plan.reason
