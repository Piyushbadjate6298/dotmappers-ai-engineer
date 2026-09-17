from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FilterCondition(BaseModel):
    column: str
    operator: Literal[
        "eq",
        "ne",
        "in",
        "not_in",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "is_null",
        "not_null",
    ] = "eq"
    value: Any | None = None


class DateScope(BaseModel):
    kind: Literal["all", "this_week", "this_month", "last_n_days", "between"] = "all"
    days: int | None = Field(default=None, ge=1, le=3650)
    start: str | None = None
    end: str | None = None


class QueryPlan(BaseModel):
    operation: Literal["count", "list", "aggregate", "group_aggregate", "anomalies", "unsupported"]
    filters: list[FilterCondition] = Field(default_factory=list)
    metric: str | None = None
    aggregation: Literal["count", "mean", "median", "min", "max", "sum"] | None = None
    group_by: str | None = None
    sort: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=25, ge=1, le=500)
    date_scope: DateScope = Field(default_factory=DateScope)
    resolution_sla_hours: float | None = Field(default=None, gt=0)
    reason: str | None = None

    @field_validator("metric", "group_by")
    @classmethod
    def normalize_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None
