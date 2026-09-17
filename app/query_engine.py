from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .query_schema import DateScope, FilterCondition, QueryPlan


ALLOWED_COLUMNS = {
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
}
NUMERIC_COLUMNS = {"response_time_hrs", "resolution_time_hrs", "customer_rating"}
DEFAULT_LIST_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


@dataclass
class QueryResult:
    answer: str
    rows: list[dict[str, Any]]
    meta: dict[str, Any]


class QueryEngine:
    def __init__(self, dataframe: pd.DataFrame, reference_time: pd.Timestamp):
        self.df = dataframe.copy()
        self.reference_time = pd.Timestamp(reference_time)

    def _validate_column(self, column: str) -> None:
        if column not in ALLOWED_COLUMNS:
            raise ValueError(f"Unsupported column: {column}")

    @staticmethod
    def _string_series(series: pd.Series) -> pd.Series:
        return series.astype("string").str.casefold()

    def _apply_filter(self, df: pd.DataFrame, condition: FilterCondition) -> pd.DataFrame:
        self._validate_column(condition.column)
        series = df[condition.column]
        op = condition.operator
        value = condition.value

        if op == "is_null":
            return df[series.isna()]
        if op == "not_null":
            return df[series.notna()]

        if condition.column in NUMERIC_COLUMNS:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Filter for {condition.column} requires a numeric value") from exc
            if op == "eq":
                mask = series == numeric_value
            elif op == "ne":
                mask = series != numeric_value
            elif op == "gt":
                mask = series > numeric_value
            elif op == "gte":
                mask = series >= numeric_value
            elif op == "lt":
                mask = series < numeric_value
            elif op == "lte":
                mask = series <= numeric_value
            elif op in {"in", "not_in"}:
                values = value if isinstance(value, list) else [value]
                parsed = [float(item) for item in values]
                mask = series.isin(parsed)
                if op == "not_in":
                    mask = ~mask
            else:
                raise ValueError(f"Operator '{op}' is not supported for numeric column {condition.column}")
            return df[mask.fillna(False)]

        if condition.column == "created_at":
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                raise ValueError("Date filter value could not be parsed")
            if op == "eq":
                mask = series == parsed
            elif op == "ne":
                mask = series != parsed
            elif op == "gt":
                mask = series > parsed
            elif op == "gte":
                mask = series >= parsed
            elif op == "lt":
                mask = series < parsed
            elif op == "lte":
                mask = series <= parsed
            else:
                raise ValueError(f"Operator '{op}' is not supported for created_at")
            return df[mask.fillna(False)]

        folded = self._string_series(series)
        if op in {"in", "not_in"}:
            values = value if isinstance(value, list) else [value]
            wanted = [str(item).casefold() for item in values]
            mask = folded.isin(wanted)
            if op == "not_in":
                mask = ~mask
        elif op == "contains":
            mask = folded.str.contains(str(value).casefold(), na=False, regex=False)
        elif op == "eq":
            mask = folded == str(value).casefold()
        elif op == "ne":
            mask = folded != str(value).casefold()
        else:
            raise ValueError(f"Operator '{op}' is not supported for text column {condition.column}")
        return df[mask.fillna(False)]

    def apply_date_scope(self, df: pd.DataFrame, scope: DateScope) -> pd.DataFrame:
        kind = scope.kind
        if kind == "all":
            return df

        ref = self.reference_time
        if kind == "this_week":
            start = ref.normalize() - pd.Timedelta(days=ref.weekday())
            return df[(df["created_at"] >= start) & (df["created_at"] <= ref)]

        if kind == "this_month":
            start = ref.normalize().replace(day=1)
            return df[(df["created_at"] >= start) & (df["created_at"] <= ref)]

        if kind == "last_n_days":
            if not scope.days:
                raise ValueError("date_scope.days is required for last_n_days")
            start = ref - pd.Timedelta(days=scope.days)
            return df[(df["created_at"] >= start) & (df["created_at"] <= ref)]

        if kind == "between":
            if not scope.start or not scope.end:
                raise ValueError("date_scope.start and date_scope.end are required for between")
            start = pd.to_datetime(scope.start, errors="coerce")
            end = pd.to_datetime(scope.end, errors="coerce")
            if pd.isna(start) or pd.isna(end):
                raise ValueError("Could not parse between date range")
            # If the end is a date-only string, include that full date.
            if len(scope.end.strip()) <= 10:
                end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            return df[(df["created_at"] >= start) & (df["created_at"] <= end)]

        raise ValueError(f"Unsupported date scope: {kind}")

    def _apply_resolution_sla(self, df: pd.DataFrame, hours: float | None) -> pd.DataFrame:
        if hours is None:
            return df
        age_hours = (self.reference_time - df["created_at"]).dt.total_seconds() / 3600
        breached = (
            (df["resolution_time_hrs"].notna() & (df["resolution_time_hrs"] > hours))
            | (df["resolution_time_hrs"].isna() & (age_hours > hours))
        )
        return df[breached.fillna(False)]

    def filter_dataframe(self, plan: QueryPlan) -> pd.DataFrame:
        filtered = self.apply_date_scope(self.df, plan.date_scope)
        for condition in plan.filters:
            filtered = self._apply_filter(filtered, condition)
        filtered = self._apply_resolution_sla(filtered, plan.resolution_sla_hours)
        return filtered

    @staticmethod
    def _json_ready(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record in records:
            clean: dict[str, Any] = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean[key] = None
                elif isinstance(value, pd.Timestamp):
                    clean[key] = value.isoformat()
                elif hasattr(value, "item"):
                    clean[key] = value.item()
                else:
                    clean[key] = value
            result.append(clean)
        return result

    def execute(self, plan: QueryPlan) -> QueryResult:
        if plan.operation == "anomalies":
            raise ValueError("Anomaly plans must be handled by the anomaly detector")

        filtered = self.filter_dataframe(plan)
        matched = len(filtered)
        base_meta = {
            "matched_rows": matched,
            "data_reference_time": self.reference_time.isoformat(),
            "date_scope": plan.date_scope.model_dump(),
        }

        if plan.operation == "count":
            return QueryResult(
                answer=(f"There is {matched} ticket matching this request." if matched == 1 else f"There are {matched} tickets matching this request."),
                rows=[{"count": matched}],
                meta=base_meta,
            )

        if plan.operation == "list":
            listed = filtered.sort_values("created_at", ascending=False).head(plan.limit)
            rows = self._json_ready(listed[DEFAULT_LIST_COLUMNS].to_dict(orient="records"))
            suffix = "" if matched <= plan.limit else f" Showing the first {plan.limit}."
            return QueryResult(
                answer=f"Found {matched} matching ticket{'s' if matched != 1 else ''}.{suffix}",
                rows=rows,
                meta={**base_meta, "returned_rows": len(rows)},
            )

        if plan.operation == "aggregate":
            if not plan.metric or not plan.aggregation:
                raise ValueError("aggregate requires metric and aggregation")
            self._validate_column(plan.metric)
            series = filtered[plan.metric].dropna()
            if series.empty:
                value: float | int | None = None
            elif plan.aggregation == "count":
                value = int(series.count())
            elif plan.aggregation == "mean":
                value = float(series.mean())
            elif plan.aggregation == "median":
                value = float(series.median())
            elif plan.aggregation == "min":
                value = float(series.min()) if plan.metric in NUMERIC_COLUMNS else series.min()
            elif plan.aggregation == "max":
                value = float(series.max()) if plan.metric in NUMERIC_COLUMNS else series.max()
            elif plan.aggregation == "sum":
                value = float(series.sum())
            else:  # pragma: no cover - protected by pydantic
                raise ValueError(f"Unsupported aggregation: {plan.aggregation}")

            display = "No value is available" if value is None else (
                f"{value:.2f}" if isinstance(value, float) else str(value)
            )
            aggregation_label = {
                "mean": "Average",
                "median": "Median",
                "min": "Minimum",
                "max": "Maximum",
                "sum": "Total",
                "count": "Count",
            }.get(plan.aggregation, plan.aggregation.title())
            metric_label = plan.metric.replace("_", " ")
            return QueryResult(
                answer=f"{aggregation_label} {metric_label}: {display}.",
                rows=[{"metric": plan.metric, "aggregation": plan.aggregation, "value": value}],
                meta=base_meta,
            )

        if plan.operation == "group_aggregate":
            if not plan.group_by or not plan.aggregation:
                raise ValueError("group_aggregate requires group_by and aggregation")
            self._validate_column(plan.group_by)
            metric = plan.metric or "ticket_id"
            self._validate_column(metric)

            grouped = filtered.groupby(plan.group_by, dropna=False)
            if plan.aggregation == "count":
                values = grouped.size().rename("value")
            else:
                if metric not in NUMERIC_COLUMNS:
                    raise ValueError(f"Aggregation '{plan.aggregation}' requires a numeric metric")
                metric_group = grouped[metric]
                if plan.aggregation == "mean":
                    values = metric_group.mean().rename("value")
                elif plan.aggregation == "median":
                    values = metric_group.median().rename("value")
                elif plan.aggregation == "min":
                    values = metric_group.min().rename("value")
                elif plan.aggregation == "max":
                    values = metric_group.max().rename("value")
                elif plan.aggregation == "sum":
                    values = metric_group.sum().rename("value")
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported aggregation: {plan.aggregation}")

            result_df = values.reset_index().dropna(subset=["value"])
            result_df = result_df.sort_values("value", ascending=plan.sort == "asc").head(plan.limit)
            rows = self._json_ready(result_df.to_dict(orient="records"))
            if rows:
                top = rows[0]
                value = top["value"]
                shown = f"{value:.2f}" if isinstance(value, float) else str(value)
                answer = f"{top[plan.group_by]} is the first result for this query with a value of {shown}."
            else:
                answer = "No matching grouped result is available."
            return QueryResult(
                answer=answer,
                rows=rows,
                meta={**base_meta, "groups_returned": len(rows)},
            )

        raise ValueError(f"Unsupported operation: {plan.operation}")
