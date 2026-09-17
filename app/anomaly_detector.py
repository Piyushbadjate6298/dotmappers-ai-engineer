from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .query_engine import QueryEngine
from .query_schema import DateScope


class AnomalyDetector:
    """Detect statistical resolution-time outliers and urgent unresolved tickets."""

    def __init__(self, dataframe: pd.DataFrame, reference_time: pd.Timestamp):
        self.df = dataframe.copy()
        self.reference_time = pd.Timestamp(reference_time)
        self.engine = QueryEngine(self.df, self.reference_time)

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value

    def detect(self, scope: DateScope | None = None, unresolved_hours: float = 24.0) -> dict[str, Any]:
        scope = scope or DateScope()
        scoped = self.engine.apply_date_scope(self.df, scope).copy()
        anomalies: list[dict[str, Any]] = []

        # 1) Statistical outliers among resolved tickets using the IQR rule.
        resolved = scoped[
            scoped["resolution_time_hrs"].notna() & (scoped["status"].str.casefold() == "resolved")
        ].copy()
        iqr_threshold: float | None = None
        if len(resolved) >= 4:
            q1 = float(resolved["resolution_time_hrs"].quantile(0.25))
            q3 = float(resolved["resolution_time_hrs"].quantile(0.75))
            iqr = q3 - q1
            iqr_threshold = q3 + (1.5 * iqr)
            outliers = resolved[resolved["resolution_time_hrs"] > iqr_threshold]
            for _, row in outliers.iterrows():
                anomalies.append(
                    self._record(
                        row,
                        anomaly_type="resolution_time_outlier",
                        severity="medium",
                        reason=(
                            f"Resolution time {row['resolution_time_hrs']:.1f}h exceeds the "
                            f"IQR upper bound of {iqr_threshold:.1f}h for the selected period."
                        ),
                    )
                )

        # 2) Business-rule anomaly: High/Critical unresolved tickets older than N hours.
        age_hours = (self.reference_time - scoped["created_at"]).dt.total_seconds() / 3600
        urgent_unresolved_mask = (
            scoped["priority"].str.casefold().isin(["high", "critical"])
            & scoped["status"].str.casefold().isin(["open", "escalated"])
            & (age_hours > unresolved_hours)
        )
        urgent = scoped[urgent_unresolved_mask].copy()
        urgent_age = age_hours[urgent_unresolved_mask]
        for idx, row in urgent.iterrows():
            age = float(urgent_age.loc[idx])
            severity = "critical" if row["priority"].casefold() == "critical" else "high"
            anomalies.append(
                self._record(
                    row,
                    anomaly_type="urgent_unresolved",
                    severity=severity,
                    reason=(
                        f"{row['priority']} priority ticket is {row['status']} and has remained "
                        f"unresolved for {age:.1f}h (> {unresolved_hours:.1f}h)."
                    ),
                    age_hours=age,
                )
            )

        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda item: (severity_rank.get(item["severity"], 9), item["ticket_id"]))
        counts = Counter(item["anomaly_type"] for item in anomalies)
        return {
            "total_anomalies": len(anomalies),
            "by_type": dict(counts),
            "scope": scope.model_dump(),
            "reference_time": self.reference_time.isoformat(),
            "iqr_upper_bound_hours": round(iqr_threshold, 3) if iqr_threshold is not None else None,
            "unresolved_urgent_threshold_hours": unresolved_hours,
            "anomalies": anomalies,
        }

    def _record(
        self,
        row: pd.Series,
        *,
        anomaly_type: str,
        severity: str,
        reason: str,
        age_hours: float | None = None,
    ) -> dict[str, Any]:
        return {
            "ticket_id": self._clean_value(row["ticket_id"]),
            "anomaly_type": anomaly_type,
            "severity": severity,
            "reason": reason,
            "created_at": self._clean_value(row["created_at"]),
            "category": self._clean_value(row["category"]),
            "priority": self._clean_value(row["priority"]),
            "status": self._clean_value(row["status"]),
            "agent_id": self._clean_value(row["agent_id"]),
            "resolution_time_hrs": self._clean_value(row["resolution_time_hrs"]),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
        }
