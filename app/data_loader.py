from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
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

STRING_COLUMNS = ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]
NUMERIC_COLUMNS = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]


@dataclass(frozen=True)
class DataSnapshot:
    dataframe: pd.DataFrame
    reference_time: pd.Timestamp


def load_ticket_data(path: str | Path) -> DataSnapshot:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - pandas gives many parser error types
        raise ValueError(f"Could not read CSV dataset: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].copy()

    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if df["created_at"].isna().any():
        bad_rows = df.index[df["created_at"].isna()].tolist()[:5]
        raise ValueError(f"Invalid created_at values found at rows: {bad_rows}")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["ticket_id"].isna().any() or (df["ticket_id"] == "").any():
        raise ValueError("ticket_id cannot be blank")
    if df["ticket_id"].duplicated().any():
        duplicates = df.loc[df["ticket_id"].duplicated(), "ticket_id"].tolist()[:5]
        raise ValueError(f"Duplicate ticket_id values found: {duplicates}")

    if df.empty:
        raise ValueError("Dataset contains no rows")

    reference_time = df["created_at"].max()
    return DataSnapshot(dataframe=df, reference_time=reference_time)
