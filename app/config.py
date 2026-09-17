from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "support_tickets.csv")))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
ALLOW_HEURISTIC_FALLBACK = os.getenv("ALLOW_HEURISTIC_FALLBACK", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
