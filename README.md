# SupportScope — AI Support Ticket Intelligence

**Candidate:** Piyush Badjate  
**Assessment:** DOTMappers IT Pvt. Ltd. — AI Engineer / AI Intern

SupportScope is a small Python application that lets a user ask plain-English questions about the supplied `support_tickets.csv` file and inspect operational anomalies. The project exposes the same functionality through a REST API and a Gradio UI.

The central design choice is simple: **the LLM interprets the question, but Python calculates the answer**. The model produces a restricted JSON query plan, Pydantic validates it, and Pandas executes it against the CSV. This keeps numerical results reproducible and avoids asking a language model to calculate directly over the raw dataset.

## Features

- Loads and validates the provided 500-row support-ticket CSV.
- Natural-language questions for counts, lists, averages, grouped agent results, date scopes, and SLA-style requests.
- LLM query planning through Groq or a local Ollama model.
- Validated structured query plans before any dataframe operation is executed.
- Statistical anomaly detection for unusually long resolution times using the IQR rule.
- Business-rule detection for unresolved High/Critical tickets older than a configurable threshold.
- REST API with health, stats, query, and anomaly endpoints.
- Interactive Gradio dashboard mounted inside the same FastAPI application.
- Deterministic fallback parser for a small set of common questions when the configured LLM is temporarily unavailable.
- Automated tests for the main data and query paths.

## Architecture

```text
User / Gradio UI
       |
       v
    FastAPI
       |
       v
SupportTicketService
   |           |
   |           +--------------------+
   v                                v
LLM Planner                   Anomaly Detector
(Groq/Ollama)                (IQR + business rule)
   |
   v
Validated QueryPlan
   |
   v
Pandas Query Engine
   |
   v
support_tickets.csv
```

### Why I used this approach

Sending the whole CSV to an LLM for every question would be slower, harder to validate, and more likely to produce inconsistent calculations. In this project the LLM is responsible only for language understanding. It maps a question to a limited query schema, while Pandas handles the actual filtering, aggregation, sorting, and counting.

The query plan is intentionally restrictive. The model can only choose known operations, columns, filters, date scopes, and aggregations. Unsupported questions return a controlled message instead of inventing an answer.

## Project structure

```text
Piyush_Badjate_DOTMappers_AI_Engineer_Assessment/
├── app/
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── config.py
│   ├── data_loader.py
│   ├── llm_service.py
│   ├── main.py
│   ├── query_engine.py
│   ├── query_schema.py
│   ├── services.py
│   └── ui.py
├── data/
│   └── support_tickets.csv
├── tests/
│   └── test_core.py
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
├── WALKTHROUGH.md
└── README.md
```

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the environment file

Windows:

```powershell
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 4. Configure an LLM

#### Groq free tier

Add your key to `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

No API key is committed to this repository.

#### OR Ollama locally

Example:

```bash
ollama pull qwen2.5:3b
```

Then use:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

## Vercel Deployment

1. Deploy this repository to Vercel. Vercel automatically detects the root `app.py` entry point (`from app.main import app`).
2. Configure Environment Variables in the Vercel Project Dashboard:
   - `LLM_PROVIDER`: `groq`
   - `GROQ_API_KEY`: `<your-groq-api-key>`
   - `GROQ_MODEL`: `openai/gpt-oss-20b`
3. Key routes to verify after deployment:
   - `/health` — System status and ticket count (works even if API key is unconfigured)
   - `/stats` — Dataset summary
   - `/docs` — Swagger API documentation
   - `/ui/` — Interactive Gradio UI dashboard

## Run


The API and UI start together with one command:

```bash
python run.py
```

Open:

- UI: `http://127.0.0.1:8000/ui/`
- API docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

## UI

The dashboard has four sections:

1. **Ask the data** — natural-language questions, result rows, and optional query-plan details.
2. **Anomaly monitor** — selectable date scope and unresolved-ticket threshold.
3. **Dataset overview** — status, priority, category distributions, and a recent-row preview.
4. **System design** — short explanation of the execution path.

The query-plan and metadata panels are intentionally collapsible. They are useful during the technical walkthrough without cluttering the normal user experience.

## REST API

### `GET /health`

Returns application status, loaded row count, reference timestamp, and configured LLM information.

### `GET /stats`

Returns high-level dataset counts used by the UI.

### `POST /query`

Example request:

```json
{
  "question": "How many tickets are currently open?"
}
```

The response contains:

- natural-language answer,
- validated query plan,
- supporting result rows,
- execution metadata,
- whether the LLM or resilience fallback planned the query.

### `GET /anomalies`

Examples:

```text
GET /anomalies
GET /anomalies?scope=this_week
GET /anomalies?scope=this_month&unresolved_hours=24
```

## Relative dates

The supplied dataset is historical. Its latest timestamp is:

```text
2024-03-30 18:06
```

For reproducible results, phrases such as **"this week"** and **"this month"** are interpreted relative to that latest dataset timestamp, not the computer's current date. Otherwise running the assessment in 2026 would make the historical snapshot misleading.

## Missing values

Unresolved tickets can have null `resolution_time_hrs` and `customer_rating` values.

The loader keeps those null values as null:

- unresolved resolution time is not replaced with `0`,
- missing customer ratings are excluded naturally from mean calculations,
- age for unresolved tickets is calculated from `created_at` to the dataset reference time.

## SLA query behavior

For a question such as:

```text
Show me all Critical tickets not resolved within 12 hours.
```

A ticket is treated as missing the 12-hour resolution target when either:

- it was resolved after more than 12 hours, or
- it is still unresolved and was already older than 12 hours at the dataset reference timestamp.

This captures both late resolutions and still-open breaches.

## Anomaly detection

Two methods are used.

### Resolution-time outliers

For resolved tickets in the selected time scope:

```text
IQR = Q3 - Q1
Upper bound = Q3 + 1.5 × IQR
```

Tickets above the upper bound are flagged as `resolution_time_outlier`.

### Urgent unresolved tickets

A ticket is flagged as `urgent_unresolved` when:

- priority is `High` or `Critical`,
- status is `Open` or `Escalated`,
- unresolved age is greater than the configured threshold (24 hours by default).

Every flagged ticket includes a readable reason.

## Verified assessment examples

The following results were checked against the supplied CSV and the query engine:

| Question | Result |
|---|---:|
| How many tickets are currently open? | **111** |
| Which agent resolved the most tickets this month? | **AGT-01 — 16** |
| Show me all Critical tickets not resolved within 12 hours. | **34 tickets** |
| What is the average customer rating for Technical category tickets? | **3.74** |
| Are there any anomalies in resolution times this week? | **8 anomalies** |

Additional check:

- Agent with lowest average customer rating: **AGT-08 — 3.48**
- Default 24-hour anomaly scan across all data: **101 records**
- Full-data IQR upper bound: **48.15 hours**

## Tests

Run:

```bash
python -m pytest -q
```

Current test coverage checks:

- dataset loading and unique ticket IDs,
- open-ticket count,
- average Technical rating,
- relative month semantics,
- resolution SLA handling,
- grouped agent calculation,
- anomaly records and reasons,
- all five sample question patterns,
- unsupported/out-of-domain requests.

## Error handling

The application handles common failure cases explicitly:

- missing or malformed CSV,
- invalid date values,
- duplicate ticket IDs,
- unknown query columns or operators,
- invalid model JSON,
- unavailable Groq/Ollama service,
- questions outside the supplied dataset.

The fallback parser is there for resilience, not as a replacement for the assessment's LLM requirement. During the final demo, Groq or Ollama should be configured so query metadata shows `used_llm: true`.

## Known limitations

- The natural-language planner is designed around the supplied dataset schema rather than arbitrary CSV files.
- The fallback parser intentionally supports only common query patterns.
- IQR is a simple statistical rule; a production system could use historical baselines per category/priority or a trained anomaly model.
- There is no authentication because this is a local assessment prototype.
- The application reads the CSV at startup. A production system would normally use a database or warehouse.

## If I were extending this further

My next changes would be a database-backed data layer, role-based API access, query/audit history, cached analytics, more flexible time-range questions, and anomaly thresholds learned separately by ticket category and priority.
