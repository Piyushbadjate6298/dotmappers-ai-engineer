# Architecture Walkthrough Notes

These are the points I would be ready to explain during the post-submission discussion.

## 1. Start with the problem

The input is a historical CSV with 500 support tickets. The system needs to make that data queryable in natural language, detect anomalies, expose an API, and provide a minimal UI.

The main risk I wanted to avoid was letting an LLM directly calculate answers from raw CSV text. That is difficult to validate and can produce inconsistent numeric results.

## 2. Main design decision

I split the job into two parts:

```text
Natural language understanding -> LLM
Data calculation              -> Python/Pandas
```

The LLM returns a JSON query plan. Pydantic validates that plan before the query engine touches the dataframe.

For example:

```text
"Which agent resolved the most tickets this month?"
```

becomes approximately:

```json
{
  "operation": "group_aggregate",
  "filters": [
    {"column": "status", "operator": "eq", "value": "Resolved"}
  ],
  "metric": "ticket_id",
  "aggregation": "count",
  "group_by": "agent_id",
  "sort": "desc",
  "limit": 1,
  "date_scope": {"kind": "this_month"}
}
```

Then Pandas performs the actual grouping/counting.

## 3. Why a restricted schema

The model is not allowed to generate Python or arbitrary dataframe expressions. It can only select from known operations, columns, operators, aggregations, and date scopes.

This gives me three useful properties:

- easier validation,
- less chance of hallucinated columns,
- a query plan I can show during debugging or the walkthrough.

## 4. Historical date handling

The CSV ends on March 30, 2024. If I interpreted "this week" using today's system clock, most queries would return nothing and all unresolved tickets would look years old.

So relative time is anchored to the maximum `created_at` timestamp in the dataset. This keeps results reproducible whenever the evaluator runs the project.

## 5. Missing values

I preserve missing `resolution_time_hrs` and `customer_rating` values.

A null resolution time means the ticket is unresolved; it is not zero. Pandas automatically excludes missing ratings from average calculations.

## 6. "Not resolved within N hours"

I treat this as an SLA breach rather than just an unresolved-status filter.

It includes:

- tickets resolved later than the target, and
- tickets still unresolved after the target time.

That is why the execution engine has a dedicated `resolution_sla_hours` field in the query plan.

## 7. Anomaly logic

I used two simple, explainable methods.

### Statistical rule

For resolved tickets:

```text
upper bound = Q3 + 1.5 * IQR
```

Resolution times above the bound are flagged.

### Business rule

High/Critical tickets that are still Open/Escalated and older than a threshold are flagged separately.

This combines a data-driven signal with an operational rule that would matter to a support team.

## 8. LLM providers

The application supports:

- Groq free tier, or
- Ollama locally.

I use Groq for the easiest demo setup, but the rest of the application does not depend on one specific provider.

There is a small deterministic fallback parser for common questions if the model endpoint is unavailable. I would be clear that this is only resilience; the intended assessment path uses the LLM.

## 9. API and UI

FastAPI exposes:

```text
GET  /health
GET  /stats
POST /query
GET  /anomalies
```

Gradio is mounted at `/ui/` in the same application. That means the evaluator starts everything with:

```bash
python run.py
```

Swagger is available at `/docs`.

## 10. Questions I expect

### Why not send the whole CSV to the LLM?

Because calculations should be deterministic, easier to validate, and cheaper. The model is better used for intent parsing.

### Why Pandas instead of SQL?

The supplied input is one 500-row CSV, so Pandas keeps the prototype small. The service/query-engine separation would make a SQL implementation straightforward later.

### Why IQR?

It is simple, transparent, does not require training data, and gives a defensible baseline for a small assessment dataset.

### How would I scale it?

I would move ticket data into a database/warehouse, translate query plans to SQL, add caching and authentication, persist audit logs, and calculate anomaly baselines per segment rather than across the full dataset.

### How do I prevent unsupported questions?

The system prompt tells the model to return `unsupported`, the query plan is schema-validated, and the engine only accepts known columns/operations. If the question asks for information that is not in the dataset, the system should say so rather than invent it.

## 11. Demo order

A simple demo flow:

1. Open the dashboard and show the 500-ticket snapshot.
2. Ask: `How many tickets are currently open?`
3. Expand the query plan briefly to show structured LLM output.
4. Ask: `Which agent resolved the most tickets this month?`
5. Open the anomaly monitor and run `This week` with 24 hours.
6. Open `/docs` and show the same capabilities as REST endpoints.
7. Mention the tests and explain the historical reference-time decision.
