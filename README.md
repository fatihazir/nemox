# Maritime matching experiment

A small experiment: can an LLM read a freight broker's email while plain code makes the decisions and a person confirms the important steps? The routine runs automatically; exceptions go to a person, each with a reason.

It is built on assumptions about how such a system could work, with synthetic data and no access to real systems.

## How it works

- The LLM extracts the email into a strict Pydantic `Cargo` record and, at the end, drafts the customer email from a JSON of verified facts. Every number in the draft is checked against those facts.
- Plain Python matches ships (hard filters plus a score) and a rule decides: `auto_propose`, `ask_sender` or `human_review`.
- Mode A is a fixed workflow. Mode B lets the model call tools; the one risky tool is gated by the same rule.

## Files

| File | Content |
|---|---|
| `maritime_matching_demo.ipynb` | Offline version with a canned `FakeLLM` |
| `llm_demo.ipynb` | Real models: structured output, Mode A and B, golden-set evaluation, provider comparison, fallback, checkpoint and resume |
| `llm_demo_langgraph.ipynb` | The same flow as a LangGraph graph with a pause before the write step |
| `pydantic_test.ipynb` | Early sketch of schema validation and a tool definition |
| `golden_set.json` | 12 labelled test emails |
| `results/` | One JSON per evaluation run |
| `state/` | Checkpoint written by the resume demo |
| `trace.jsonl`, `trace_langgraph.jsonl` | Local traces (emails and phone numbers redacted) |

## Results

12 golden emails, prompt `extract-v1`, temperature 0. Costs are estimates from an editable price table.

| Model | Fully correct | Valid on first try | Avg time per call | Est. cost |
|---|---|---|---|---|
| openai / gpt-4.1-mini | 100.0 % | 100.0 % | 1304 ms | USD 0.0044 |
| groq / openai/gpt-oss-20b | 91.7 % | 100.0 % | 396 ms | USD 0.0013 |
| google / gemini-3.5-flash-lite | 91.7 % | 100.0 % | 1605 ms | USD 0.0059 |

- 8 of 12 emails were routine, 4 went to a person (two cargoes, large cargo, low score, missing ship type).
- gpt-oss-20b missed the second cargo in the two-cargo email. The action was still right, but for the wrong reason.
- Gemini returned `weizen` instead of `wheat` for the German email.
- The email with the injected instruction led to the same decision as its clean twin, in both modes.

## Run

```bash
# API keys are read from a .env file in this or a parent folder
jupyter lab llm_demo.ipynb

# headless; the provider comparison needs the approval flag
DEMO_APPROVE_COMPARISON=1 jupyter nbconvert --to notebook --execute --inplace llm_demo.ipynb
```

Requires `openai`, `pydantic` and `python-dotenv`. Optional: `logfire`, and `langgraph` with `nbformat` for the LangGraph notebook.

Environment variables: `OPENAI_MODEL`, `GROQ_MODEL`, `GOOGLE_MODEL`, `XAI_MODEL`, `DEMO_APPROVE_COMPARISON`, `DEMO_BUDGET_USD`, `OPENAI_BIG_MODEL`.

## Limits

- Synthetic data: 6 ships, a few distances, 12 emails written for this test. The TMS is a dict.
- The golden set was also used while developing, so the scores are optimistic.
- One email is 8 percentage points, so differences between models are indicative only.
- Logfire was not run, and tool calling was only tested on OpenAI.
