# Multi-agent research assistant

A LangGraph + LangChain pipeline that takes a research topic, decomposes it
into sub-questions, researches each one in parallel, drafts a report, and
critiques/revises it before returning a final answer — with cross-session
memory and a human approval checkpoint built in.

## How it works

```
user input
   -> memory lookup        (checks past findings on this topic)
   -> supervisor            (breaks topic into 3-4 sub-questions)
   -> human approval        (pauses — you approve or edit the plan)
   -> researchers (parallel) (one per sub-question, Tavily web search)
   -> writer                (drafts the report from all sources)
   -> critic                (approves, or sends feedback)
        -> back to writer if feedback (capped at 2 revisions)
        -> finalize if approved
   -> final report           (also stores new findings for next time)
```

Two things make this more than the standard LangGraph tutorial:

- **Dynamic fan-out** — the number of parallel researchers scales with
  however many sub-questions the supervisor produces (via `Send`), not a
  hardcoded number.
- **Persistent memory across runs** — a local Chroma vector store lets
  later runs reuse earlier findings instead of re-researching from scratch.
- **Human-in-the-loop** — the graph pauses (`interrupt()`) before any paid
  search calls happen, so you can approve or edit the research plan first.

## Tech stack

- `langgraph` — orchestration (state graph, fan-out/fan-in, conditional edges)
- `langchain-anthropic` — Claude Haiku 4.5 for cheap routing, Sonnet 4.6 for
  drafting/reviewing
- `langchain-tavily` — web search tool
- `chromadb` — local vector store for cross-session memory
- `langgraph-checkpoint-postgres` — persistence layer (swappable for sqlite)
- `pydantic` — structured output between agents

## Project structure

| File | Purpose |
|---|---|
| `state.py` | Shared state schema that flows through every node |
| `memory.py` | Cross-session memory: query/store findings in Chroma |
| `tools.py` | Tavily search tool wrapper |
| `nodes.py` | The agent functions (memory lookup, supervisor, approval, researcher, writer, critic, finalize) |
| `graph.py` | Builds and compiles the graph: nodes, fan-out, revise loop, checkpointer |
| `main.py` | CLI entry point — runs the graph, handles the approval pause/resume, prints the report |
| `requirements.txt` | Dependencies |
| `.env` | Required environment variables |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your real API keys and Postgres URI
```

Requires a running Postgres database for the checkpointer — `POSTGRES_URI`
in `.env` should point to it.

## Usage

```bash
python main.py "impact of AI on the job market in 2026"
```

When the plan comes up, press Enter to approve it as-is, or type replacement
sub-questions separated by `;`.

## Notes

- Revisions are capped at 2 — the critic loop can't run forever.
- Each researcher summary is truncated to 1000 characters to keep writer
  input costs predictable.
- Memory entries are deduplicated by hashing `(topic, sub_question, summary)`,
  so re-running the same research won't create duplicate vector store entries.

## Possible extensions

- Evaluation harness (faithfulness/citation accuracy checks via LangSmith)
- LangSmith tracing for debugging individual runs
- Wrap as a FastAPI service instead of a CLI
