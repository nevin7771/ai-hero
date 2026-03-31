# AI Hero Agent Project

End-to-end implementation from the AI Hero 7-day crash course.

## What This Project Does

This repository builds a documentation assistant over:
- `langchain-ai/langchain`
- `microsoft/semantic-kernel`
- `openai/openai-cookbook`

The agent ingests markdown docs, chunks/indexes content, retrieves context with text/vector/hybrid search, and answers with source references.

## Repository Structure

- `course/` - notebook-based implementation from course days
- `project/project.ipynb` - consolidated Day 1-7 notebook
- `project/streamlit_app.py` - deployable Streamlit app (Day 6)
- `project/requirements.txt` - Streamlit app dependencies
- `project/day7_*.csv|json` - Day 7 benchmark and share artifacts

## Quick Start

```bash
cd project
uv sync
uv run streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

## Required Secrets

Set `OPENAI_API_KEY` using one of:

1) Environment variable:
```bash
export OPENAI_API_KEY="your_key"
```

2) Local Streamlit secret file (not committed):
`project/.streamlit/secrets.toml`
```toml
OPENAI_API_KEY = "your_key"
```

## Evaluation Summary (Day 7)

- Best retrieval mode: `hybrid`
- Hit@5: `0.95`
- MRR: `0.8494`
- Answer-vs-context token F1: `0.3778`
- Benchmark questions: `5`

See `project/eval_report.md` for details.
