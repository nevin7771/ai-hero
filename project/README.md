## AIHero Project (Day 1-7)

Production-oriented version of the course work:
- retrieval pipeline
- agent with tool calling
- evaluation
- Streamlit deployment

## Dataset

### Sources
- `langchain-ai/langchain`
- `microsoft/semantic-kernel`
- `openai/openai-cookbook`

### Ingestion Rules
- Download repository zipballs via GitHub API
- Parse `.md` and `.mdx` files
- Normalize to records:
  - `id`
  - `repo`
  - `filename`
  - `text`

### Chunking
- Primary strategy: markdown section chunking by headings (`#`, `##`, etc.)
- Output schema per chunk:
  - `chunk_id`
  - `repo`
  - `filename`
  - `text`

## Retrieval + Agent

- Retrieval modes:
  - `text_search`
  - `vector_search`
  - `hybrid_search` (RRF fusion)
- Agent flow:
  - OpenAI function calling (`search_docs` tool)
  - PydanticAI tool-based agent

## Run Locally

```bash
cd project
uv sync
uv run streamlit run streamlit_app.py
```

## Streamlit Cloud Deploy (Day 6)

- Main file: `project/streamlit_app.py`
- Requirements: `project/requirements.txt`
- Add secret in app settings:
  - `OPENAI_API_KEY`

## Evaluation (Day 5/7)

Notebook: `project/project.ipynb`

- Retrieval metrics:
  - Hit@5
  - MRR
- Answer quality proxy:
  - token overlap F1 vs retrieved context
- Benchmark export artifacts:
  - `day7_benchmark_*.csv`
  - `day7_summary_*.json`
  - `day7_final_share_*.json`

See `project/eval_report.md` for latest numbers.
