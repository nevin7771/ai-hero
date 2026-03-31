import io
import json
import os
import re
import zipfile

import numpy as np
import requests
import streamlit as st
from minsearch import Index, VectorSearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer


@st.cache_data(show_spinner=False)
def read_repo_data(repo_owner: str, repo_name: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/zipball"
    resp = requests.get(url, headers={"User-Agent": "aihero-streamlit-agent"}, timeout=60)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    rows: list[dict] = []
    for f in zf.infolist():
        if not f.filename.lower().endswith((".md", ".mdx")):
            continue
        content = zf.read(f.filename).decode("utf-8", errors="ignore")
        rows.append({"filename": f.filename, "content": content})
    zf.close()
    return rows


def normalize_docs(rows: list[dict], repo_name: str) -> list[dict]:
    docs = []
    for i, row in enumerate(rows):
        text = (row.get("content") or "").strip()
        if not text:
            continue
        docs.append(
            {
                "id": f"{repo_name}-{i}",
                "repo": repo_name,
                "filename": row.get("filename", ""),
                "text": text,
            }
        )
    return docs


def section_chunking(text: str) -> list[str]:
    heading_pattern = re.compile(r"^#{1,6}\s+.*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [text[:1200]]

    chunks = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            chunks.append(section_text)
    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        for idx, piece in enumerate(section_chunking(doc["text"])):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['id']}-section-c{idx}",
                    "repo": doc["repo"],
                    "filename": doc["filename"],
                    "text": piece,
                }
            )
    return all_chunks


@st.cache_resource(show_spinner=True)
def build_indexes():
    rows = []
    rows += normalize_docs(read_repo_data("langchain-ai", "langchain"), "langchain")
    rows += normalize_docs(read_repo_data("microsoft", "semantic-kernel"), "semantic-kernel")
    rows += normalize_docs(read_repo_data("openai", "openai-cookbook"), "openai-cookbook")
    chunks = build_chunks(rows)

    text_index = Index(text_fields=["text", "filename", "repo"], keyword_fields=["repo"])
    text_index.fit(chunks)

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    vectors = np.array([embedding_model.encode(c["text"]) for c in chunks])
    vector_index = VectorSearch()
    vector_index.fit(vectors, chunks)
    return chunks, text_index, embedding_model, vector_index


def text_search(query: str, text_index: Index, k: int = 5):
    return text_index.search(query, num_results=k)


def vector_search(query: str, embedding_model: SentenceTransformer, vector_index: VectorSearch, k: int = 5):
    q = embedding_model.encode(query)
    return vector_index.search(q, num_results=k)


def hybrid_search(query: str, text_index: Index, embedding_model: SentenceTransformer, vector_index: VectorSearch, k: int = 5):
    t = text_search(query, text_index, k)
    v = vector_search(query, embedding_model, vector_index, k)
    scores = {}
    items = {}
    for rank, item in enumerate(t, start=1):
        key = item["chunk_id"]
        items[key] = item
        scores[key] = scores.get(key, 0.0) + 0.7 * (1.0 / (60 + rank))
    for rank, item in enumerate(v, start=1):
        key = item["chunk_id"]
        items[key] = item
        scores[key] = scores.get(key, 0.0) + 1.0 * (1.0 / (60 + rank))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [items[key] for key, _ in ranked[:k]]


def answer_with_tool_calling(question: str, mode: str, text_index: Index, embedding_model: SentenceTransformer, vector_index: VectorSearch):
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return "Missing OPENAI_API_KEY in Streamlit secrets.", []

    client = OpenAI(api_key=api_key)

    def run_search_docs(query: str, search_mode: str = "hybrid", num_results: int = 5):
        if search_mode == "text":
            results = text_search(query, text_index, num_results)
        elif search_mode == "vector":
            results = vector_search(query, embedding_model, vector_index, num_results)
        else:
            results = hybrid_search(query, text_index, embedding_model, vector_index, num_results)
        return [
            {
                "repo": r.get("repo"),
                "filename": r.get("filename"),
                "chunk_id": r.get("chunk_id"),
                "text": (r.get("text") or "")[:900],
            }
            for r in results
        ]

    tool = {
        "type": "function",
        "name": "search_docs",
        "description": "Search docs with text/vector/hybrid mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string", "enum": ["text", "vector", "hybrid"]},
                "num_results": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }

    messages = [
        {"role": "system", "content": "Use search_docs before answering doc questions. Cite filenames."},
        {"role": "user", "content": question},
    ]
    first = client.responses.create(model="gpt-4o-mini", input=messages, tools=[tool])
    sources = []
    for item in first.output:
        if getattr(item, "type", None) == "function_call" and item.name == "search_docs":
            args = json.loads(item.arguments)
            args.setdefault("mode", mode)
            tool_results = run_search_docs(**args)
            sources = tool_results
            messages.append(item)
            messages.append({"type": "function_call_output", "call_id": item.call_id, "output": json.dumps(tool_results)})

    final = client.responses.create(model="gpt-4o-mini", input=messages, tools=[tool])
    return final.output_text, sources


st.set_page_config(page_title="AIHero Docs Agent", page_icon="🤖", layout="wide")
st.title("🤖 AIHero Docs Agent (Day 6)")
st.caption("Streamlit Cloud deployment for Day 6/7")

with st.sidebar:
    mode = st.selectbox("Search mode", ["hybrid", "text", "vector"], index=0)
    st.markdown("Set `OPENAI_API_KEY` in Streamlit app secrets.")

chunks, text_idx, emb_model, vec_idx = build_indexes()
st.write(f"Indexed chunks: **{len(chunks)}**")

question = st.text_area("Ask a question", placeholder="How do I implement tool calling for agents?", height=100)
if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = answer_with_tool_calling(question, mode, text_idx, emb_model, vec_idx)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Top Sources")
        for s in sources[:5]:
            st.write(f"- {s['repo']} | {s['filename']} | {s['chunk_id']}")
