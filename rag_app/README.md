# WattBot RAG

Retrieval-augmented generation over research paper corpora. Streamlit chat UI backed by vLLM, Jina V4 embeddings, and optional cross-encoder reranking. 2025 WattBot Challenge winner.

## Architecture

```
  +---------------------+
  |   Streamlit App     |  (Workspace, CPU only)
  |   Port 8501         |
  +--+------+-------+---+
     |      |       | HTTP (internal cluster DNS)
     v      v       v
  +------+ +------+ +----------+
  | vLLM | |Embed | | Reranker |
  | 8000 | | 8080 | |   8082   |
  |GPU80%| |GPU10%| | GPU 10%  |
  +------+ +------+ +----------+
```

All 4 services fit on ~1 GPU via fractional allocation. Reranker is optional.

## RunAI Deployment

Full deployment guide: **[docs/README.md](docs/README.md)**

Assumes the cluster's `shared-models` data volume is available
(admin-provisioned). To own and write to your own instead, see
[Provision Your Own Shared Models PVC](docs/setup-shared-models.md)
*(advanced)*.

Follow these docs in order:

1. [Setup Workspace](docs/setup-workspace.md) — clone repo, build vector index
2. [Deploy vLLM Server](docs/deploy-vllm.md) — LLM inference with Qwen 7B
3. [Deploy Embedding Server](docs/deploy-embedding.md) — Jina V4 query encoding
4. [Deploy Reranker Server](docs/deploy-reranker.md) *(optional)*
5. [Deploy Streamlit App](docs/deploy-streamlit.md) — browser UI

Additional: [Troubleshooting](docs/troubleshooting.md) | [Managing Models](docs/managing-models.md) | [Reference](docs/reference.md)

## Key Files

```
rag_app/
├── app.py                          # Streamlit chat UI
├── pages/1_Corpus.py               # Corpus exploration page
├── vendor/
│   ├── KohakuRAG/                  # RAG engine
│   └── KohakuVault/                # Rust+PyO3 SQLite vector store
├── scripts/
│   ├── embedding_server.py         # FastAPI Jina V4 server
│   ├── reranker_server.py          # FastAPI cross-encoder server
│   └── add_papers.py               # Corpus management
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── data/                           # Corpus, embeddings, metadata
├── docs/                           # Deployment guides (10 docs)
├── requirements_local.txt          # GPU/local inference deps
└── requirements_remote.txt         # Remote client deps (minimal)
```
