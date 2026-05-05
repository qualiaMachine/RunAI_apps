# 05 — Examples

> **Step 5** in the [New User Guide](../README.md#new-user-guide). You should have
> worked through [00 Overview](00-overview.md), [02 First
> Workspace](02-first-workspace.md), [03 Share a Model as a vLLM
> Endpoint](03-share-as-endpoint.md), and [04 Storage](04-storage.md)
> before picking an example to deploy.

The repo currently has two example deployments — PoCs that have been
exercised on real data and are growing as more use cases come up.
They're meant to be copied and adapted, not used as-is. Pick the one
closest to your use case and follow that app's README.

## Document extraction (OCR pipeline)

> **[`ocr_app/`](../ocr_app/README.md)**

Vision-language extraction of structured JSON from PDFs and scanned
images. Built around Qwen3-VL-32B-Instruct-AWQ. Chunk-based two-pass
notebook pipeline: per-chunk extraction with continuation flags,
programmatic merging across chunks, doc-level synthesis pass.

Use this if you're processing:
- Grant administration documents (award notices, budgets, terms,
  proposals) — `notebooks/RSP_example_extraction_pipeline.ipynb`
- Library / archival material (books, manuscripts, sheet music, maps,
  multilingual scans) — `notebooks/library_extraction_pipeline.ipynb`

Use this as a *pattern* if you're processing anything page-shaped where
layout, tables, signatures, or annotations matter — fork the per-chunk
prompt and the merge logic, leave the rest.

Storage: just the workspace's inline volume for inputs (PoC) or an NFS
/ PVC Data Source (production). Output JSON lives next to the notebook
inside the workspace volume — no separate output PVC needed.

## Retrieval-augmented chat (RAG pipeline)

> **[`rag_app/`](../rag_app/README.md)**

A multi-user chatbot over a curated corpus. Streamlit UI backed by
vLLM (Qwen 7B/14B/72B), Jina V4 embeddings, and an optional
cross-encoder reranker. Each component is its own RunAI Inference
workload so they can scale independently. 2025 WattBot Challenge
winner.

Use this if you're building:
- "ChatGPT for our research papers / lab manuals / institutional
  knowledge"
- Q&A over a fixed document set with citation-style sourcing
- Anything where multiple users hit one model server concurrently
  (continuous batching matters)

Use this as a *pattern* if you want autoscaling Inference workloads,
fractional GPU allocation, or service-to-service calls over internal
cluster DNS.

Storage: a project PVC for the vector index (~130 MB), the
cluster-wide `shared-models` Data Volume for model weights (read-only,
admin-provisioned).

## Building your own

There's nothing magic about either app. The actual RunAI patterns are:

- **Workspace + Jupyter** for interactive development → see
  [02 First Workspace](02-first-workspace.md)
- **Inference workload** behind an autoscaling Knative endpoint, called
  via internal cluster DNS → see [`rag_app/docs/deploy-vllm.md`](../rag_app/docs/deploy-vllm.md)
  as a template
- **Shared model weights** via the cluster-wide Data Volume so you
  don't re-download a 20 GB checkpoint per workload → see
  [04 Storage](04-storage.md) and
  [`rag_app/docs/setup-shared-models.md`](../rag_app/docs/setup-shared-models.md)
- **Project PVCs** for outputs and shared in-project state → see
  the storage walkthrough in [04 Storage](04-storage.md)

If you can describe your workload as some combination of those
patterns, you can build it from these examples. If you can't, that's a
good time to ask your DoIT contact whether the cluster is the right
tool for the job.

## Next

This is the end of the new-user guide. From here you go into one of
the app READMEs — [`ocr_app/README.md`](../ocr_app/README.md) for
document extraction or [`rag_app/README.md`](../rag_app/README.md)
for retrieval-augmented chat — and follow that app's deployment
steps. Both reference back into this guide for the underlying
concepts when something looks new.
