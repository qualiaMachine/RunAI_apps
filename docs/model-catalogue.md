# Shared Model Catalogue — Draft Plan

> This doc proposes which models the pilot hosts as *standing shared endpoints* on the cluster's two
> GPUs, how we track whether they're still the right models, and what
> triggers a swap. A small, curated set of always-on endpoints — OpenAI-compatible —
> means researchers integrate against a stable API instead of learning
> RunAI or finding the best models on their own.

---

## Trade-off: variety vs. replicas

With 2× 96 GB in the pilot phase, the naive version of this choice
is hosting **more distinct capabilities** vs. **more copies of fewer
models**. Two consolidations dissolve most of the tension:

1. **Vision folds into the generalist.** Qwen3.5-27B is natively
   multimodal, so no separate VL endpoint is needed.
2. **The retrieval stack got tiny.** The 2026 embedding/reranker
   frontier is sub-1B models (see the catalogue rows below) — the
   full embedding + reranker pair now costs ~0.15 GPU, not a card.

**Recommendation: replicate the generalist — one 0.75 replica per
GPU — and keep the rest of the catalogue to lightweight retrieval
models in the remaining 0.5.**

What the two replicas buy:

- **2× throughput** for the endpoint that will carry almost all
  campus traffic (chat, RAG generation, code, agents, *and* OCR).
  Each replica's vLLM instance continuously batches tens of
  concurrent requests (~44 GB KV cache per replica); two replicas
  double that ceiling.
- **Availability.** Rolling restarts, zero-downtime model swaps, and
  node maintenance without taking the campus LLM down — the failure
  mode a single-replica catalogue simply accepts.

What replicas do **not** buy: load isolation. Both replicas sit
behind one URL and the load balancer spreads requests across them —
no one picks a replica, so a token-heavy OCR batch spreads over both
and can slow interactive chat everywhere. If measurements show batch
traffic degrading chat, the fixes are per-client rate limits,
off-peak scheduling, or splitting into two independently addressed
single-replica workloads (`general` + `general-batch`, same model,
same total VRAM) — trading pooled capacity for true isolation.

Cheaper levers still come first when saturation hits: tighten
quantization (the repo's
[8-bit vs 4-bit notebook](../8bit-vs-4bit-latency_32B.ipynb) measured
AWQ at ~1.7× the tokens/s of 8-bit), lower `--max-model-len`, cap
`max_tokens`, rate-limit per project. What we give up is a standing
mid-size utility model — high-volume cheap-token jobs share the
generalist under rate limits instead of getting their own endpoint.
If the GPU count grows (NRP allocation, future purchase), spend new
cards on variety — audio, code-specialist, long-context — not on a
third replica.

---

## Draft catalogue v0 — what to stand up

Three standing services: a **replicated multimodal generalist** (one
0.75 replica per GPU) plus a lightweight embedding + reranker stack.
There is **no separate vision endpoint** — the generalist is natively
multimodal, and the Qwen team reports the 3.5 generation outperforms
the dedicated Qwen3-VL line across visual-understanding benchmarks,
so document extraction / OCR traffic rides on `general` too. The
[OCR validation gate](#ocr-validation-gate) below is the safety check
before `ocr_app` re-points to it. And there is **no small/fast
model** — with two generalist replicas there's enough pooled
capacity that high-volume jobs share the same endpoint under
per-client rate limits instead of getting a dedicated one.

> **On the generalist pick:** **Qwen3.5-27B** — the Qwen3.5
> generation's dense model — leads its VRAM class across metrics
> (reported MMLU-Pro 86.1, GPQA Diamond 85.5, vs. ~79/69 for
> Qwen3-VL-32B), is **natively multimodal** (unified vision-language,
> early-fusion), has 262k native context, Apache 2.0, an official FP8
> release, and vLLM support. It clearly supersedes both Qwen3-32B and
> Gemma 3 27B for this slot, and it keeps the catalogue in the model
> family the cluster already caches. **Qwen3.6-27B** (April 2026, also
> dense + multimodal, notably stronger on coding/agentic benchmarks —
> SWE-bench Verified 77.2) is the immediate candidate under
> [trigger #1](#triggers-for-changing-the-catalogue); benchmark both
> during stand-up and let the table decide.

### The catalogue table

One table, one row per endpoint: identity, the case for standing it
up, and the published quality benchmarks that justify the pick. This
is the **living record** — the
[triggers](#triggers-for-changing-the-catalogue) watch these columns,
and a row that can't fill its justification and benchmark cells
shouldn't be standing.

| Endpoint | Model | GPU fraction | Why it earns a slot | Quality benchmarks (published) | Last reviewed |
|----------|-------|--------------|---------------------|-------------------------------|---------------|
| `general` ×2 replicas | `Qwen/Qwen3.5-27B-FP8` (~28 GB/replica) | 0.75 + 0.75 (one per GPU) | The workhorse: chat, RAG generation, code assistance, agent backends, **and all vision/OCR traffic** (natively multimodal — no separate VL endpoint needed). Two replicas → 2× throughput and rolling restarts (one shared URL; the load balancer spreads requests — see trade-off section for what that does *not* isolate). ~44 GB KV cache per replica. | [MMLU-Pro 86.1 · GPQA-Diamond 85.5](https://apxml.com/models/qwen35-27b) · [LiveCodeBench 80.7](https://llm-stats.com/models/qwen3.5-27b). No 27B-specific published OCRBench/DocVQA yet — measure at the [OCR gate](#ocr-validation-gate) ([Qwen3.5 flagship: MMMU 85.0, OmniDocBench 90.8](https://techie007.substack.com/p/qwen-35-the-complete-guide-benchmarks)) | 2026-07 |
| `embedding` | `jinaai/jina-embeddings-v5-text` (~0.7 B, ~2 GB) *(alternates below)* | 0.10 | Every RAG project on campus needs vectors; the 2026 frontier is sub-1B models, so this costs almost nothing. Swapping later forces every consumer to rebuild indexes — run a golden-set eval on real campus corpora before locking. | [MTEB v2 71.7 at 677M params](https://app.ailog.fr/en/blog/news/rag-benchmark-mteb-2026) — beats Qwen3-Embedding-8B ([70.6](https://github.com/QwenLM/Qwen3-Embedding)) at ~1/10 the size. Alternates: [Harrier-OSS-v1 (MIT, 74.3 multilingual MTEB v2)](https://app.ailog.fr/en/blog/news/rag-benchmark-mteb-2026) — verify size + vLLM support; [Qwen3-Embedding-4B, 69.45](https://github.com/QwenLM/Qwen3-Embedding) (Apache) if Jina's CC-BY-NC license is a problem | 2026-07 |
| `reranker` | `jinaai/jina-reranker-v3` (~0.6 B, ~1.5 GB) | 0.05 | Cheapest retrieval-quality upgrade there is; completes the retrieval stack behind `embedding`. Listwise architecture — fast enough for interactive RAG. | [BEIR nDCG@10 61.94 at 0.6B](https://arxiv.org/pdf/2509.25085) ([model card](https://huggingface.co/jinaai/jina-reranker-v3)); [strongest sub-200 ms option in 2026 comparisons](https://aimultiple.com/rerankers). Alternate: [Qwen3-Reranker-0.6B](https://github.com/QwenLM/Qwen3-Embedding) (Apache) — but [autoregressive scoring makes it slower per query](https://aimultiple.com/rerankers) | 2026-07 |
| *(open)* | — | 0.10 (GPU 0) + 0.25 (GPU 1) | The rest of the small-model budget. Held for [demand trigger #3](#triggers-for-changing-the-catalogue) — Whisper/transcription, a code specialist, long-context. | — | 2026-07 |
| **Σ VRAM** | **~60 GB weights** | GPU 0: 0.90 used · GPU 1: 0.75 used | Of 192 GB total: ~31 GB weights on GPU 0 (replica + retrieval stack), ~28 GB on GPU 1 (replica). The remaining ~130 GB is the point — KV cache for concurrency, plus the open fractions. | — | — |

**Measured columns come later.** Once endpoints stand up, extend the
table with on-cluster numbers — TTFT p95 (*time to first token*: how
long a user waits before the response starts streaming; *p95* = the
slowest 1-in-20 requests, which captures how bad it feels under load
better than the average), tokens/s at N concurrent,
max concurrency, Wh/1k output tokens
(via [`scripts/hardware_metrics.py`](../scripts/hardware_metrics.py)
and the [latency notebook](../8bit-vs-4bit-latency_32B.ipynb) pattern),
and requests/week. Those columns feed the saturation and idle
triggers, and Wh/1k tokens is the first number peer campuses will ask
for. They're omitted until there's real data to put in them.

Keep the canonical copy of this table in this file; if/when other
campuses want to consume it, export the same columns as CSV/JSON
(see [Growth path](#growth-path-nrp-and-peer-campuses)).

### OCR validation gate

`ocr_app` currently runs against `Qwen3-VL-32B-Instruct-AWQ`, and
those weights stay on the PVC either way. Before the app re-points to
`general`:

1. Run the chunk-extract pipeline through `general` on the app's
   sample documents; compare extraction quality against the VL-32B
   baseline (plus OCRBench/DocVQA for the table).
2. If the 27B regresses on layout-heavy documents, drop the GPU 1
   `general` replica and stand `vision` (VL-32B-AWQ) back up on that
   card — the pre-replica layout, nothing lost.
3. If it passes, watch for contention: OCR batches share the same
   load-balanced endpoint as chat, so rate-limit batch clients and
   schedule large runs off-peak. If chat TTFT p95 degrades anyway,
   that's [trigger #2](#triggers-for-changing-the-catalogue) — and
   the split-into-`general`+`general-batch` option in the trade-off
   section is the escalation path.

**Explicitly not a standing service:** 70B+ models tensor-parallel
across both cards (Qwen2.5-72B, Qwen3-VL-72B, ~150B at 4-bit — all
feasible per [00 Overview](00-overview.md#hardware)). One TP-2 model
consumes the whole cluster. Offer these **on-request / scheduled**
(e.g. a lab books a week for evaluation), never as catalogue rows.

### ML Marathon mode

Marathon events are bursty: many students, small requests, short
window. Plan: keep the catalogue up, give marathon teams the shared
`general` endpoint with per-team rate limits, and use the open
fractions for event scratch. If a marathon project needs a model
outside the catalogue, that's the on-request path above — not a
permanent row.

---

## Triggers for changing the catalogue

A model gets *reviewed* on any trigger below; it gets *replaced* only
after the [swap process](#swap-process) passes.

| # | Trigger | Threshold | Likely action |
|---|---------|-----------|---------------|
| 1 | **Frontier gap** | A vLLM-supported open model in the same VRAM class leads the hosted model by **≥5 points** on the role's tracked benchmark | Candidate swap |
| 2 | **Saturation** | Time-to-first-token p95 > 10 s (the slowest 5% of users wait 10+ s before the response starts) or requests sitting in vLLM's queue, persisting **2+ consecutive normal weeks** across both `general` replicas | Cheapest fix first: tighter quant (frees VRAM for concurrency) → lower max context → rate limits → *then* discuss reclaiming open fractions or new hardware |
| 3 | **Unmet demand** | **≥3 distinct groups** request a capability the catalogue lacks (long-context, audio, code-specialist, …) | Slot review — what gets evicted or shrunk to make room? |
| 4 | **Idle** | An endpoint below ~20 req/wk for a full quarter | Demote to on-request; free the fraction |
| 5 | **Upstream event** | Model deprecated, license change, vLLM drops support, security advisory in the serving stack | Immediate review |
| 6 | **Cadence** | **Quarterly** review of every row, regardless — refresh Last reviewed, re-check the benchmark column against the current frontier | Keeps the table honest between events |

### Swap process

1. Open a GitHub issue in this repo: proposed model, role, published
   benchmark scores, VRAM/quant plan, what it replaces.
2. Provision weights to the shared PVC
   ([managing-models.md](../rag_app/docs/managing-models.md) — check
   vLLM compatibility first).
3. Stand up on GPU 0's unallocated fraction (or off-hours) and measure
   the same columns as the incumbent: benchmarks, TTFT, tok/s,
   Wh/1k tokens.
4. It wins on the numbers → announce with **2 weeks' notice**, swap
   the endpoint, update the table, keep the old weights on the PVC
   for one review cycle for rollback.
5. Update this doc and any per-app docs whose model defaults changed
   (see the repo's `CLAUDE.md` doc-tracking table).

---

## Growth path: NRP and peer campuses

Design choices above that keep the federation door open:

- **OpenAI-compatible everywhere.** Every endpoint speaks `/v1` (vLLM
  default), so a client — or a gateway — can route across sites
  without code changes.
- **Portable catalogue schema.** The catalogue table's columns are
  site-agnostic; exporting them as CSV/JSON lets each
  "BadgerBrain-like" deployment publish its own catalogue, and a
  LiteLLM-style gateway can eventually federate them (route to
  whichever site hosts the requested model, with local overflow to
  NRP-hosted endpoints for burst or 70B+ requests).
- **Energy column as a shared benchmark.** Wh/1k tokens measured the
  same way at each site is a rare apples-to-apples number across
  heterogeneous hardware — and a natural first joint publication with
  NRP/peer institutions.
- **Division of labor.** Local GPUs are the long-term home for
  workloads that will eventually touch institutional/sensitive data
  (pending security review); NRP is the natural overflow for large
  on-request models and burst capacity on public data.

Concrete next step when a collaboration starts: exchange catalogue
exports + measurement methodology before exchanging any traffic.

---

## Stand-up checklist (v0)

- [ ] Provision `Qwen/Qwen3.5-27B-FP8`, `jinaai/jina-embeddings-v5-text`,
      and `jinaai/jina-reranker-v3` to the shared PVC
      (`Qwen3-VL-32B-AWQ` and Jina V4 are already cached); confirm the
      Jina models' CC-BY-NC licensing is acceptable for the pilot, else
      fall back to the Apache `Qwen3-Embedding-4B` / `Qwen3-Reranker-0.6B`
- [ ] Deploy `general` as a RunAI Inference workload with **2 replicas
      at 0.75 GPU** (one lands on each card), plus the embedding and
      reranker endpoints, per [03 Share as endpoint](03-share-as-endpoint.md)
- [ ] Run the benchmark + `hardware_metrics.py` pass; add the
      measured columns to the table
- [ ] Before locking the embedding model, run a 100–500-query golden
      set built from real campus corpora (embedding swaps force every
      consumer to reindex — this is the highest-stakes pick); check
      `Harrier-OSS-v1` size/vLLM support while at it
- [ ] Benchmark `Qwen3.6-27B` head-to-head with `Qwen3.5-27B` during
      stand-up (trigger #1 candidate — stronger coding/agentic scores)
- [ ] Run the [OCR validation gate](#ocr-validation-gate) — chunk-extract
      on `ocr_app`'s sample docs through `general` vs. the VL-32B-AWQ
      baseline — before re-pointing `ocr_app` and updating its docs
- [ ] Publish endpoint URLs + this catalogue to pilot users; set the
      first quarterly review date
