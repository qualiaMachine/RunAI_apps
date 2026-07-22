# Shared Model Catalogue — Draft Plan

> **Status: draft / not yet stood up.** This doc proposes which models
> the pilot hosts as *standing shared endpoints* on the cluster's two
> GPUs, how we track whether they're still the right models, and what
> triggers a swap. It assumes the concepts from
> [00 Overview](00-overview.md) (hardware, workload types) and the
> serving patterns from [03 Share as endpoint](03-share-as-endpoint.md).
> Constraints from the [Usage Policy](usage-policy.md) apply — public
> data only during the pilot.

## Why a catalogue at all

Individual labs standing up their own copies of the same 7B model is
the failure mode this cluster exists to avoid. A small, curated set of
always-on endpoints — one URL per capability, OpenAI-compatible —
means researchers integrate against a stable API instead of learning
RunAI, and the two GPUs serve many groups instead of two.

The catalogue is also the unit of collaboration: if NRP or peer
universities stand up their own "BadgerBrain-like" deployments, a
published catalogue (models + endpoints + measured performance) is
what makes federation possible later. See
[Growth path](#growth-path-nrp-and-peer-campuses).

---

## The core trade-off: variety vs. replicas

With 2× 96 GB and a campus-wide audience, we can either host **more
distinct capabilities** or **more copies of fewer models**.

**Recommendation: optimize for variety. Do not budget standing
replicas at this scale.**

Rationale:

1. **vLLM already multiplexes users.** Continuous batching means one
   replica of a ~30B quantized model serves tens of concurrent
   requests; the real ceiling is KV-cache memory, not replica count.
   On a 96 GB card, a ~20–35 GB (AWQ/FP8) model leaves 40–60 GB of
   KV cache — that *is* the multi-user capacity.
2. **A replica costs a whole capability.** Duplicating the generalist
   across both cards buys roughly 2× throughput but evicts the
   embedding/reranker/small-fast stack — the things that make the
   cluster useful for RAG pipelines and high-volume jobs, which is
   where campus demand actually is.
3. **There are cheaper levers than replicas** when an endpoint
   saturates, in order: tighten quantization (the repo's
   [8-bit vs 4-bit notebook](../8bit-vs-4bit-latency_32B.ipynb)
   measured AWQ at ~1.7× the tokens/s of 8-bit for Qwen3-VL-32B),
   lower `--max-model-len`, cap `max_tokens` per request, add rate
   limits per project. Only after those are exhausted does a replica
   enter the conversation — as a *measured* response to the
   [saturation trigger](#triggers-for-changing-the-catalogue), not a
   default.
4. **Availability is the honest cost of no replicas.** A restart or
   node drain means that endpoint is down. During the pilot that is
   acceptable (and already what the Usage Policy promises); the
   small-fast model on the other card doubles as a degraded-mode
   fallback for the generalist.

Revisit this recommendation if the GPU count grows (NRP allocation,
future purchase) — at 4+ cards, one replica of the highest-traffic
endpoint becomes the right call.

---

## Draft catalogue v0 — what to stand up

Four standing endpoints: the baseline capabilities (general-purpose
LLM, embedding, reranker) plus a small high-throughput model. There
is **no separate vision endpoint** — the generalist is natively
multimodal, and the Qwen team reports the 3.5 generation outperforms
the dedicated Qwen3-VL line across visual-understanding benchmarks,
so document extraction / OCR traffic rides on `general` too. That
consolidation is the biggest capacity win in this plan; the
[OCR validation gate](#ocr-validation-gate) below is the safety check
before `ocr_app` re-points to it.

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
up, the quality benchmarks that justify the pick, and performance
measured on our hardware. This is the **living record** — the
[triggers](#triggers-for-changing-the-catalogue) watch these columns,
and a row that can't fill its justification and benchmark cells
shouldn't be standing. Blank cells (—) get filled during stand-up.

| Endpoint | Model | GPU (frac) | Why it earns a slot | Quality benchmarks (published) | Gap-to-frontier | TTFT p95 | Tok/s @ N | Max conc. | Wh/1k tok | Req/wk | Last reviewed |
|----------|-------|-----------|---------------------|-------------------------------|-----------------|----------|-----------|-----------|-----------|--------|---------------|
| `general` | `Qwen/Qwen3.5-27B-FP8` (~28 GB) | 0 (0.80) | The workhorse: chat, RAG generation, code assistance, agent backends, **and all vision/OCR traffic** (natively multimodal — no separate VL endpoint needed). ~45 GB left for KV cache → long context, many concurrent users. | MMLU-Pro 86.1 · GPQA-D 85.5 · LiveCodeBench — · OCRBench — · DocVQA — | — | — | — | — | — | — | — |
| `embedding` | `Qwen/Qwen3-Embedding-4B` BF16 (~8 GB) *(alt: Jina V4, on PVC, proven in `rag_app`)* | 1 (0.15) | Every RAG project on campus needs vectors; smallest footprint, widest leverage. Swapping later forces every consumer to rebuild indexes — pick carefully once. | MTEB v2 retrieval NDCG@10 — | — | — | — | — | — | — | — |
| `reranker` | `Qwen/Qwen3-Reranker-4B` BF16 (~8 GB) | 1 (0.10) | Cheapest retrieval-quality upgrade there is; completes the retrieval stack behind `embedding`; pattern proven in `rag_app`. | MTEB reranking — | — | — | — | — | — | — | — |
| `small-fast` | `Qwen/Qwen3-8B-FP8` (~9 GB) | 1 (0.30) | Throughput tier: batch/classification sweeps, ML Marathon traffic (isolated from `general`), degraded-mode fallback when `general` restarts. Justified by tok/s and Wh/1k tok, not quality scores. | MMLU-Pro — | — | — | — | — | — | — | — |
| *(open slot)* | — | 1 (0.45) | Freed by folding vision into `general`. Held for [demand trigger #3](#triggers-for-changing-the-catalogue) — Whisper/transcription, a code specialist, long-context — and the rollback slot if the OCR gate fails. | — | — | — | — | — | — | — | — |
| *(headroom)* | — | 0 (0.20) | Burst scratch: ML Marathon experiments, benchmark runs, candidate models during [swaps](#swap-process). | — | — | — | — | — | — | — | — |

Weights on GPU 1 total ~25 GB across three services, leaving generous
KV cache plus the open slot — same fractional-GPU pattern `rag_app`
already runs in production.

**What each column means:**

| Column | Meaning |
|--------|---------|
| Why it earns a slot | The standing justification. If this cell goes stale (usage dies, capability absorbed elsewhere), the row is a demotion candidate regardless of benchmarks. |
| Quality benchmarks | Role-specific published scores — generalist: MMLU-Pro, GPQA-Diamond, LiveCodeBench, OCRBench, DocVQA; embedding: MTEB v2 retrieval; reranker: MTEB reranking. Never compare across roles. |
| Gap-to-frontier | Points behind the best vLLM-servable open model in the same VRAM class on the same benchmarks. The number [trigger #1](#triggers-for-changing-the-catalogue) watches. |
| TTFT p95 / Tok/s @ N / Max conc. | Measured on-cluster at realistic concurrency (record N), via [`scripts/hardware_metrics.py`](../scripts/hardware_metrics.py) and the [latency notebook](../8bit-vs-4bit-latency_32B.ipynb) pattern — not vendor numbers. |
| Wh/1k tok | Energy per 1k output tokens — the WattBot angle, and the first number peer campuses will ask for. |
| Req/wk | Usage; drives the idle and saturation triggers. |

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
2. If the 27B regresses on layout-heavy documents, stand `vision`
   back up in the open slot — the catalogue grows back to five rows
   and nothing is lost.
3. If it passes, watch for contention: OCR batches are token-heavy,
   and they now share `general` with chat traffic. Schedule large
   batch runs off-peak or rate-limit them per project; if `general`'s
   TTFT p95 still degrades, that's [trigger #2](#triggers-for-changing-the-catalogue)
   and a dedicated vision endpoint returns to the open slot.

**Explicitly not a standing service:** 70B+ models tensor-parallel
across both cards (Qwen2.5-72B, Qwen3-VL-72B, ~150B at 4-bit — all
feasible per [00 Overview](00-overview.md#hardware)). One TP-2 model
consumes the whole cluster. Offer these **on-request / scheduled**
(e.g. a lab books a week for evaluation), never as catalogue rows.

### ML Marathon mode

Marathon events are bursty: many students, small requests, short
window. Plan: keep the catalogue up, point marathon traffic at
`small-fast` and `general` with per-team rate limits, and repurpose
GPU 0's 0.20 headroom for event scratch. If a marathon project needs a
model outside the catalogue, that's the on-request path above — not a
permanent row.

---

## Triggers for changing the catalogue

A model gets *reviewed* on any trigger below; it gets *replaced* only
after the [swap process](#swap-process) passes.

| # | Trigger | Threshold | Likely action |
|---|---------|-----------|---------------|
| 1 | **Frontier gap** | A vLLM-supported open model in the same VRAM class leads the hosted model by **≥5 points** on the role's tracked benchmark | Candidate swap |
| 2 | **Saturation** | TTFT p95 > 10 s or sustained queueing during **2+ consecutive normal weeks** | Tighter quant → lower max context → rate limits → *then* discuss a replica |
| 3 | **Unmet demand** | **≥3 distinct groups** request a capability the catalogue lacks (long-context, audio, code-specialist, …) | Slot review — what gets evicted or shrunk to make room? |
| 4 | **Idle** | An endpoint below ~20 req/wk for a full quarter | Demote to on-request; free the fraction |
| 5 | **Upstream event** | Model deprecated, license change, vLLM drops support, security advisory in the serving stack | Immediate review |
| 6 | **Cadence** | **Quarterly** review of every row, regardless — refresh Last reviewed, re-check gap-to-frontier | Keeps the table honest between events |

### Swap process

1. Open a GitHub issue in this repo: proposed model, role, published
   benchmark scores, VRAM/quant plan, what it replaces.
2. Provision weights to the shared PVC
   ([managing-models.md](../rag_app/docs/managing-models.md) — check
   vLLM compatibility first).
3. Stand up on GPU 0's headroom fraction (or off-hours) and measure
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

- [ ] Provision `Qwen/Qwen3.5-27B-FP8`, `Qwen/Qwen3-8B` (FP8),
      `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Reranker-4B` to the
      shared PVC (`Qwen3-VL-32B-AWQ` and Jina V4 are already cached)
- [ ] Deploy the four endpoints per [03 Share as endpoint](03-share-as-endpoint.md),
      fractions as tabled above
- [ ] Run the benchmark + `hardware_metrics.py` pass; fill the
      measured columns
- [ ] Decide Jina V4 vs. Qwen3-Embedding-4B on MTEB + measured latency
      (Jina wins ties: already proven in `rag_app` and multimodal)
- [ ] Benchmark `Qwen3.6-27B` head-to-head with `Qwen3.5-27B` during
      stand-up (trigger #1 candidate — stronger coding/agentic scores)
- [ ] Run the [OCR validation gate](#ocr-validation-gate) — chunk-extract
      on `ocr_app`'s sample docs through `general` vs. the VL-32B-AWQ
      baseline — before re-pointing `ocr_app` and updating its docs
- [ ] Publish endpoint URLs + this catalogue to pilot users; set the
      first quarterly review date
