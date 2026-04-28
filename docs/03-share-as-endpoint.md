# 03 — Share a Model as a vLLM Endpoint

> **Step 3** in the [New User Guide](../README.md#new-user-guide). Builds directly on
> [02 First Workspace](02-first-workspace.md) — assumes you already
> have `Qwen2.5-7B-Instruct` loading from the shared-models Data
> Volume in a workspace.

In 02 you loaded the model directly into the notebook kernel — fine
for one user, but it pins a GPU fraction to your workspace whether
you're actively prompting or not, and a second user wanting the same
model has to load their own copy. The cluster pattern for "many users,
one model" is to deploy the model **once** as a vLLM Inference
workload, then have everyone's notebooks call its HTTP endpoint over
internal cluster DNS.

This doc walks through that conversion: take the Qwen2.5-7B example
from 02, host it as an autoscaling endpoint, then revisit the
notebook in a workspace with **zero GPU** that calls the endpoint
instead of loading weights.

By the end you'll have:
- A `qwen-Qwen2.5--7B--Instruct` Inference workload exposing an
  OpenAI-compatible HTTP endpoint
- The same workspace from 02, now reconfigured to 0 GPU, with a
  notebook that gets the same answer by calling the endpoint
- An intuition for when each pattern is appropriate

You'll tear both down at the end. At the current 2-GPU pilot scale
we don't leave per-project endpoints running — see [Step
D](#step-d-tear-down-when-youre-done) for the full reasoning. The
pattern itself is what matters here, not the persistent artifact.

## When to host vs load directly

Use the rule of thumb:

| | Direct load (02 pattern) | Endpoint (this doc) |
|---|---|---|
| **GPU cost** | Pinned to the workspace whenever it's running | Shared across all callers; autoscales from 0 if idle |
| **Cold start** | ~30–60s once per workspace start | ~30s once per autoscale; near-zero per call |
| **Concurrency** | Single-threaded `model.generate()` blocks | Continuous batching — 2–4× more throughput |
| **Setup effort** | One workspace with a notebook | One Inference workload + workspaces that consume it |
| **Best for** | Solo experimentation, debugging, testing future RunAI apps | Anything multi-user, anything called from multiple workloads, anything that should scale |

If you're the only user and you'll prompt the model interactively for
an hour and stop, just stay on the 02 pattern. As soon as a second
person wants the same model — or the same workload itself wants to be
called by another service — host it.

### Where this is heading

The walkthrough below has you spin up an endpoint for one model in
your own project, but the bigger play once the cluster scales past
the 2-GPU pilot pod is a small, curated set of **always-on shared
endpoints** that any lab can call without doing any of this setup
themselves. Picture a per-cluster catalog along the lines of:

| Endpoint | Use case |
|----------|----------|
| `qwen-Qwen2.5--7B--Instruct` / `--14B--Instruct` / `--72B--Instruct` | General-purpose chat at three size/cost points |
| `qwen-Qwen3--VL--32B--Instruct--AWQ` | Vision-language extraction (the OCR app already shares one of these) |
| `meta--Llama--3.1--8B--Instruct` / `--70B--Instruct` | Llama-family alternative for labs that prefer it |
| `bge--reranker--v2--m3` or similar | Cross-encoder reranker for RAG |
| `jinaai--jina--embeddings--v4` | Multilingual embeddings (the RAG app already shares this) |

Each one would be a single Inference workload, autoscaled `min=0` so
unused models release their GPU, weights mounted read-only from the
cluster-wide `shared-models` Data Volume so there's exactly one copy
to maintain per model. A lab that wants to build a RAG over their
papers wouldn't deploy any of these — they'd just point their
notebook or app at
`http://qwen-Qwen2.5--72B--Instruct.runai-shared-models.svc.cluster.local/v1`
and start asking questions.

We're not there yet — at 2 GPUs, hosting more than two or three
models simultaneously isn't realistic, and the current pilot only
runs one shared endpoint (`qwen3--vl--32b--instruct-awq` for OCR).
But this is the direction, and every endpoint a lab stands up under
this doc's pattern is one less wheel that has to be reinvented when
the cluster grows. If your use case would benefit from a specific
model being available cluster-wide, tell Chris/Mike — that's how the
catalog gets prioritized.

## Step A. Deploy Qwen2.5-7B as a vLLM Inference workload

In the RunAI UI:

1. **Workloads** > **+ NEW WORKLOAD** > **Inference**
2. Pick the **Custom** inference type (you're bringing your own image
   and command, not using a built-in template).
3. Basic settings:
   - **Project:** your project
   - **Workload name:** `qwen-Qwen2.5--7B--Instruct`

   > **Naming convention.** RunAI workload names can't contain `/`,
   > so the convention used elsewhere in this repo (and what the
   > shared OCR endpoint follows) is: replace `/` with `-` and every
   > existing `-` with `--`. So the HuggingFace ID
   > `Qwen/Qwen2.5-7B-Instruct` becomes the workload name
   > `qwen-Qwen2.5--7B--Instruct` — fully reversible, lowercases the
   > org prefix (Kubernetes service names have to start with a lowercase
   > letter), and preserves the model's own capitalization so it's
   > obvious which model the workload is hosting.
4. **Environment image** — Custom:
   - **Image URL:** `vllm/vllm-openai:latest` — same image every
     other deployment in this repo uses
     ([rag_app/docs/deploy-vllm.md](../rag_app/docs/deploy-vllm.md),
     [ocr_app/docs/deploy-vllm.md](../ocr_app/docs/deploy-vllm.md)).
     The image entrypoint already runs
     `python -m vllm.entrypoints.openai.api_server`, so all
     configuration is via Arguments below.
5. **Runtime settings:**
   - **Command:** *(leave empty — vLLM's image entrypoint runs
     `python -m vllm.entrypoints.openai.api_server`)*
   - **Arguments:**
     ```
     Qwen/Qwen2.5-7B-Instruct --dtype auto --max-model-len 8192
     ```

     The model ID goes in as a **positional argument**, not behind a
     `--model` flag — that's what `vllm.entrypoints.openai.api_server`
     expects, and it's the shape every vLLM workload in this repo
     uses. `--dtype auto` lets vLLM pick bf16 here (same precision as
     02's direct load); `--max-model-len 8192` caps KV cache so the
     model fits comfortably in a 50% GPU fraction. Don't set
     `--gpu-memory-utilization` here — RunAI's GPU fraction setting
     below already constrains the pod's allocation.

     > **Save GPU when you can — quantize.** A 7B model in bf16 at
     > 50% of a 96 GB card has plenty of headroom, so this
     > walkthrough doesn't bother. But the moment you move to 14B,
     > 32B, 70B+, or want to stack multiple endpoints on one GPU,
     > quantization is the lever that buys you back the headroom.
     > Two vLLM routes for the same Qwen 2.5 7B example:
     >
     > **bf16 baseline** *(this walkthrough)* — full precision, no
     > quantization, runs on any HF model:
     > ```
     > Qwen/Qwen2.5-7B-Instruct --dtype auto --max-model-len 8192
     > ```
     >
     > **bitsandbytes** — works on the same `Qwen/Qwen2.5-7B-Instruct`
     > weights you already have cached. ~2× memory savings, small
     > accuracy/throughput cost, no separate model build needed:
     > ```
     > Qwen/Qwen2.5-7B-Instruct --quantization bitsandbytes --load-format bitsandbytes --dtype auto --max-model-len 8192
     > ```
     >
     > **AWQ** — uses the pre-quantized build
     > `Qwen/Qwen2.5-7B-Instruct-AWQ` *(note: different model ID
     > from the bf16 / bitsandbytes version)*. Faster and more
     > accurate than bitsandbytes when an AWQ build exists.
     > **`HF_HUB_OFFLINE=1` will hard-fail if `Qwen/Qwen2.5-7B-Instruct-AWQ`
     > isn't already cached on the `shared-models` volume — verify
     > with the listing cell from 02, and if it's missing email
     > Chris/Mike to add it.**
     > ```
     > Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin --dtype half --max-model-len 8192
     > ```
     >
     > With either quantized form on a 7B you can drop GPU
     > fractioning from `50%` to `25%` and still have headroom —
     > frees up half a card for someone else's workload.
     >
     > See [`rag_app/docs/deploy-vllm.md`](../rag_app/docs/deploy-vllm.md)
     > for the full table of model + flag combinations the rag_app
     > has tested (Qwen 14B, 30B-A3B, 32B, 72B, OpenScholar 8B).
     > For the conceptual side — what bf16 vs int8 vs int4 actually
     > does to a model and when each makes sense — the ML-X Nexus
     > has a primer:
     > [Quantization and Precision](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Notebooks/Quantization-and-Precision.html).

   - **Environment variables:**

     | Name | Value |
     |------|-------|
     | `HF_HOME` | `/models/.cache/huggingface` |
     | `HF_HUB_CACHE` | `/models/.cache/huggingface` |
     | `HF_HUB_OFFLINE` | `1` |

6. **Compute resources:**
   - **GPU devices:** `1`
   - **GPU fractioning:** Enabled — `50%` (Qwen 7B in bf16 + KV cache
     fits comfortably on 48 GB of a 96 GB card)
7. **Data & storage:** **+ Data Volume** > `shared-models`, mount path
   `/models`, read-only.
8. **Endpoint:**
   - **Container port:** `8000`
   - **Auth:** Internal (no external ingress needed — only your
     project's workloads will call it)
9. **Autoscaling:**
   - **Min replicas:** `0` (so it scales to zero when idle and
     releases the GPU)
   - **Max replicas:** `1`
   - **Metric:** `Concurrency`, value `4`
10. **CREATE INFERENCE**.

It will take ~30 seconds to spin up the first time (image pull is
cached after that, and weights load from the read-only PVC). Watch
the **Pods** tab — the workload is healthy when the pod's readiness
probe passes.

The endpoint is reachable from any workload in the same project at:

```
http://qwen-Qwen2.5--7B--Instruct.runai-<your-project>.svc.cluster.local/v1
```

It speaks the OpenAI Chat Completions API, so any OpenAI-compatible
client works.

## Step B. Reuse the 02 workspace at zero GPU

You don't need a fresh workspace for this — the
`first-workspace` you built in [02](02-first-workspace.md) already has
everything you need (Jupyter, the persistent `/work` volume, the
shared-models mount that we no longer use, the `bash -c` boilerplate).
The only thing that has to change is the GPU.

1. **Stop** `first-workspace` from the RunAI UI.
2. **Edit** it:
   - **Compute resources** > **GPU devices:** change to `0`. Disable
     GPU fractioning. That's the whole point — the model lives in
     the Inference workload now, your notebook just calls it.
   - **Runtime settings** > **Arguments:** add `openai` to the pip
     install list so the next Start has the client library
     available. The full string becomes:
     ```
     -c "curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp; mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null; ln -sf /tmp/RunAI_apps /work/repo; pip install --no-cache-dir transformers accelerate openai; jupyter-lab --ip=0.0.0.0 --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/work"
     ```
     (The `transformers` and `accelerate` installs are now wasted CPU
     since you won't load the model in-process, but leaving them in
     keeps the args identical to 02 minus the `openai` addition,
     which is easier to remember than maintaining two near-duplicate
     arg strings.)
3. **Save** the edits and **Start** the workspace.
4. Once it's `Running`, open Jupyter and **duplicate the notebook
   from 02** (right-click > Duplicate, rename to something like
   `endpoint-client.ipynb`). You'll edit the same three cells in
   place to call the endpoint instead of loading the model.

The workspace boots in seconds this time — no GPU to schedule and no
model weights to load.

## Step C. Call the endpoint from a notebook

Open the duplicated `endpoint-client.ipynb`. You'll replace the three
cells from 02's notebook in place — same prompt, same expected
answer, but the model lives somewhere else now.

### Cell 1 — confirm the endpoint is reachable

```python
import os, urllib.request, json

PROJECT = os.environ["RUNAI_PROJECT"]   # set automatically by RunAI
BASE_URL = f"http://qwen-Qwen2.5--7B--Instruct.runai-{PROJECT}.svc.cluster.local/v1"

with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
    print(json.loads(r.read())["data"][0]["id"])
```

You should see `Qwen/Qwen2.5-7B-Instruct`. If the request hangs or
returns 404, the workload isn't healthy yet — check the **Pods** tab
on `qwen-Qwen2.5--7B--Instruct` and wait for readiness.

### Cell 2 — send a prompt via the OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key="not-used")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"},
    ],
    max_tokens=120,
    temperature=0,
)
print(resp.choices[0].message.content)
```

Same prompt as 02, same answer (modulo sampling), no GPU on this
workspace. The first call after the endpoint scales from zero takes
~10 seconds; subsequent calls are sub-second.

### Cell 3 — concurrency proof

Spin up a small concurrent burst to feel continuous batching at work.
This would block on the 02-pattern workspace.

```python
import asyncio
from openai import AsyncOpenAI

aclient = AsyncOpenAI(base_url=BASE_URL, api_key="not-used")

async def ask(i):
    r = await aclient.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": f"Say the word number {i} and stop."}],
        max_tokens=8,
    )
    return i, r.choices[0].message.content

results = await asyncio.gather(*(ask(i) for i in range(10)))
for i, ans in results:
    print(i, "->", ans)
```

Ten requests hitting one GPU, processed in parallel via vLLM's
continuous batching. On the 02 pattern, those would queue up
single-threaded.

## Step D. Tear down when you're done

When you're done with the notebook, **Stop** `first-workspace` and then
**delete** `qwen-Qwen2.5--7B--Instruct` from **Workloads**. At the
current 2-GPU pilot scale we don't leave personal endpoints running —
even with Min replicas = 0, an idle Inference workload still occupies
project quota and a `pending` pod can block scheduling for whoever
wants the GPU next. The autoscaler is doing its job; the cluster just
doesn't have headroom for many simultaneous catalog entries yet.

Recreating the Inference workload from your saved settings takes a
minute, so the cost of a delete-and-recreate cycle is small. If you
find yourself recreating the same endpoint daily, that's a strong
signal it should become a shared catalog entry — flag it to
Chris/Mike (see the
[shared-endpoints catalog](#where-this-is-heading) up top).

> **Why even configure autoscaling, then?** Because the underlying
> pattern is right; only the scale isn't there yet. The same
> Inference + autoscale-to-zero workload moves cleanly into the
> shared-models project once a model graduates to catalog status,
> and at that point it's fine to leave running because *one* such
> workload per popular model is much cheaper than dozens of
> per-project copies.

## Bridging to the real apps

This is exactly the pattern both production apps use:

- [`rag_app/docs/deploy-vllm.md`](../rag_app/docs/deploy-vllm.md)
  deploys a near-identical vLLM Inference workload (`wattbot-chat`)
  for the WattBot chatbot, plus separate Inference workloads for the
  embedding server and reranker.
- The OCR app's
  [`ocr_app/docs/deploy-vllm.md`](../ocr_app/docs/deploy-vllm.md)
  hosts Qwen3-VL-32B the same way (with `--quantization awq` for the
  larger model), and the OCR notebook's `VLM_MODE = "remote"` (the
  default) calls it via cluster DNS just like Step C above.

When you read those app docs, you'll recognize the structure: pick
image, set args, attach `shared-models`, configure autoscaling, point
your client at `http://<workload>.runai-<project>.svc.cluster.local`.

## Next

Now that you've seen both the direct-load and the
endpoint-and-call-it patterns, the next question is where the *data*
those models read and write actually lives. Head to
[04 Storage](04-storage.md) — Data Source vs Data Volume, the
hands-on PVC → Data Volume promotion, and the recipes for getting
lab data onto the cluster in the first place.
