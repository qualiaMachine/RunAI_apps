# Handing Out Model Access — Admin Runbook

> **Admin doc, not part of the [New User Guide](../README.md#new-user-guide).**
> Everything here assumes gateway admin access and cluster-admin
> contacts. Participants don't need to read it — what they get is the
> two lines of config in [Step 3](#step-3--what-participants-get-hand-off).

How to give a cohort access to the shared models without giving
everyone a Run:ai account. Written for the ML Marathon, but the setup is
the general pattern for any cohort — a class, a lab, a pilot group.

## Two tiers of access

| Tier | Gets | Onboarding | Use when |
|------|------|-----------|----------|
| **Endpoint-only** | A LiteLLM virtual key + the gateway base URL | Mint a key, send two lines of config. No Run:ai account. | Someone wants to *call* models — most hackathon participants, app builders, notebook users |
| **BYO-M** | A scoped Run:ai project (AI Practitioner role, project scope) | Access rule + project + quota, via cluster admin | Someone wants to *run* their own model or fine-tune — see [02 First workspace](02-first-workspace.md) and [07 CLI submission](07-cli-submission.md) |

The two systems are independent: LiteLLM knows nothing about Run:ai
accounts, and Run:ai roles don't gate who can call a gateway endpoint.
"No Run:ai access" is therefore the *entire* configuration for tier 1 —
there is nothing to switch off.

> **Be precise about what this separates.** Knative serving hostnames
> (`<workload>-runai-<project>.deepthought.doit.wisc.edu`) are reachable
> from campus/VPN **without a key**. A tier-1 user who learns a hostname
> can bypass the gateway entirely. That makes gateway keys *accounting
> and quota*, not access control, until serving workloads require their
> own auth (vLLM's `--api-key`). Say it that way in any usage policy.

## Prerequisites

- **Gateway dashboard admin** — `https://llm-gw01.doit.wisc.edu`, log in
  with the master key (`LITELLM_MASTER_KEY`)
- **Maintainer on the `se-litellm` GitLab repo** — *only* if you're
  adding a model to the catalog for the event. Cohort setup itself needs
  no git access. Note Developer can commit but can't manage CI/CD
  variables, which a new backend endpoint requires.
- **Run:ai access to `shared-models`** — only if you're also changing
  autoscaling or asking for quota.
- Participants must be on GlobalProtect (campus VPN) to reach the
  gateway.
- The models participants need should be in the catalog **days ahead**.
  That's the only part of this that needs a pipeline deploy; everything
  else is instant.

## Design: teams as the unit, per-user keys inside them

Create a LiteLLM **Team** per hackathon team, then mint a key per
participant *within* that team. This gives you:

- **Team-level** budgets and rate limits — the fairness unit
- **Per-user** rows in the Usage dashboard — the attribution unit
- Surgical revocation: kill one key without breaking their teammates

If registration is walk-up and you don't have a roster in advance, fall
back to one key per team. A shared key ends up pasted in the team's repo
regardless; better to plan for it than to pretend otherwise.

**Attribution is independent of model scoping.** Every request is
recorded against its key, user, team, *and* model regardless of what a
key is permitted to call — so granting everyone the whole catalog (as
below) costs nothing in tracking. The layers do different jobs:

| Layer | Answers | Where |
|-------|---------|-------|
| Organization | Which department/program | Dashboard |
| **Team** | Which hackathon team, lab, or course — and its budget and rate limits | Dashboard |
| Internal User | Which person, across all their keys | Dashboard |
| Key | Which credential, with its own alias and limits | Dashboard |
| Access group | Which models a key may call — **not used here** | GitLab |

## Where each step happens

| Task | Surface | Effect |
|------|---------|--------|
| Create teams, mint/revoke keys, set budgets and rate limits | **LiteLLM dashboard** (or its API) | Instant |
| Add a model to the catalog, set its pricing | **GitLab** MR on `config/litellm_config.yaml` | After pipeline deploy |
| Autoscaling replicas, GPU quota | **Run:ai** | On redeploy / admin action |

The proxy runs `store_model_in_db: false`, so anything describing a
*model* is git-managed and anything describing a *person* is
database-managed. **Cohort setup is entirely the second kind** — no MR,
no pipeline, nothing that can be blocked by campus git being flaky on
the morning of an event.

## Model scoping — not used

Everyone with a key gets the whole catalog. Keys are created without a
model restriction, so there are **no access groups to define and no
GitLab MR in this runbook** — the entire setup below is dashboard work.

Two consequences to keep in mind:

- **A model added to the catalog is immediately available to every
  existing key.** Fine while the catalog is a handful of models we own;
  revisit if BYO-M endpoints start registering, or if a model arrives
  that shouldn't be a default (a second image model, anything with
  licensing constraints).
- **That includes the image model**, so every key holder can generate
  images. Acceptable under the [usage policy](usage-policy.md) for a
  campus pilot; see the content note in
  [`image_app/README.md`](../image_app/README.md).

If that changes, scoping is one `access_groups:` line per model in
`config/litellm_config.yaml` (a GitLab MR), referenced by name when
minting keys.

## Step 1 — Create the teams (dashboard)

**Teams → + Create Team**, one per hackathon team. Set:

| Field | Value |
|-------|-------|
| Team name | e.g. `marathon-team-07` |
| Models | leave unrestricted (whole catalog) |
| Max budget / TPM / RPM | see [Guardrails](#guardrails-dashboard) below |

## Step 2 — Mint the keys (dashboard or API)

**UI:** Virtual Keys → + Create New Key → assign the team, alias it with
the participant's NetID, and let it inherit the team's model access.

**Scripted**, from a roster (one `netid,team_id` per line):

```bash
export LITELLM_MASTER_KEY=sk-...       # do not paste into shared terminals
while IFS=, read -r netid team_id; do
  curl -s https://llm-gw01.doit.wisc.edu/key/generate \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"key_alias\":\"marathon-$netid\",
         \"team_id\":\"$team_id\",
         \"models\":[\"marathon-models\"],
         \"rpm_limit\":60,
         \"metadata\":{\"event\":\"ml-marathon\",\"netid\":\"$netid\"}}" \
    | python -c "import sys,json; d=json.load(sys.stdin); print('$netid', d['key'])"
done < roster.csv > keys.txt
```

`keys.txt` is then your distribution list. It contains live credentials —
treat it accordingly and delete it after the event.

## Step 3 — What participants get (hand-off)

Two lines of config and a snippet:

- **Base URL:** `https://llm-gw01.doit.wisc.edu/v1`
- **API key:** their `sk-…`

```python
from openai import OpenAI
client = OpenAI(base_url="https://llm-gw01.doit.wisc.edu/v1", api_key="sk-...")

resp = client.chat.completions.create(
    model="qwen3-27b",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

Image generation uses the same client with `client.images.generate(...)`
against the image model's public name — see
[`image_app/README.md`](../image_app/README.md).

Participants can check their own usage without a gateway login:

```bash
curl -s https://llm-gw01.doit.wisc.edu/key/info \
  -H "Authorization: Bearer sk-<their-key>"
```

## Guardrails (dashboard)

**Use rate limits, not budgets, as the enforcement mechanism.**
`rpm_limit` and `tpm_limit` work regardless of model pricing.
`max_budget` only bites if the model has a working cost entry in
`model_info` — and per-image pricing for the image model is an
[open item](#known-limitations), so budgets on that model currently
enforce nothing.

Suggested starting points for a hackathon, per key:

| Limit | Value | Why |
|-------|-------|-----|
| `rpm_limit` | 60 | Generous for interactive use, stops a runaway loop |
| `tpm_limit` | *(unset)* | Add only if a team is starving others |
| `max_budget` | *(unset)* | Meaningless until pricing is verified |

If you're also load-testing (below), set these high or leave them off
for the teams generating load — otherwise the gateway throttles the
traffic before it reaches the endpoint and you measure LiteLLM, not
Knative.

## Monitoring usage (dashboard + Grafana)

- **Usage** page — spend and token counts broken down by key, team,
  user, and model. Token and request counts are tracked independently
  of pricing, so attribution works even where cost shows `$0.00`.
- **Logs** page — per-request rows: model, key, duration, status.
  Prompt/response bodies are **not** stored (`turn_off_message_logging:
  true` in the proxy config), so this is metadata only.
- **Grafana** — all three gateway containers ship logs to Loki
  (`service=litellm|nginx|postgres`), which is where request-rate over
  time is easiest to plot.

## Load and autoscaling measurement (Run:ai)

Inference workloads default to `min_replicas=1, max_replicas=1` — no
autoscaling at all. Scaling has to be requested at submit time, and
quota has to allow the replica count: the `shared-models` project has
**2.00 GPUs**, so a 1.00-GPU model reaches exactly two replicas with
nothing else running. Small models are what produce a legible curve.

### Working submit commands

Verified on this cluster, Sept 2026. Run from Git Bash; the
`MSYS_NO_PATHCONV=1` prefix stops Windows from rewriting `/models` into
a `C:\...` path (see [07 CLI submission](07-cli-submission.md)).

**Embedder — scale-to-zero, up to 4 replicas:**

```bash
./runai-cli-amd64.exe inference delete qwen3-vl-embedding-8b
sleep 20
MSYS_NO_PATHCONV=1 ./runai-cli-amd64.exe inference submit qwen3-vl-embedding-8b \
  -i vllm/vllm-openai:latest \
  --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.25 \
  --existing-pvc=claimname=shared-model-repository-project-3w4iu,path=/models \
  --serving-port=container=8000,protocol=http \
  --min-replicas 0 --max-replicas 4 \
  --metric concurrency --metric-threshold 16 \
  --scale-to-zero-retention-seconds 300 \
  -e HF_HOME=/models/.cache/huggingface -e HF_HUB_CACHE=/models/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -- Qwen/Qwen3-VL-Embedding-8B --runner pooling --max-model-len 8192
```

**CHURRO-3B — same shape:**

```bash
./runai-cli-amd64.exe inference delete churro-3b
sleep 20
MSYS_NO_PATHCONV=1 ./runai-cli-amd64.exe inference submit churro-3b \
  -i vllm/vllm-openai:latest \
  --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.20 \
  --existing-pvc=claimname=shared-model-repository-project-3w4iu,path=/models \
  --serving-port=container=8000,protocol=http \
  --min-replicas 0 --max-replicas 3 \
  --metric concurrency --metric-threshold 16 \
  --scale-to-zero-retention-seconds 300 \
  -e HF_HOME=/models/.cache/huggingface -e HF_HUB_CACHE=/models/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -- stanford-oval/churro-3B --max-model-len 16384
```

**TrOCR-Kurrent — custom server, not a vLLM args-only submit:**

This one runs `ocr_app/scripts/trocr_server.py` on the stock vLLM image,
so the whole startup is a `bash -c` string: pull the repo tarball,
install the transformers 4.x shim to `/tmp/deps`, run the server. That
makes cold start longer than the others, hence
`--initialization-timeout-seconds 1800`. Note there is **no `|| sleep
3600`** — that trap is for debugging a crash-loop and must come off
before the workload goes into service, or a dead server sits there
looking healthy for an hour.

```bash
./runai-cli-amd64.exe inference delete trocr-kurrent
sleep 20
MSYS_NO_PATHCONV=1 ./runai-cli-amd64.exe inference submit trocr-kurrent \
  -i vllm/vllm-openai:latest --image-pull-policy IfNotPresent \
  --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.15 \
  --existing-pvc=claimname=shared-model-repository-project-3w4iu,path=/models \
  --serving-port=container=8000,protocol=http \
  --min-replicas 0 --max-replicas 3 \
  --metric concurrency --metric-threshold 16 \
  --scale-to-zero-retention-seconds 300 \
  --initialization-timeout-seconds 1800 \
  -c -- bash -c 'curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && pip install --no-cache-dir --target /tmp/deps "transformers>=4.42,<5" sentencepiece protobuf && PYTHONPATH=/tmp/deps python3 /tmp/RunAI_apps-main/ocr_app/scripts/trocr_server.py'
```

### Four things that cost an hour to learn

- **Do not pass `--cpu-core-request` / `--cpu-memory-request`.** They
  were accepted in August and are now rejected at admission — the
  workload fails in ~5 seconds with a Knative revision that never
  schedules a pod and no useful error message. Omit them and cluster
  defaults apply. **Nick's `qwen38-27b-vllm` still carries
  `--cpu-memory-request 64G`; it survives only because it was admitted
  in August and will fail this way if it is ever redeployed.**
- **Size the GPU fraction against weights + KV + encoder cache, not
  weights alone.** vLLM ≥0.28 profiles CUDA-graph memory, which shrank
  the usable budget: the embedder failed at 0.25→0.20 with
  `Available KV cache memory: -0.06 GiB` even though its weights are
  only 15.5 GiB. Get the weights figure from
  `provision_shared_models.py vram <model>` and leave several GiB of
  headroom.
- **`min-replicas 0` still starts one replica immediately.**
  `--initial-replicas` defaults to 1 when min is 0; it idles down after
  the retention window plus Knative's stabilization period (~5–8 min).
- **Which models get `min 0`.** Occasional/benchmark models
  (`churro-3b`, `trocr-kurrent`, `hidream-image-app`) — cold start is
  ~90 s, which is fine when someone is deliberately invoking them.
  Anything on a user's critical path stays at `min 1`: the chat model
  behind the gateway, and the embedder **once the marathon starts**,
  since a RAG pipeline calls it inline.

### What scale-to-zero feels like to a caller

Scaling to zero does **not** take the endpoint down. The Knative route
stays live and points at the activator, which holds an incoming request
open, triggers the scale-up, and forwards it once the pod is ready. No
connection refused, no dropped request — the first caller after an idle
period just waits out the cold start.

Two timeouts can still spoil that first request:

- **The client's own timeout.** Cold start is ~90 s for the vLLM
  args-only models and longer for `trocr-kurrent`, which pip-installs
  before it loads. A client with a 30 s or 60 s timeout gives up
  mid-wait. Anything calling a `min 0` endpoint needs a generous
  timeout — the load-test script below uses 300 s for exactly this
  reason — and that includes LiteLLM's upstream timeout when the call
  arrives through the gateway.
- **The revision request timeout.** The activator will not hold a
  request indefinitely. If cold start runs past it the caller gets a
  504 while the pod comes up fine, and the *next* caller — now hitting
  a warm replica — succeeds. That produces the confusing signature of
  intermittent 504s that appear to fix themselves; check whether the
  workload was scaled to zero before chasing it as a bug.

So a `min 0` endpoint is always reachable, but only usable by a caller
willing to wait. That is the trade being made when a model is put at
`min 0`, and it is why anything on a user's critical path stays at
`min 1`.

### Generating load

```python
# uv run --with httpx loadtest.py
import asyncio, httpx, time

URL = "https://qwen3-vl-embedding-8b-runai-shared-models.deepthought.doit.wisc.edu/v1/embeddings"
PAYLOAD = {"model": "Qwen/Qwen3-VL-Embedding-8B",
           "input": "the quick brown fox jumps over the lazy dog " * 20}
STEPS, SECONDS, COOLDOWN = [5, 15, 30, 60], 120, 20

async def worker(c, stop_at, lat, err):
    while time.time() < stop_at:
        t0 = time.perf_counter()
        try:
            (await c.post(URL, json=PAYLOAD)).raise_for_status()
            lat.append(time.perf_counter() - t0)
        except Exception as e:
            err.append(repr(e))

async def step(conc, secs):
    lat, err = [], []
    stop_at = time.time() + secs
    async with httpx.AsyncClient(timeout=300) as c:   # 300s: first call waits for cold start
        await asyncio.gather(*(worker(c, stop_at, lat, err) for _ in range(conc)))
    if lat:
        s = sorted(lat); n = len(s)
        print(f"conc={conc:3d} n={n:5d} rps={n/secs:6.1f} "
              f"p50={s[n//2]*1000:7.0f}ms p95={s[int(n*.95)-1]*1000:7.0f}ms err={len(err)}", flush=True)

async def main():
    for c in STEPS:
        await step(c, SECONDS); await asyncio.sleep(COOLDOWN)

asyncio.run(main())
```

Watch replicas in a second terminal — Run:ai is authoritative for what
the scaler actually did, not LiteLLM:

```bash
while true; do date +%H:%M:%S; ./runai-cli-amd64.exe inference list | grep <workload>; sleep 15; done
```

Concurrency threshold 16 at Knative's default 70% means a replica is
added at ~11 in-flight requests, so the ramp above should walk 1 → 2 →
3 → cap. Expect a **p95 spike right after each scale-up** while the new
replica loads weights and compiles (~90 s), and treat the first step's
latency as the scale-from-zero cost — that number is what decides
whether `min 0` is acceptable for a given model.

## Known limitations

- **No SSO** (enterprise feature). Participants can't log into the
  gateway UI, so key distribution is manual and usage questions route
  through an admin — mitigated by the `/key/info` snippet above.
- **No audit logs, no fine-grained RBAC** (both enterprise). You can't
  give a team lead scoped admin.
- **Direct endpoint access bypasses the gateway** — see the note at the
  top. Gateway totals are a floor, not the true usage number.
- **Image-model spend reports `$0.00`.** The per-image cost key in
  `model_info` hasn't been verified against this LiteLLM version's
  pricing schema. Token-priced chat models are unaffected. Verify by
  generating one image and checking the cost field on its row in the
  Logs page before relying on any dollar figure.

## After the event

- Revoke keys in bulk: Virtual Keys → filter by team → delete, or
  `POST /key/delete` with the key list from `keys.txt`.
- Delete `keys.txt` and any copies.
- Export the Usage dashboard first if the numbers feed a writeup —
  deleted keys drop out of the default view.
