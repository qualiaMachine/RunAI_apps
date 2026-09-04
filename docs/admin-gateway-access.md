# Handing Out Model Access — Admin Runbook

> **Admin doc, not part of the [New User Guide](../README.md#new-user-guide).**
> Everything here assumes gateway admin access and cluster-admin
> contacts. Participants don't need to read it — what they get is the
> two lines of config in [Step 4](#step-4--what-participants-get-hand-off).

How to give people access to the shared models without giving everyone a
Run:ai account. Written for the ML Marathon, but it is the same three
steps for anyone: **a Team, and a key per person inside it.**

"Team" is just the grouping label — whatever you'd want a usage line
item for:

| Real-world group | Team name |
|------------------|-----------|
| A hackathon team | `marathon-team-07` |
| A lab or department | `wams` |
| A course section | `stat479-fa26` |
| One researcher with no group | `wams` — put them in their unit's team |

A team of one is fine and is the right call for a lone researcher: when
the second person from that unit shows up, they're one more key rather
than a restructure. Don't create a team per person.

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
- **1Password** — where the master key lives and where the virtual keys
  you mint should end up. The desktop app is enough; the `op` CLI is
  optional and only helps if you have a service account (see below).
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

Create a LiteLLM **Team** per group, then mint a key per person *within*
that team. This gives you:

- **Team-level** budgets and rate limits — the fairness unit
- **Per-user** rows in the Usage dashboard — the attribution unit
- Surgical revocation: kill one key without breaking their teammates

If registration is walk-up and you don't have a roster in advance, fall
back to one key per team. A shared key ends up pasted in the team's repo
regardless; better to plan for it than to pretend otherwise.

**Organizations exist in LiteLLM as a tier above Team — ignore them.**
Teams already carry the budgets, the rate limits, and the usage
breakdown. An org layer only earns its keep if you need one budget
spanning several teams, or want to delegate key-minting to a unit's own
admin (that delegation may require Enterprise — verify before planning
around it). Neither applies at pilot scale.

**Attribution is independent of model scoping.** Every request is
recorded against its key, user, team, *and* model regardless of what a
key is permitted to call — so granting everyone the whole catalog (as
below) costs nothing in tracking. The layers do different jobs:

| Layer | Answers | Where |
|-------|---------|-------|
| Organization | A tier above Team — **not used here** | Dashboard |
| **Team** | Which group — hackathon team, lab, department, course — and its budget and rate limits | Dashboard |
| Internal User | Which person, across all their keys | Dashboard |
| Key | Which credential, with its own alias and limits | Dashboard |
| Access group | Which models a key may call — **not used here** | GitLab |

## Where each step happens

| Task | Surface | Effect |
|------|---------|--------|
| Create teams, mint/revoke keys, set budgets and rate limits | **`scripts/provision_gateway_keys.py`** (the proxy's API); dashboard for one-offs | Instant |
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
GitLab MR in this runbook** — the setup below is one script run.

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

## Steps 1–3 — Run the script

> **None of this touches GitLab.** The `se-litellm` pipeline builds and
> deploys the proxy — image, config, masked secrets, TLS, Postgres. Teams,
> users and keys are *rows in that Postgres*, created at runtime through
> the proxy's API. There is no file to commit and no pipeline to run, which
> is also why none of it can be blocked by campus git being down on the
> morning of an event. GitLab is only involved when a **model** changes.
> (`se-litellm`'s own README says the same under *Managing access*.)

Everything up to hand-off is one command. `scripts/provision_gateway_keys.py`
creates any teams that don't exist and mints one key per person, writing
them to `keys.csv`. Nothing is clicked in the dashboard. With a 1Password
service account you can add `--use-op` and it files each key and emits
share links too; without one, that part is a minute in the 1Password app.

**Write the roster.** One row per person. Start from the committed
template — [`scripts/roster.example.csv`](../scripts/roster.example.csv),
which `--example` also prints:

```bash
python scripts/provision_gateway_keys.py --example > roster.csv
# edit roster.csv: replace the sample rows with your people
```

`roster.csv`, `keys.csv`, `share-links.csv` and `file_in_1password.*` are
all gitignored — some hold live credentials, the rest list real people.

```
netid,team,email,rpm_limit,duration
bbadger,wams,bbadger@wisc.edu,,
astudent,marathon-team-07,astudent@wisc.edu,,7d
```

| Column | Notes |
|--------|-------|
| `netid` | Becomes the key alias (`<team>-<netid>`), the `user_id`, and the 1Password item title |
| `team` | Any grouping label — created automatically if it doesn't exist yet |
| `email` | Who the share link is locked to |
| `rpm_limit` | **Leave blank** — see [Guardrails](#guardrails-dashboard). Fill in only to cap a key you have a specific reason to distrust |
| `duration` | Blank for no expiry. Set it (`7d`) for anything event-shaped so cleanup is automatic; **leave blank for standing users** or you break a pipeline mid-project |

**Set the master key**, then run the two commands:

```powershell
$env:LITELLM_MASTER_KEY = "sk-..."     # bash: export LITELLM_MASTER_KEY=sk-...

python scripts\provision_gateway_keys.py roster.csv            # dry run: plan only
python scripts\provision_gateway_keys.py roster.csv --apply    # mint the keys
.\file_in_1password.ps1                                        # file + share them
```

The first command creates any missing teams and mints the keys, then
writes `file_in_1password.ps1`. The second files each key in 1Password as
an API Credential, shares it with its owner, and deletes itself. No manual
clicking either side.

> **Why it's two commands.** 1Password's desktop integration authorizes by
> *calling application*: a terminal you typed into is trusted, `python.exe`
> is not. So `op` works when you run it and fails from inside a script —
> confirmed with a bare `python -c` one-liner, and no shell, subprocess or
> output-capture trick changes it. Emitting the commands for your shell to
> run sidesteps it, because then your terminal is `op`'s parent.
>
> Run the emitted script **from a terminal**, not from an editor or IDE
> task runner — those hit the same restriction.
>
> A **service account token** (`OP_SERVICE_ACCOUNT_TOKEN`) is the only
> thing that lifts the restriction. With one, pass `--use-op` and the
> Python script does the 1Password half directly, in one command. Worth
> setting up if this ever needs to run unattended.

The dry run is the default and prints exactly which teams and keys it
would create. `--apply` is the only thing that writes.

Useful flags: `--vault` (default `DoIT-AI`), `--gateway`,
`--expires-in` (share-link lifetime, default `14d`), `--op-script`, and
`--master-key-ref` (only with `--use-op`).

**Re-running is safe.** Anyone who already has a key is skipped — matched
on the gateway's key aliases, or on the vault when running `--use-op` — so
adding rows and re-running onboards only the new people. If that check
can't run, the script says so rather than quietly minting duplicates.

`file_in_1password.ps1` holds live credentials until you run it, at which
point it deletes itself. If you abandon a run partway, delete it by hand.

## Step 4 — What participants get (hand-off)

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

To see what models they can call:

```bash
curl -s https://llm-gw01.doit.wisc.edu/v1/models \
  -H "Authorization: Bearer sk-<their-key>"
```

Participants can check their own usage without a gateway login:

```bash
curl -s https://llm-gw01.doit.wisc.edu/key/info \
  -H "Authorization: Bearer sk-<their-key>"
```

### Four things to say when you send the key

Worth stating explicitly — each one is a support ticket otherwise:

1. **You must be on GlobalProtect (campus VPN)** to reach the gateway,
   from anywhere including on-campus wifi.
2. **Keep the key out of your repo.** Read it from an environment
   variable (`OPENAI_API_KEY` works with the `openai` client
   unmodified), not a literal in a notebook that gets pushed to GitHub.
3. **Always call the gateway URL**, not a
   `*.deepthought.doit.wisc.edu` model hostname someone shared. The
   direct hostnames answer without a key, so calls that bypass the
   gateway are invisible in usage reporting — and unattributed traffic
   is what gets a pilot's quota questioned.
4. **The first call after an idle period can take a couple of minutes**
   on models configured to scale to zero, because a GPU replica is
   starting. Set a generous client timeout rather than treating it as a
   failure — see
   [What scale-to-zero feels like to a caller](#what-scale-to-zero-feels-like-to-a-caller).

## Guardrails (dashboard)

**Use rate limits, not budgets, as the enforcement mechanism.**
`rpm_limit` and `tpm_limit` work regardless of model pricing.
`max_budget` only bites if the model has a working cost entry in
`model_info` — and per-image pricing for the image model is an
[open item](#known-limitations), so budgets on that model currently
enforce nothing.

**Start with no limits at all.**

| Limit | Value | Why |
|-------|-------|-----|
| `rpm_limit` | *(unset)* | Add when the dashboard shows a real problem, not before |
| `tpm_limit` | *(unset)* | The better lever than rpm when it comes to that — see below |
| `max_budget` | *(unset)* | Meaningless until pricing is verified |

Leaving them unset is deliberate, not laziness:

- **Attribution doesn't depend on them.** Every request is recorded
  against its key, user, team and model whether or not a limit exists,
  so usage tracking — the actual point of the pilot — is unaffected.
- **Limits are a database row**, addable in seconds with no MR, no
  pipeline, no redeploy. There is no reason to guess a number now when
  you can set an informed one later.
- **vLLM already provides backpressure.** Oversubscription degrades to
  latency, not collapse; the queue is a better throttle than a number
  picked without measuring.
- **A needless 429 reads as an outage.** Clients that don't retry with
  backoff just fail, and you'd spend an event debugging your own rate
  limit rather than the service.

What you're accepting: a runaway loop — an agent retry storm, a
`while True` in a notebook — can saturate the queue for everyone with
nothing stopping it automatically. The blast radius is degraded latency,
not lost data or spend, and revoking the key takes seconds once the Logs
page shows it. Fine while someone is watching; reconsider for a key
running unattended over a long weekend.

When you do add a limit, two things to keep in mind:

- **`rpm` is the wrong unit for token-heavy work.** One request with a
  100k-token context costs far more GPU than a hundred short chats, and
  rpm treats them identically. For document extraction or batch
  summarization, `tpm_limit` is the honest control.
- **Per-key limits stop individuals; team limits protect the pilot.**
  Forty people at 60 rpm each is well past what two GPUs serve, so
  per-key ceilings never protected capacity — the team limit is the
  fairness lever.

For real numbers rather than reasoning, run the load test below.

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

- Revoke keys in bulk: Virtual Keys → filter by team → delete. Keys
  minted with `duration` expire on their own, so this is a sweep for
  anything issued without one.
- Archive or delete the event's 1Password vault once keys are revoked,
  and delete `keys.csv` / `share-links.csv` and the roster. Anything
  left holding live credentials is the thing that outlives the event.
- Export the Usage dashboard first if the numbers feed a writeup —
  deleted keys drop out of the default view.
