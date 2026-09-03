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

If the point of increased traffic is to measure Knative autoscaling,
two things need to be true and currently aren't:

1. **The workload has to be allowed to scale.** Inference workloads
   default to `min_replicas=1, max_replicas=1` — no autoscaling at all.
   Set max replicas and an autoscaling metric at submit time
   (`--min-replicas 1 --max-replicas N --metric concurrency
   --metric-threshold X`, or the Replica autoscaling fields in the UI).
2. **Quota has to allow N replicas.** The `shared-models` project has
   **2.00 GPUs**. A 1.00-GPU model can reach exactly two replicas with
   nothing else running. For a meaningful scaling curve either raise
   department quota or pick a small model — the 8B embedder at 0.25
   scales to 4–6 replicas within current quota.

Replica counts and GPU utilization come from Run:ai, not LiteLLM;
that's the authoritative source for what the scaler actually did.

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
