# 06 — Expose a vLLM Endpoint Outside the Cluster

> **Step 6** in the [New User Guide](../README.md#new-user-guide). Builds on
> [03 Share a Model as a vLLM Endpoint](03-share-as-endpoint.md) — assumes
> you've already stood up the internal-only Qwen2.5-7B endpoint there
> and want to make it reachable from a non-RunAI client.

03 left you with an endpoint addressable only from inside the cluster
(`http://<workload>.runai-<project>.svc.cluster.local`). That's the right
default — most callers in this repo are also RunAI workloads — but it
doesn't help when the caller is a service running somewhere else in
DoIT (Denodo, Tableau, an institutional app server, your laptop). This
doc covers the deltas: enable RunAI's public ingress, smoke-test from
outside the workload's project, then coordinate the firewall rule that
lets a specific external host through.

By the end you'll have:
- A vLLM Inference workload with **External (Public access)** enabled,
  exposing the same OpenAI-compatible API on an `https://` URL
- A curl + `openai` client smoke test passing from your VPN'd laptop
- A clean handoff packet for a cluster admin to open the firewall path
  from a specific source host (e.g. a Denodo server) to the endpoint

## Two flavors of "external"

RunAI's **External (Public access)** toggle and the data-center Palo
Alto firewall are two independent layers — both have to be right for
a non-RunAI caller to reach the endpoint. Which layers you actually
need to touch depends on where the caller sits:

| Situation | RunAI ingress (you) | Palo Alto firewall (DoIT cluster admin, Mike Cammilleri) |
|-----------|---------------------|----------------------------------------------------------|
| **Human caller on VPN** (your laptop, a colleague's browser) | Toggle **External (Public access)** in **Endpoint > Access** to provision an `https://` URL | VPN already permitted to reach the cluster ingress — no ticket needed |
| **Service-to-service caller in the data center** (an institutional app on another VLAN, not on VPN) | Same toggle, same URL | Cross-VLAN traffic blocked by default — ticket adds the caller's source IP → cluster ingress on 443. You can't add this rule yourself. |

The cluster-side steps below are identical for both. The firewall
piece only matters for the second case.

## Step A. Re-deploy with External access

If you already followed 03 and have `qwen-qwen25--7b--instruct`
running with Auth: Internal, the only field that needs to change is
the endpoint access. **Stop** the workload, **Edit**, change Step 8
of the 03 walkthrough:

| Field | 03 (internal) | This doc (external) |
|-------|---------------|---------------------|
| **Container port** | `8000` | `8000` |
| **Auth** | Internal | **External (Public access)** |

Save and **Start** the workload. Everything else from 03 — image,
arguments, env vars, GPU fractioning, autoscaling, the
`shared-models` mount — stays the same.

If you're starting from scratch instead of editing 03's workload,
follow 03 Step A end-to-end and just pick **External (Public
access)** when you get to step 8.

> **Why the same port works.** RunAI fronts the same container port
> (`8000`) with a TLS-terminating ingress when you toggle public
> access. vLLM doesn't have to know about TLS; the ingress handles
> certs and rewrites internally.

## Step B. Find the public URL

After the workload's pod becomes Ready (~30s on a warm image pull),
the public URL appears in the RunAI UI:

1. **Workloads** > `qwen-qwen25--7b--instruct` > the workload's
   detail pane.
2. Look for the **Connections** / **Endpoints** section (UI label
   varies by RunAI version) — it lists an `https://...` URL.
3. Copy that URL. The UI shows the **base** only; the OpenAI-compatible
   API lives under `/v1`, so the full base URL you'll hand to clients
   is the UI URL with `/v1` appended.

RunAI exposes Inference workloads via Knative-style FQDNs on this
cluster — one subdomain per workload, derived from the workload name
and the project namespace:

```
https://<workload-name>-runai-<project>.<cluster-host>
```

**Worked example.** For workload `qwen-qwen25--7b--instruct` in
project `jupyter-endemann01` on the `deepthought.doit.wisc.edu`
cluster, the UI shows:

```
https://qwen-qwen25--7b--instruct-runai-jupyter-endemann01.deepthought.doit.wisc.edu
```

…and the full base URL for the OpenAI API is therefore:

```
https://qwen-qwen25--7b--instruct-runai-jupyter-endemann01.deepthought.doit.wisc.edu/v1
```

That's the URL you'll paste into the smoke tests below and into the
firewall request in Step D. Save it somewhere — you'll need it three
more times.

> **If the URL the UI shows is `…svc.cluster.local…`**, the
> workload is still on Auth: Internal — re-check Step A.

## Step C. Smoke-test from your VPN'd laptop

Connect to the campus VPN, then open a terminal locally (not a RunAI
workspace — the point is to prove the endpoint is reachable from
outside the cluster).

```bash
# Replace the project segment with your own (e.g. jupyter-yourname)
export VLLM_URL="https://qwen-qwen25--7b--instruct-runai-jupyter-endemann01.deepthought.doit.wisc.edu/v1"

# 1. Endpoint is alive and serving the expected model — should print: Qwen/Qwen2.5-7B-Instruct
curl -s "$VLLM_URL/models" \
  | python -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])"

# 2. Chat completion works — should print: OK
curl -s "$VLLM_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Reply with just the word OK."}],
    "max_tokens": 8,
    "temperature": 0
  }' \
  | python -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Both commands extract just the field that matters with a one-line
Python parse, so a healthy run prints two short strings and nothing
else. To see the full JSON instead — useful when something looks
off — drop the `python -c` pipe and use `curl -i` so you can also
inspect status code and headers.

If you'd rather use the OpenAI Python client (this is what Denodo and
most other consumers will do under the hood):

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://qwen-qwen25--7b--instruct-runai-jupyter-endemann01.deepthought.doit.wisc.edu/v1",
    api_key="not-used",
)
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Reply with just the word OK."}],
    max_tokens=8,
    temperature=0,
)
print(resp.choices[0].message.content)  # -> OK
```

If both pass, the cluster-side work is done. The endpoint will accept
calls from any client the firewall lets through.

### Common smoke-test failures

| Symptom | Likely cause |
|---------|--------------|
| `curl: (6) Could not resolve host` | Not on VPN, or DNS not refreshed after VPN connect. Reconnect the VPN client. |
| `curl: (7) Failed to connect ... 443` | DNS resolved but VPN policy doesn't permit your laptop to reach the cluster ingress. Contact Mike. |
| HTTP 200 with an HTML body (RunAI UI's `<!DOCTYPE html>` shell, often ~2 KB, `Content-Type: text/html`) | URL is wrong — you've hit the cluster's UI hostname instead of the workload's Knative subdomain. Re-copy the URL from the workload's Connections panel; the right form is `https://<workload>-runai-<project>.<cluster-host>`. |
| HTTP 503 / "no healthy upstream" | Workload pod hasn't passed readiness yet, or it's BackOff-restarting. Watch the **Pods** tab on the workload; if it's restart-looping, open the `user-container` logs to see why vLLM is exiting. |
| HTTP 404 on `/v1/models` | You copied the UI URL but didn't append `/v1`. The OpenAI API is at `<base>/v1`. |
| HTTP 200 on `/models` but `/chat/completions` returns model-not-found | The `model` field in the request body must exactly match the HF model ID vLLM was launched with — check the args in Step A and the `data[0].id` from `/models`. |

## Step D. Network access for a non-VPN caller (cluster admin)

> **You can skip this section if your caller is already on VPN or
> already has a firewall path to the cluster ingress.** It only
> applies when a new external service (e.g. a Denodo server on a
> separate VLAN) needs to be allowed through.

Send the cluster admin (Mike) a short request with these four pieces
of information so they can add a Palo Alto rule without a back-and-forth:

| Item | Where you get it | Example |
|------|------------------|---------|
| **Source host(s)** | Ask the consuming team for the IP or subnet of the server that will call the endpoint. | `10.x.y.z` or `10.x.y.0/24` |
| **Destination URL** | The URL from Step B. | `https://qwen-qwen25--7b--instruct-runai-jupyter-endemann01.deepthought.doit.wisc.edu/v1` |
| **Destination port** | Always 443 for the public URL. | `443` |
| **Why** | One-line reason so the rule is auditable. | "Denodo PoC — DAPIR LLM endpoint, public data only" |

The admin's side of the work is a single Palo Alto rule allowing the
source host(s) to the cluster ingress VIP on 443. Once it's in,
re-run the Step C smoke tests *from the source host* (not from your
laptop) to confirm.

> **TODO — cluster admin section.** The exact Palo Alto rule
> shape, how the destination VIP is chosen for our RunAI ingress,
> and any DNS/route considerations for cross-VLAN callers are
> admin-side details that aren't captured here yet. After the next
> live setup with Mike, fill this section in so future labs can
> self-serve the request.

## Step E. Hand-off packet

For the team that's going to plug this into their tool, send:

1. The base URL ending in `/v1`
2. The model ID (must match exactly in their `model` field) —
   `Qwen/Qwen2.5-7B-Instruct` for this walkthrough
3. A working `curl` from Step C, with the URL pre-filled, so they
   have a known-good reference to compare against if their client
   misbehaves
4. A note that the API is OpenAI-compatible — most tools that accept
   an "OpenAI base URL + API key" config field work as-is (use any
   non-empty placeholder for the key; vLLM doesn't validate it
   unless you've configured `--api-key`)

If they need authentication beyond network-level firewall (i.e.
secret-based access on top of an allowed source IP), that's a
separate vLLM `--api-key` flag in the workload arguments — not
covered here. For an internal PoC with a firewall-restricted source,
network-level controls are usually sufficient.

## When to tear down

Same guidance as 03 Step D: at the current 2-GPU pilot scale we
don't leave per-project endpoints running. Once the consuming
service has finished its tests:

1. Coordinate with the consumer that you're stopping the endpoint.
2. **Stop** then **Delete** the Inference workload.
3. Leave the firewall rule in place — it's harmless when the
   destination isn't serving, and it'll save a re-request next
   time.

If the endpoint is going to live on past the PoC, that's the signal
it should graduate to a shared-models catalog entry (see 03's
[Where this is heading](03-share-as-endpoint.md#where-this-is-heading))
rather than living in your project.

## Next

If your consuming service is going to need data colocated with the
endpoint (uploads, RAG corpora, intermediate caches), revisit
[04 Storage](04-storage.md) for the Data Source vs Data Volume
patterns and pick the right tier for it.
