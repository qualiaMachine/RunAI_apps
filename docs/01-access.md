# 01 — Access

> **Step 1** in the [New User Guide](README.md). Should take ~one
> conversation and one VPN connection.

## How to get access

1. **Email Chris or Mike at DoIT** about the AI cluster pilot. Tell
   them roughly what you're looking to do (an app to deploy, a model
   to fine-tune, an interactive notebook with GPU, etc.) so they can
   create or assign you to an appropriate project.
2. They'll send you the **portal URL** for the cluster's RunAI web
   UI.
3. **Connect to the campus VPN** before opening the URL — the portal
   isn't reachable from off-campus networks without it.
4. Open the URL in a browser. You'll land on the RunAI dashboard.
   From there you create workspaces, inference workloads, and data
   sources by clicking through the UI — no CLI required.

That's the whole onboarding flow. Subsequent sessions are just
"connect to VPN, open the URL, log in."

## What you'll have once you're in

- A **project** scoped to you (or your lab / team). All workloads,
  PVCs, and data sources you create live in this project's namespace.
- Access to the cluster-wide **`shared-models`** Data Volume, which
  holds pre-cached weights for the models the example apps use (Qwen
  series, Jina V4, Qwen3-VL). You mount this read-only into your
  workloads — see [04 Storage](04-storage.md).
- A GPU quota. Whatever Chris/Mike assigned at onboarding determines
  how many GPUs (or fractions) your workloads can request
  concurrently.

If you need more GPU quota, an additional model on `shared-models`,
or a Data Source pointing at a network share — those are also
"email Chris/Mike" tasks.

## Web UI vs CLI

This guide and the per-app deployment docs assume the **web UI**. The
RunAI CLI exists and works, but it's not yet documented or tested
against this repo's deployment patterns. If you want to use the CLI,
let Chris know — getting it installed and verified end-to-end is in
progress and your use case can help prioritize.

## Next

Head to [02 First workspace](02-first-workspace.md) to spin up a
Jupyter notebook and load a model from the shared-models volume.
