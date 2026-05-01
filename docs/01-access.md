# 01 — Access

> **Step 1** in the [New User Guide](../README.md#new-user-guide). Should take ~one
> conversation and one VPN connection.

## How to get access

1. **Email Chris or Mike at DoIT** about the AI cluster pilot. Tell
   them roughly what you're looking to do (an app to deploy, a model
   to fine-tune, an interactive notebook with GPU, etc.) so they can
   plan the right project assignment.
2. They'll send you the **portal URL** for the cluster's RunAI web
   UI.
3. **Connect to the campus VPN**, then open the URL and log in once.
   The dashboard will look mostly empty — that's expected. The
   first login is what registers your identity in RunAI; the admin
   can't grant you anything until that record exists.
4. **Tell Chris/Mike you've logged in.** They'll attach your project
   assignment, GPU quota, and `shared-models` Data Volume access to
   your account.
5. **Reload the dashboard.** You should now see your project in the
   scope dropdown and be able to create workspaces, inference
   workloads, and data sources by clicking through the UI — no CLI
   required.

After that, subsequent sessions are just "connect to VPN, open the
URL, log in."

## What you'll have once you're in

- A **project** scoped to you (or your lab / team). All workloads,
  PVCs, and data sources you create live in this project's namespace.
- Access to the cluster-wide **`shared-models`** Data Volume, which
  holds pre-cached weights for the models the example apps use (Qwen
  series, Jina V4, Qwen3-VL). You mount this read-only into your
  workloads.
- A GPU quota. Whatever Chris/Mike assigned at onboarding determines
  how many GPUs (or fractions) your workloads can request
  concurrently.

If you need more GPU quota, an additional model on `shared-models`,
or a Data Source pointing at a network share — those are also
"email Chris/Mike" tasks.

## Web UI vs CLI

This guide and the per-app deployment docs assume the **web UI**. The
RunAI CLI exists and works, but deployment patterns in this repo
aren't yet documented or tested against it. If you want to use the
CLI, let Chris know — getting it verified end-to-end against these
apps is in progress and your use case can help prioritize.

If you want a starting point now, the install + login flow is:

1. **Download the binary** from the RunAI UI: *Help → Researcher
   Command Line Interface*, then pick the build for your OS.
2. **Put it on your PATH** and make it executable:
   ```bash
   mv runai /usr/local/bin/runai
   chmod +x /usr/local/bin/runai
   runai version
   ```
3. **Log in.** `runai login` opens a browser; authenticate with your
   UW–Madison NetID, then return to the terminal for the success
   confirmation.
4. **Pin your default project** so you don't have to pass it every
   time: `runai project set <your-project-name>`.
5. **Sanity-check access** with `runai list workloads` — empty output
   is fine if you haven't submitted anything yet.

Beyond that (submitting workloads, mounting `shared-models`,
exposing endpoints), the CLI equivalents of what these guides do via
the UI aren't written up yet.

## Next

Head to [02 First workspace](02-first-workspace.md) to spin up a
Jupyter notebook and load a model from the shared-models volume.
