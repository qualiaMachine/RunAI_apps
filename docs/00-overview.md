# 00 — Overview

> **Step 0** in the [New User Guide](README.md). Read this first to
> decide whether the cluster is the right tool, and to learn the three
> concepts the rest of the guide assumes.

## What the cluster is for

The RunAI cluster (`doit-ai-cluster`) is a small DoIT pilot — two
96 GB GPUs with high-speed interconnect, big enough to host models up
to ~150B parameters. RunAI is the scheduling layer on top; if you've
used a major cloud's notebook + endpoint UI, the experience is
similar (workspaces, fractional GPU allocation, autoscaling
endpoints).

It fills niches that the cloud and CHTC don't serve well:

- **Long-term LLM/VLM apps the institution wants to support.** When a
  service is going to live for years, renting cloud capacity
  permanently doesn't make sense.
- **Long-running training or fine-tuning jobs.** These get expensive
  fast in the cloud.
- **Sensitive / PHI-related data workflows.** Pending the relevant
  cybersecurity reviews — talk to your DoIT contact about current
  status.
- **Hosting a model close to institutional data** — research corpora,
  imaging archives, anything that's awkward or impossible to send to a
  third-party API.
- **Sharing model weights and curated datasets once across many
  users.** A 20–700 GB model download lives in one place; everyone's
  workloads mount it read-only.

It's *not* the right tool when:

- The data has to leave the institution to be useful anyway (e.g.
  you'd genuinely prefer ChatGPT and the data isn't sensitive).
- You want a one-off interactive Python session — your laptop is
  faster to spin up.
- You need persistent custom services (24/7 web apps with their own
  databases, user accounts, etc.). RunAI Inference workloads can do
  this, but it's overkill for non-AI hosting.
- You need cloud-scale concurrency *today*. At the current pilot
  size, two GPUs realistically serve **2–5 concurrent users per app**
  via RunAI's GPU partitioning. Plans to scale up depend on real
  usage from labs like yours, so it's worth flagging your needs early
  rather than waiting for the cluster to grow into them.

## The three concepts you actually need

Every later doc assumes you've internalized these. They map directly to
RunAI UI sections.

### 1. Workload — your running thing

Three flavors:

| Type | What it is | Typical use |
|------|------------|-------------|
| **Workspace** | An interactive pod with a UI (Jupyter, VS Code, custom URL). Persists when you close the browser; you Stop/Start it manually. | Notebooks, terminals, dev work, batch scripts you babysit |
| **Training** | A pod that runs a job to completion and then exits. | Fine-tuning, large batch jobs, scheduled pipelines |
| **Inference** | An autoscaling pod (or pods) behind a network endpoint. | Hosting a model that other workloads call over HTTP |

Almost everything in this repo's tutorials starts with a **Workspace**.

### 2. Data Source — storage you mount into a workload

A pointer to "data lives here, mount it at this path." Types include
**PVC** (cluster-managed disk), **NFS** (network share), **S3**, **Git**,
and a few others. Per-project. You'll create one per dataset, model, or
output location your workload reads or writes.

### 3. Data Volume — a shared, read-only wrapper around a populated PVC

Lets one project's PVC be mounted by **other projects** read-only. The
origin project keeps RW. Used for things like a 20 GB model cache that
every research group on the cluster wants to use without each
re-downloading it.

> **Don't conflate Data Source and Data Volume.** Almost everything you
> mount in your own project is a **Data Source**. Data Volume is the
> "share what I've built across the whole cluster" tier. The
> [Storage doc](04-storage.md) walks through the difference with a
> hands-on example.

## What lives in this repo

This repo is the institution's reference deployments — copy-pasteable
workloads for the most common pilot use cases:

| Path | What it is | When you'd use it |
|------|------------|-------------------|
| [`ocr_app/`](../ocr_app/README.md) | Vision-language document extraction (Qwen3-VL-32B). Turns PDFs/scans into structured JSON. | Grant administration, archival corpora, library digitization, anything where layout matters |
| [`rag_app/`](../rag_app/README.md) | Retrieval-augmented chatbot over a curated corpus (Qwen 7B/14B/72B). | Q&A over institutional knowledge bases, research literature search, "ChatGPT for our docs" |
| [`scripts/`](../scripts) | Shared utilities used by both apps. | You usually don't touch this directly. |

You don't have to use these — they're examples of the *pattern*. If
you're building your own workload, treat the
`*/docs/setup-workspace.md` files as the closest things to a template.

## How to read the rest of this guide

Pick the path that matches what you're doing right now.

**"I just got a login and have no idea what to do."**
→ [01 Access](01-access.md), then [02 First workspace](02-first-workspace.md),
   then [04 Storage](04-storage.md). Skim 03 and 05.

**"I have a corpus I want to extract / chat with and the cluster looks
relevant."**
→ Skim 00–04, then [05 Examples](05-examples.md) to pick between the
   OCR pipeline and the RAG chatbot. Each app's README links back to
   specific sections of 04 when storage decisions come up.

**"I'm a workflow/docs admin onboarding lab PIs onto the cluster."**
→ Read 00–05 in full so you know what to copy/cut/customize.

**"I'm the cluster admin (kubectl, install/upgrade, StorageClass
work)."**
→ This guide isn't for you — see
   [NVIDIA's RunAI docs](https://run-ai-docs.nvidia.com/) for the
   admin/install side. Section 03 lists the four questions a workflow
   admin will likely come ask you once.

## A note on what changes underneath you

RunAI is third-party software (NVIDIA, currently v2.24). UI buttons
get renamed, fields move between tabs, and concepts occasionally shift
between releases (the Data Source vs Data Volume split, for example, is
a v2.24-era feature). This guide tries to teach the *mental model* —
the things that change slowly — and link out to NVIDIA's official docs
for click-by-click steps. When something in this guide doesn't match
what you see on screen, the official docs are the source of truth and
this guide is the bug report.
