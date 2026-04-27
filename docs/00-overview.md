# 00 — Overview

> **Step 0** in the [New User Guide](README.md). Read this first to
> decide whether the cluster is the right tool, and to learn the three
> concepts the rest of the guide assumes.

## What the cluster is for

The RunAI cluster (`doit-ai-cluster`) is a shared GPU environment for
running AI workloads close to institutional data — document corpora,
research datasets, anything that's awkward or impossible to send out to
a third-party API. It's not a substitute for your laptop, and it's not
a personal cloud account. Practically, it's good at:

- **Hosting models that need a GPU.** A researcher who would otherwise
  pay for ChatGPT-style API access can call a self-hosted equivalent
  inside the cluster — with private data, no per-token billing, and
  full control over the model and prompts.
- **Running long jobs without holding open a laptop.** Batch
  extraction over thousands of documents, vector index builds,
  fine-tuning runs.
- **Sharing model weights once across many users.** A 20–700 GB model
  download lives in one place; everyone's workloads mount it
  read-only.
- **Sharing curated datasets across projects.** Once a dataset is
  staged on cluster storage, multiple research groups can read it
  without each holding a copy.

It's *not* the right tool when:

- The data has to leave the institution to be useful (e.g. you need
  ChatGPT specifically and the data isn't sensitive).
- You want a one-off interactive Python session — your laptop is
  faster to spin up.
- You need persistent custom services (24/7 web apps with their own
  databases, user accounts, etc.). RunAI workloads can do this with
  Inference workloads, but it's overkill for non-AI hosting.

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
> [Storage doc](03-storage.md) walks through the difference with a
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
→ [01 Access](README.md) (TBD), then [02 First workspace](02-first-workspace.md),
   then [03 Storage](03-storage.md). Skim the rest.

**"I have a corpus I want to extract / chat with and the cluster looks
relevant."**
→ Skim 00–03, then [04 Examples](04-examples.md) to pick between the
   OCR pipeline and the RAG chatbot. Each app's README links back to
   specific sections of 03 when storage decisions come up.

**"I'm a workflow/docs admin onboarding lab PIs onto the cluster."**
→ Read 00–04 in full so you know what to copy/cut/customize.

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
