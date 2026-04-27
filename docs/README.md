# New User Guide

A progressive walkthrough of the RunAI cluster for researchers and lab
admins onboarding to the pilot. Read in order if you're new; skip
ahead if you already have your bearings.

| # | Doc | Read this if... |
|---|-----|-----------------|
| 00 | [Overview](00-overview.md) | You've been told "use the RunAI cluster" and don't yet know what that means or whether it fits your work |
| 01 | [Access](01-access.md) | You need a login, project assignment, or storage quota and aren't sure what to ask DoIT |
| 02 | [First workspace](02-first-workspace.md) | You want a working Jupyter notebook on the cluster with this repo cloned and a shared model loaded, in ~15 minutes |
| 03 | [Share a model as a vLLM endpoint](03-share-as-endpoint.md) | You want to host a model once and have multiple users / workloads hit it via HTTP, instead of every user loading their own copy onto a GPU |
| 04 | [Storage](04-storage.md) | You need to know where data lives — short-term scratch through cluster-wide shared datasets — and how to get it from "a drive in my lab" to "mountable in a workload" |
| 05 | [Examples](05-examples.md) | You're ready to deploy something — pointers to the OCR pipeline, the RAG/chatbot, and the patterns to copy when building your own |

The OCR-specific and RAG-specific deployment guides live in the app
READMEs — [`ocr_app/README.md`](../ocr_app/README.md) and
[`rag_app/README.md`](../rag_app/README.md) — with per-step details
under each app's `docs/`. Those assume you've already worked through
00–04 here.

> **Audience assumption.** These docs are written for a researcher or
> lab admin who can use a terminal and edit code, but who is not a
> Kubernetes administrator. If you *are* the cluster admin running
> RunAI itself, NVIDIA's [official
> docs](https://run-ai-docs.nvidia.com/) cover the install/operate
> side that this guide deliberately skips.
