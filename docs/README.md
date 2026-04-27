# New User Guide

A progressive walkthrough of the RunAI cluster for researchers and lab
admins onboarding to the pilot. Read in order if you're new; skip
ahead if you already have your bearings.

| # | Doc | Read this if... |
|---|-----|-----------------|
| 00 | [Overview](00-overview.md) | You've been told "use the RunAI cluster" and don't yet know what that means or whether it fits your work |
| 01 | *Access* (TBD) | You need a login, project assignment, or storage quota and aren't sure what to ask DoIT |
| 02 | *First workspace* (TBD) | You want a working Jupyter notebook on the cluster with this repo cloned, in 10 minutes |
| 03 | [Storage](03-storage.md) | You need to know where data lives — short-term scratch through cluster-wide shared datasets — and how to get it from "a drive in my lab" to "mountable in a workload" |
| 04 | *Pick an app* (TBD) | You're ready to deploy something — choosing between the OCR pipeline, the RAG/chatbot, or your own workload |
| 05 | *Going to production* (TBD) | Your pilot worked and you need autoscaling, multi-user access, batch ingestion, or shared model weights |

Sections marked **TBD** are stubs; the framing in 00 explains where they
fit. The OCR-specific and RAG-specific deployment guides live under
[`ocr_app/docs/`](../ocr_app/docs/README.md) and
[`rag_app/docs/`](../rag_app/docs/README.md) — those assume you've
already worked through 00–03 here.

> **Audience assumption.** These docs are written for a researcher or
> lab admin who can use a terminal and edit code, but who is not a
> Kubernetes administrator. If you *are* the cluster admin running
> RunAI itself, NVIDIA's [official
> docs](https://run-ai-docs.nvidia.com/) cover the install/operate
> side that this guide deliberately skips.
