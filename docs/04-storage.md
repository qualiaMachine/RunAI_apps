# 04 — Storage

> **Step 4** in the [New User Guide](../README.md#new-user-guide). Read [00 Overview](00-overview.md)
> first if "Data Source vs Data Volume" doesn't ring a bell.

Storage is where new users get the most lost — RunAI exposes several
overlapping primitives, and the right one depends on how long the data
needs to live and who else needs to read it. This doc walks you through
the spectrum hands-on: you'll create a PVC the reliable way (as part
of a workspace, so it actually binds), populate it, register it as a
shareable **Data Source**, and finally promote it to a cluster-shared
**Data Volume**. By the end you'll have felt the difference between
the three concepts instead of just read about them.

## The three time horizons

Match your data to the shortest horizon that fits — it's the cheapest
to set up and the easiest to clean up.

| Horizon | What it is | Survives... | When to use |
|---------|------------|-------------|-------------|
| **Ephemeral** | The pod's own filesystem (no PVC attached) | The current pod only — gone on restart | Cache files, scratch output you'll re-derive, anything you don't care about |
| **Project** | A **PVC Data Source** (or NFS Data Source) attached to your workloads | Workspace restarts and deletion; lives until you delete the Data Source | Notebooks, in-progress datasets, model output, anything one project needs across multiple workloads |
| **Cluster-shared** | A **Data Volume** wrapped around a populated PVC, scoped to other projects/departments | The origin project keeps RW; sharers mount RO | Pre-trained model weights, shared reference datasets, anything multiple projects read |

Most pilots only need the middle tier. The cluster-shared tier comes up
when you have something genuinely worth sharing (a 20 GB model
download, a curated corpus that took a week to build).

## Hands-on: workspace-PVC → Data Source → Data Volume

This walkthrough takes ~15 minutes if your access is already set up
(see [01 Access](01-access.md)). At the end you'll have:

1. A workspace whose attached PVC has a file you wrote
2. That PVC registered as a **Data Source** in your project, attachable
   to any other workload by name
3. A **Data Volume** wrapped around it that any other project on the
   cluster could mount read-only

You can keep these as your sandbox or delete them — they're cheap
either way.

### Why the order matters on this cluster

You *could* try to create the Data Source first (RunAI's UI lets you).
But the cluster's StorageClass uses `WaitForFirstConsumer` binding
mode, which means a freshly-provisioned PVC stays in `Pending` state
until a pod actually mounts it. Wrapping a `Pending` PVC in a Data
Volume fails with `OriginalPvcNotBound` (see
[`rag_app/docs/troubleshooting.md`](../rag_app/docs/troubleshooting.md#pvc-wont-bind--originalpvcnotbound-error)).
The walkthrough below sidesteps that entirely by creating the PVC as
part of a workspace, so the workspace's pod is already there to
consume it the moment it exists. Binding is automatic.

### Step A. Create a workspace with a fresh PVC

1. **Workloads** > **+ NEW WORKLOAD** > **Workspace**
2. **Project:** your project
3. **Template:** **Start from scratch**
4. **Workspace name:** `sandbox-workspace`
5. **Environment image:** `nvcr.io/nvidia/pytorch:25.02-py3`
   *(any image with `bash` works — pick this one if you don't have
   a preference)*
6. **Tools:** Jupyter on port 8888
7. **Runtime settings — Arguments:**
   ```
   -c "jupyter-lab --ip=0.0.0.0 --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*'"
   ```
8. **Compute resources:** `0` GPUs, default CPU/memory (no model
   loading happens here)
9. **Data & storage** — click **+ Volume** (or "+ New PVC", depending
   on your RunAI version). Fill in:
   - **Storage class:** any RWX-capable class your cluster offers —
     ask your cluster admin once for the list. `local-path` works
     for a single-node sanity check but pins the data to one node.
   - **Access mode:** Read-write by many nodes (RWX) if available,
     otherwise Read-write by one node (RWO).
   - **Claim size:** `5Gi`
   - **Volume mode:** Filesystem
   - **Container path:** `/sandbox`
   - **Volume persistency:** Persistent *(survives Stop/Start)*
10. **CREATE WORKSPACE**

Wait for the workspace to flip to `Running`. The PVC binds the moment
the pod schedules — no `Pending` purgatory, no `OriginalPvcNotBound`.

### Step B. Open Jupyter, write a file, observe persistence

1. Click the workspace name → **Jupyter**.
2. File > New > Terminal.
3. Write to `/sandbox`:
   ```
   cd /sandbox
   echo "first words on cluster storage" > hello.txt
   ls -la
   ```
4. **Stop** the workspace from the RunAI UI, then **Start** it again.
   Re-open Jupyter, re-open the terminal, `cat /sandbox/hello.txt` —
   still there. That's "Volume persistency: Persistent" doing its
   job; the PVC and its contents survive Stop/Start cycles.

But the PVC right now is tied to *this workspace's lifecycle*. If you
delete the workspace, the inline PVC goes with it. The next two steps
fix that.

### Step C. Register the PVC as a Data Source

This is what makes the storage attachable to *other* workloads in
your project — a second workspace, an Inference workload, a batch
job — by name, without re-typing the storage class / access mode /
size each time. It also gives the PVC a lifecycle independent of the
original workspace, so you can delete `sandbox-workspace` later
without losing the data.

1. **Data & Storage** > **Data Sources** > **+ NEW DATA SOURCE** >
   **PVC**
2. Settings:
   - **Scope:** your project
   - **Name:** `<your-username>-sandbox`
   - **PVC type:** **Existing PVC**
   - **Select a PVC:** pick the one your workspace just created.
     RunAI auto-names inline PVCs (typically something like
     `sandbox-workspace-...`); if more than one shows up, the
     newest entry is yours.
   - **Container path:** `/sandbox`
3. **CREATE DATA SOURCE**.

Status should flip to "No issues found" immediately — the PVC was
already bound back in Step A.

You can verify by creating a *second* workspace, attaching the
`<your-username>-sandbox` Data Source under Data & storage, and
mounting it at `/sandbox`. `cat /sandbox/hello.txt` from the second
workspace returns the same content. That's the "Project" horizon:
the Data Source outlives any individual workspace and is reusable
across them.

### Step D. Promote the PVC to a cluster-shared Data Volume

Now suppose another lab in another RunAI project wants to read your
`hello.txt`. They can't — the Data Source is scoped to your project.
A **Data Volume** wraps the same underlying PVC and exposes it
cluster-wide, read-only.

1. **Data & Storage** > **Data Volumes** > **+ NEW DATA VOLUME**.
   *(If you don't see Data Volumes in the menu, your cluster admin
   hasn't enabled the feature — see the [Data Volumes
   docs](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/assets/data-volumes).)*
2. Settings:
   - **Origin project:** your project
   - **Origin PVC:** the one your workspace and Data Source both
     point at
   - **Name:** `<your-username>-sandbox-shared`
   - **Scope(s) that can mount the volume:** add the cluster scope,
     or a second project you have access to.
3. **CREATE DATA VOLUME**.

### Step E. Feel the difference

In a workspace in a different project (or in your project, if that's
all you have access to), attach the **Data Volume** instead of the
Data Source — same data, different mount.

```
cd /sandbox
ls -la                              # hello.txt is there
echo "second words" >> hello.txt    # this fails — read-only
```

The write fails because Data Volume mounts are always read-only,
even in the origin project. Go back to a workspace that mounts the
**Data Source** directly, and the same write succeeds. That's the
core difference: Data Source = the storage primitive (RW for the
owning project), Data Volume = the share-with-others wrapper (RO).

> **Common gotcha.** Once you've shared a Data Volume, it's tempting
> to attach the Data Volume to your *own* workspaces too "just to be
> consistent." Don't — you'll lose RW access to your own data.
> Inside the origin project, always attach the underlying Data
> Source.

## Real-world recipes

Once the model clicks, you'll usually use one of these recipes rather
than building from scratch:

### "I have lab data on a USB drive and need to use it on the cluster."

The cluster can't see your drive. Stage it somewhere routable
(institutional cloud bucket, or have your cluster admin mount the
lab's file server as NFS), then create a Data Source pointing at the
staged location. The OCR app's
[`setup-storage.md`](../ocr_app/docs/setup-storage.md) walks through
the three concrete paths (PoC drag-drop, NFS Data Source, PVC + cloud
staging) including the firewall reality check.

### "I need a 20 GB model that I'd like to share with other groups."

Use [`rag_app/docs/setup-shared-models.md`](../rag_app/docs/setup-shared-models.md).
That doc creates a writable PVC in a dedicated `shared-models`
project, populates it with a model download script, and shares it as
a cluster-wide Data Volume — the same pattern as Steps A–D, applied
to a real artifact.

### "I'm running OCR over a corpus of PDFs."

You want a Data Source for the input documents (NFS if the source
data lives on a network share, otherwise a populated PVC). The
notebook pipeline writes its output into the workspace's persistent
volume, so no separate output Data Source is needed. See
[`ocr_app/docs/setup-storage.md`](../ocr_app/docs/setup-storage.md).

### "I need a vector index that survives across builds."

Project-tier PVC. The [RAG app's setup-workspace
doc](../rag_app/docs/setup-workspace.md) creates one and shows how
to build the index inside it.

## Questions to ask your cluster admin once

If you're going to be doing storage work regularly, get these answers
once and you'll rarely need to bother them again:

1. **What StorageClasses are installed?** Which support
   `ReadWriteMany` (RWX)? This is the dropdown that appears when
   you click + Volume. Without an RWX class, only one workload at
   a time can mount the PVC RW.
2. **Is the NFS Data Source type usable?** Depends on whether an NFS
   CSI driver is installed or the in-tree NFS volume support is
   available, and whether cluster nodes can route to your specific
   NFS server (firewalls!).
3. **Are S3 Data Sources enabled?** If yes, you can mount cloud
   buckets directly without a populate-then-pull step.
4. **Is the Data Volumes feature enabled?** It's per-cluster
   configuration. Without it, "share across projects" requires
   workarounds.

## Cleanup

When you're done with the walkthrough, delete in roughly the reverse
order you created things — Data Volume → Data Source → workspace —
so each step's dependents are already gone:

1. **Data & Storage** > **Data Volumes** > delete
   `<your-username>-sandbox-shared`.
2. **Data & Storage** > **Data Sources** > delete
   `<your-username>-sandbox`. The underlying PVC stays — it's still
   referenced by the workspace.
3. **Workloads** > delete `sandbox-workspace` (and any second
   workspace you spun up). When the last reference to the PVC goes
   away, the PVC and its data are deleted.

If you skip the cleanup, the storage stays attached to your
project's quota indefinitely. Cheap to leave but easy to forget
about — most pilots accumulate a few sandbox PVCs that nobody
remembers creating.

## Next

You've got the storage primitives down. Time to apply them to a real
deployment — head to [05 Examples](05-examples.md) to pick between
the OCR pipeline and the RAG chatbot, both of which are layered
directly on top of the patterns from 02, 03, and this doc.
