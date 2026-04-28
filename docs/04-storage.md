# 04 — Storage

> **Step 4** in the [New User Guide](../README.md#new-user-guide). Read [00 Overview](00-overview.md)
> first if "Data Source vs Data Volume" doesn't ring a bell.

RunAI exposes several overlapping primitives, and the right one depends on how long the data
needs to live and who else needs to read it. The good news is that
most users on this cluster are covered by the auto-mounted ~30 GB
user volume that ships with every project: notebooks, intermediate
output, small datasets all just land on it without any setup. This
doc walks you through that default first, then — only if you need
more — through creating a larger PVC inline in a workspace,
registering it as a shareable **Data Source**, and finally promoting
it to a cluster-shared **Data Volume**. By the end you'll have felt
the difference between the three concepts instead of just read about
them.

## The three time horizons

Match your data to the shortest horizon that fits — it's the cheapest
to set up and the easiest to clean up.

| Horizon | What it is | Survives... | When to use |
|---------|------------|-------------|-------------|
| **Ephemeral** | The pod's own filesystem (no PVC attached) | The current pod only — gone on restart | Cache files, scratch output you'll re-derive, anything you don't care about |
| **Project** | A **PVC Data Source** (or NFS Data Source) attached to your workloads. Includes the auto-mounted ~30 GB user volume that ships with every project. | Workspace restarts and deletion; lives until you delete the Data Source | Notebooks, in-progress datasets, model output, anything one project needs across multiple workloads |
| **Cluster-shared** | A **Data Volume** wrapped around a populated PVC, scoped to other projects/departments | The origin project keeps RW; sharers mount RO | Pre-trained model weights, shared reference datasets, anything multiple projects read |

Most pilots only need the middle tier. The cluster-shared tier comes up
when you have something genuinely worth sharing (a 20 GB model
download, a curated corpus that took a week to build).

> **Heads up — you already have ~30 GB.** Every project on this cluster
> ships with a dynamically-mounted user volume of about 30 GB. It
> attaches on its own when a workload starts; you don't pick it from a
> dropdown. Step A below uses it directly. Skip ahead to Step B only
> when you've outgrown 30 GB or need to share storage across projects.

## Hands-on: user volume → workspace PVC → Data Source → Data Volume

This walkthrough takes ~15 minutes if your access is already set up
(see [01 Access](01-access.md)). At the end you'll have:

1. Found the auto-mounted ~30 GB user volume and written a file to it
2. Created a *bigger* PVC inline in a sandbox workspace, in case 30 GB
   isn't enough
3. Registered that PVC as a **Data Source** in your project, attachable
   to any other workload by name
4. Wrapped it in a **Data Volume** that any other project on the cluster
   could mount read-only

Most readers only need Step A. Steps B–E exist so the next time
someone hands you a 200 GB dataset you know what to reach for.

### Step A. Find and use the auto-mounted user volume

Every project on this cluster has a dynamically-attached user volume
of about 30 GB. It shows up automatically the first time you start a
workload — you don't create a PVC, register a Data Source, or click
**+ Volume** to get it. This is the path of least resistance for
notebooks, small datasets, and ad-hoc files.

If you came from [02 First workspace](02-first-workspace.md), your
running workspace also has an inline `/work` PVC (`local-path`) and
the read-only `/models` Data Volume that 02 walked you through
attaching. The auto-mounted user volume is *additional* to those —
it's there whether or not you clicked **+ Volume**. The steps below
show you how to find it and pick it out from the rest.

1. Open the workspace's **Jupyter** tool, then File > New >
   Terminal.
2. List what's mounted:
   ```
   df -h
   ```
   You'll see several entries. Skip `/dev/shm`, `/tmp`, and the
   system filesystems. The remaining ones are your storage:
   - `/work` (or wherever 02 mounted its inline `local-path` PVC) —
     created by you in 02. Tied to this workspace's lifecycle.
   - `/models` — the read-only `shared-models` Data Volume.
   - **A separate ~30 GB filesystem** at some other path — that's
     the auto-mounted user volume. Mount path is set per-project,
     so trust `df -h` over any specific path you read in another
     doc.
3. Write a file to the user volume:
   ```
   cd <your-user-volume-path>
   echo "first words on cluster storage" > hello.txt
   ```
4. **Stop** the workspace from the RunAI UI, then **Start** it
   again. Re-open Jupyter > Terminal, `cat <path>/hello.txt` —
   still there. The user volume is project-scoped, so it survives
   workspace Stop / Start *and* full workspace deletion: a future
   workspace in the same project will mount the same volume with
   the same contents. (Compare with `/work` from 02, which goes
   away the moment you delete that workspace.)

For most pilots that's the entire storage story. The 1 TB high-perf
NVMe drives currently being procured will give you a similar
auto-mounted experience at a much larger size; until then, 30 GB
covers notebooks, intermediate outputs, small corpora, and most
workshop-scale work.

> **Why does 02 still bother creating `/work`?** Mostly inertia, plus
> a deterministic mount path the runtime args can hard-code (the
> Jupyter `--notebook-dir=/work` and the `/work/repo` symlink). Once
> the per-cluster auto-mount path is confirmed and stable, 02 can
> drop the inline `/work` step and lean on the auto-mount alone. For
> now treat `/work` as ephemeral-ish (gone on workspace delete) and
> the auto-mounted user volume as the durable spot.

Move on to Step B only if you've answered yes to one of:

- "I need more than 30 GB and the 1 TB drives aren't here yet."
- "I want this storage shareable across projects, not just inside mine."
- "I need to learn the Data Source / Data Volume model because I'll be
  setting up storage for someone else."

### Why the order matters in Steps B–E

You *could* try to create a Data Source directly (RunAI's UI lets
you). But the cluster's StorageClass uses `WaitForFirstConsumer`
binding mode, which means a freshly-provisioned PVC stays in
`Pending` state until a pod actually mounts it. Wrapping a `Pending`
PVC in a Data Volume fails with `OriginalPvcNotBound` (see
[`rag_app/docs/troubleshooting.md`](../rag_app/docs/troubleshooting.md#pvc-wont-bind--originalpvcnotbound-error)).
The walkthrough below sidesteps that by creating the PVC as part of
a workspace, so the workspace's pod is already there to consume it
the moment it exists. Binding is automatic.

### Step B. Create a workspace with a larger inline PVC

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
   - **Claim size:** pick something larger than the 30 GB you'd
     get for free from Step A. `100Gi` is a reasonable sandbox.
   - **Volume mode:** Filesystem
   - **Container path:** `/sandbox`
   - **Volume persistency:** Persistent *(survives Stop/Start)*
10. **CREATE WORKSPACE**

Wait for the workspace to flip to `Running`. The PVC binds the moment
the pod schedules — no `Pending` purgatory, no `OriginalPvcNotBound`.

Open Jupyter > Terminal and confirm both volumes are visible: `df -h`
will show the 30 GB user volume from Step A *and* your new `/sandbox`
PVC side by side. Drop a file on the new PVC so you can prove the
later steps actually mounted it:

```
echo "first words on cluster storage" > /sandbox/hello.txt
```

This larger PVC is still tied to *this workspace's lifecycle*. If you
delete the workspace, the inline PVC goes with it. Step C fixes that.

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
already bound back in Step B.

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

### "My data already lives on ResearchDrive (or another campus NFS share)."

You don't have to copy it onto the cluster. Ask the cluster admin to
mount the share as an NFS **Data Source** in your project — the
firewall path between the cluster and ResearchDrive is open, so it's
a configuration step on their end, not a network exception request.
You'll get a Data Source you can attach to any workload at a mount
path of your choice, and writes from your workloads land directly on
ResearchDrive.

Caveat: NFS over the network is much slower than the cluster's local
NVMe. Fine for "I want my training script to read straight from where
my data already is" or for staging output that has to end up on
ResearchDrive anyway. Not the right pick for hot loops over millions
of small files, or for vector indexes you'll re-query thousands of
times — for those, copy into a project PVC (Step B above) or wait for
the 1 TB NVMe drives that are on order.

### "I need a 20 GB model that I'd like to share with other groups."

Use [`rag_app/docs/setup-shared-models.md`](../rag_app/docs/setup-shared-models.md).
That doc creates a writable PVC in a dedicated `shared-models`
project, populates it with a model download script, and shares it as
a cluster-wide Data Volume — the same pattern as Steps B–D, applied
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

## What we already know about this cluster

A handful of storage questions come up over and over. Here's what's
already settled on `doit-ai-cluster` so you don't have to re-ask:

- **Per-user storage default.** Every project auto-mounts a ~30 GB
  user volume on workload start (Step A above). No selection step.
- **High-perf bigger storage.** 1 TB NVMe drives are on order; until
  they arrive, "I need 200 GB of fast scratch" doesn't have a clean
  answer beyond "create a larger inline PVC and accept the
  StorageClass you get."
- **NFS to ResearchDrive (and similar campus shares).** Available on
  request from the cluster admin. The firewall path is open, so it's
  a config step on their side rather than a DoIT-wide exception.
  Slower than local NVMe — see the ResearchDrive recipe above for
  when it's the right pick.
- **Data Volumes feature.** Enabled — that's how `shared-models`
  works for everyone today.

## Questions still worth asking your cluster admin once

A few items aren't fully nailed down and only matter once you start
doing serious storage work. Get these answers once and you'll rarely
need to bother them again:

1. **What StorageClasses are installed, and which support
   `ReadWriteMany` (RWX)?** This is the dropdown that appears when
   you click + Volume in Step B. Without an RWX class, only one
   workload at a time can mount your PVC RW.
2. **Are S3 Data Sources enabled?** If yes, you can mount cloud
   buckets directly without a populate-then-pull step.
3. **For non-ResearchDrive NFS shares — are they reachable?** The
   ResearchDrive path is open by default. Other lab or department NFS
   exports may still need a per-host firewall exception.

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
