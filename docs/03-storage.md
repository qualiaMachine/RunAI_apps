# 03 — Storage

> **Step 3** in the [New User Guide](README.md). Read [00 Overview](00-overview.md)
> first if "Data Source vs Data Volume" doesn't ring a bell.

Storage is where new users get the most lost — RunAI exposes several
overlapping primitives, and the right one depends on how long the data
needs to live and who else needs to read it. This doc walks you through
the spectrum hands-on: you'll create a starter PVC, populate it, and
then promote it to a cluster-shared Data Volume. By the end you'll have
felt the difference between the two main concepts instead of just read
about it.

## The three time horizons

Match your data to the shortest horizon that fits — it's the cheapest
to set up and the easiest to clean up.

| Horizon | What it is | Survives... | When to use |
|---------|------------|-------------|-------------|
| **Ephemeral** | The pod's own filesystem (no PVC attached) | The current pod only — gone on restart | Cache files, scratch output you'll re-derive, anything you don't care about |
| **Project** | A **PVC Data Source** (or NFS Data Source) attached to your workloads | Workspace restarts; lives until you delete the Data Source | Notebooks, in-progress datasets, model output, anything one project needs |
| **Cluster-shared** | A **Data Volume** wrapped around a populated PVC, scoped to other projects/departments | The origin project keeps RW; sharers mount RO | Pre-trained model weights, shared reference datasets, anything multiple projects read |

Most pilots only need the middle tier. The cluster-shared tier comes up
when you have something genuinely worth sharing (a 20 GB model
download, a curated corpus that took a week to build).

## Hands-on: starter PVC → populated → shared Data Volume

This walkthrough takes ~15 minutes if your access is already set up
(see [01 Access](README.md)). At the end you'll have:

1. A working Project-tier PVC Data Source you wrote a file to
2. A Cluster-shared Data Volume mounted into a second workspace,
   read-only, demonstrating the difference

You can keep these as your sandbox or delete them — they're cheap
either way.

### Step A. Create a starter PVC Data Source

In the RunAI UI:

1. **Data & Storage** > **Data Sources** > **+ NEW DATA SOURCE** > **PVC**
2. Settings:
   - **Scope:** your project
   - **Name:** `<your-username>-sandbox`
   - **PVC type:** New PVC
   - **Storage class:** any RWX-capable class your cluster offers — ask
     your cluster admin once for the list. `local-path` works for a
     single-node test but won't survive moving across nodes.
   - **Access mode:** Read-write by many nodes (RWX) if available,
     otherwise Read-write by one node (RWO).
   - **Claim size:** `5Gi` is plenty for the walkthrough
   - **Container path:** `/sandbox`
3. **CREATE DATA SOURCE**.

Wait for the status to flip to "No issues found." If it stays in
"Issues found," something's wrong with the StorageClass — flag your
cluster admin.

### Step B. Mount it from a workspace and write a file

1. **Workloads** > **+ NEW WORKLOAD** > **Workspace**.
2. Basic settings: pick your project, name it `sandbox-1`, leave the
   defaults.
3. Environment image: `nvcr.io/nvidia/pytorch:25.02-py3` (any image with
   `bash` works — this one is here because the OCR/RAG docs use it).
4. **Tools:** add Jupyter on port 8888.
5. **Runtime settings — Arguments:**
   ```
   -c "jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*'"
   ```
6. **Compute resources:** 0 GPUs, default CPU/memory.
7. **Data & storage:** attach the `<your-username>-sandbox` Data Source
   you just created. Mount path should default to `/sandbox`.
8. **CREATE WORKSPACE**.

Once it's `Running`:

1. Click the workspace name, then click the **Jupyter** tool link.
2. Open a Terminal in Jupyter (File > New > Terminal).
3. Write something to the PVC:

   ```
   cd /sandbox
   echo "first words on cluster storage" > hello.txt
   ls -la
   ```

You've just written to the Project-tier PVC. Stop and Start the
workspace — `hello.txt` is still there. Delete the workspace entirely
and create a new one with the same Data Source attached — `hello.txt`
is *still* there. That's the "Project" horizon: the Data Source
outlives any individual workload.

### Step C. Promote the PVC to a cluster-shared Data Volume

Now suppose another lab in another RunAI project wants to read your
`hello.txt`. They can't — the PVC is scoped to your project. Solve it
by creating a Data Volume that wraps the same underlying PVC and
exposes it cluster-wide.

1. **Data & Storage** > **Data Volumes** > **+ NEW DATA VOLUME**.
   (If you don't see "Data Volumes" in the menu, your cluster admin
   hasn't enabled the feature — see the [Data Volumes
   docs](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/assets/data-volumes).)
2. Settings:
   - **Origin project:** your project
   - **Origin PVC:** `<your-username>-sandbox`
   - **Name:** `<your-username>-sandbox-shared`
   - **Scope(s) that can mount the volume:** add the cluster scope, or
     a second project you have access to.
3. **CREATE DATA VOLUME**.

### Step D. Feel the difference

In a second workspace (in your project, or a different project that
the Data Volume is shared with), attach the **Data Volume** instead of
the Data Source — same data, different mount.

```
cd /sandbox
ls -la                              # hello.txt is there
echo "second words" >> hello.txt    # this fails — read-only
```

The write fails because Data Volume mounts are always read-only, even
in the origin project. Go back to your first workspace, which mounts
the **Data Source** directly, and the same write succeeds. That's the
core difference: Data Source = the storage primitive (RW for the
owning project), Data Volume = the share-with-others wrapper (RO).

> **Common gotcha.** Once you've shared a Data Volume, it's tempting
> to attach the Data Volume to your *own* workspaces too "just to be
> consistent." Don't — you'll lose RW access to your own data. Inside
> the origin project, always attach the underlying Data Source.

## Real-world recipes

Once the model clicks, you'll usually use one of these recipes rather
than building from scratch:

### "I have lab data on a USB drive and need to use it on the cluster."

The cluster can't see your drive. Stage it somewhere routable
(institutional cloud bucket, or have your cluster admin mount the lab's
file server as NFS), then create a Data Source pointing at the staged
location. The OCR app's [setup-storage.md](../ocr_app/docs/setup-storage.md)
walks through the three concrete paths (PoC drag-drop, NFS Data Source,
PVC + cloud staging) including the firewall reality check.

### "I need a 20 GB model that I'd like to share with other groups."

Use [`rag_app/docs/setup-shared-models.md`](../rag_app/docs/setup-shared-models.md).
That doc creates a writable PVC in a dedicated `shared-models` project,
populates it with a model download script, and shares it as a
cluster-wide Data Volume — exactly the pattern from the walkthrough
above, applied to a real artifact.

### "I'm running OCR over a corpus of PDFs."

You want a Data Source for the input documents (NFS if the source data
lives on a network share, otherwise a populated PVC). The notebook
pipeline writes its output into the workspace's persistent volume, so
no separate output Data Source is needed. See
[`ocr_app/docs/setup-storage.md`](../ocr_app/docs/setup-storage.md).

### "I need a vector index that survives across builds."

Project-tier PVC. The [RAG app's setup-workspace
doc](../rag_app/docs/setup-workspace.md) creates one and shows how to
build the index inside it.

## Questions to ask your cluster admin once

If you're going to be doing storage work regularly, get these answers
once and you'll rarely need to bother them again:

1. **What StorageClasses are installed?** Which support
   `ReadWriteMany` (RWX)? This is the dropdown that appears in *New
   PVC*. Without an RWX class, only one workload at a time can mount
   the PVC RW.
2. **Is the NFS Data Source type usable?** Depends on whether an NFS
   CSI driver is installed or the in-tree NFS volume support is
   available, and whether cluster nodes can route to your specific NFS
   server (firewalls!).
3. **Are S3 Data Sources enabled?** If yes, you can mount cloud
   buckets directly without a populate-then-pull step.
4. **Is the Data Volumes feature enabled?** It's per-cluster
   configuration. Without it, "share across projects" requires
   workarounds.

## Cleanup

When you're done with the walkthrough:

1. Delete the second workspace (and the first, if you don't want the
   sandbox).
2. **Data & Storage** > **Data Volumes** > delete `<your-username>-sandbox-shared`.
3. **Data & Storage** > **Data Sources** > delete `<your-username>-sandbox`.
   This deletes the underlying PVC too — `hello.txt` goes with it.

If you skip the cleanup, the storage stays attached to your project's
quota indefinitely. Cheap to leave but easy to forget about — most
pilots accumulate a few sandbox PVCs that nobody remembers creating.
