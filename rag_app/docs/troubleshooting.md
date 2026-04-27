# Troubleshooting

## Quick fixes

- **vLLM OOM:** Reduce `--max-model-len` (e.g., 4096) or use `--quantization awq`
- **Embedding server 503:** Model still loading (~30s on first request). Check logs.
- **Streamlit can't connect:** Verify service DNS names match your job names in the RunAI UI
- **Vector DB not found / vec_count=0:** The app found no vectors. This usually means the DB file wasn't copied from the PPVC to a writable location. The PPVC is often read-only, and KVaultNodeStore writes metadata on open, so it silently creates a new empty DB. **Fix:** Ensure your startup command copies the DB: `mkdir -p /tmp/vectordb && cp /wattbot-data/embeddings/wattbot_jinav4.db /tmp/vectordb/`. The app auto-discovers it at `/tmp/vectordb/`, or set `VECTOR_DB_PATH=/tmp/vectordb/wattbot_jinav4.db`. See [deploy-streamlit](deploy-streamlit.md) for the full command. Symlinks to read-only volumes **will not work**.
- **Mismatch errors:** Ensure `EMBEDDING_DIM=1024` matches what was used during index build
- **Job keeps crashing:** Check logs in RunAI UI (click job > Logs tab). Common causes: OOM, missing files, image pull failure

## PVC won't bind / "OriginalPvcNotBound" error

If you create a Data Volume in RunAI and see `OriginalPvcNotBound`, the
underlying PVC hasn't been claimed by any pod yet. Most clusters use
`WaitForFirstConsumer` binding mode, meaning the PVC stays `Pending`
until a workload actually mounts it.

**The fix — create the PVC with your first job:**

1. **Job 1 (e.g., `wattbot-test`):** Create a new workload and
   configure the PVC as part of that job (under **Data & Storage** >
   **New PVC**). When the job starts, the pod claims the PVC and it
   binds automatically.
2. **Next job:** Now go to **Data & Storage** > **Data Volumes** and
   create a Data Volume referencing the already-bound PVC. Attach
   that Data Volume to your next workload — it will mount successfully
   because the PVC is already bound.

**Why this happens:** RunAI's Data Volume wizard creates the PVC
object, but with `WaitForFirstConsumer`, Kubernetes won't actually
bind it to a storage backend until a pod schedules that references
it. Creating the Data Volume *before* any pod uses it leaves the PVC
in a `Pending` state, which RunAI reports as `OriginalPvcNotBound`.
The workaround is to let a job create and claim the PVC first, then
wrap it in a Data Volume afterward.

## Inference workload fails with "Readiness probe failed" on wrong port

**Symptom:** Logs show the server started successfully (e.g.
`Serving on 0.0.0.0:8080`), but the workload transitions to **Failed**
with events like:

```
Readiness probe failed: Get "http://...:8012/": context deadline exceeded
```

**Cause:** The **container port** in the RunAI Inference workload doesn't
match the port the server is actually listening on. Knative uses the
configured container port for its readiness probe. If you set port 8012
but the server listens on 8080, the probe times out and Knative kills
the revision.

**Fix:** Delete and recreate the workload with the correct container port:
- **Embedding server:** `8080` (see [deploy-embedding](deploy-embedding.md))
- **vLLM server:** `8000` (see [deploy-vllm](deploy-vllm.md))

You cannot edit the port of a running Inference workload — you must
delete and recreate it.

## Read-only file system errors in embedding server logs

The writable overlay handles this automatically. If you still see
these errors, ensure you're running the latest `embedding_server.py`
which includes `_setup_hf_cache_overlay()`.

## Missing adapters (Jina V4)

The embedding server auto-downloads adapters to `/tmp` on each cold
start if they're missing from the PVC. To fix permanently, either ask
the admin to add adapters to their PVC, or create your own PVC with
the full model (see [Managing Models](managing-models.md)).

## Can't write to `/models/` — even from the "right" project

There are three common causes, all of which look the same ("Read-only
file system"):

**1. Mounted the Data Volume instead of the Data Source.**
Data Volumes are **always** read-only, even in the creator's project.
Check the workspace config in the RunAI UI — look at what's attached
under "Data Sources" vs "Data Volumes". They can have similar names
(e.g. `shared-models` vs `shared-model-repository`) but behave
differently. You need the **Data Source** (the raw PVC) for write access.

**2. Workspace is in the wrong project.**
The PVC is namespace-scoped. If the data source was created in project
`shared-models` but your workspace is in `jupyter-yourname01`, you
can't write to it — even if the data source is visible in the UI.
Check the project column in the Workloads list. Your provisioning
workspace must be in the **same project** where the PVC was created.

**3. Another workload already has the PVC mounted (RWO).**
If the storage class is `local-path` with access mode `Read-write by
one node` (RWO), only one pod can mount it read-write at a time. If
someone else's workspace is still Running with that PVC mounted, yours
gets downgraded to read-only. **Stop the other workload first**, then
start yours.

**Best fix:** Create your own PVC in your own project — then there's no
ambiguity about write access. See
[Provision Your Own Shared Models PVC](setup-shared-models.md).

## Storage class doesn't support ReadWriteMany

If your cluster only offers ReadWriteOnce (RWO) storage, you can still
use this approach — just ensure only one workload mounts the PVC at a
time during provisioning. After populating, share it as a Data Volume
(read-only, which supports multi-mount regardless of access mode).

## PVC shows 0 bytes used after download

Some network storage classes don't report usage accurately. Check with
`du -sh /models/.cache/huggingface/` from the provisioning workspace
instead of relying on RunAI's storage metrics.
