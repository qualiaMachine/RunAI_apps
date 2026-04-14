# Setup Data Volumes

> **Step 0** in the [deployment guide](README.md).

Before deploying any workloads, set up the storage.

## Cluster storage layout

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | Shared models PVC | RO (reuse from WattBot if available) | varies | Qwen3-VL-32B weights |
| `/data/documents/` | Input documents PVC | RO for batch jobs | depends on corpus | Source PDFs and TIFFs |
| `/data/extracted/` | Output PVC | RW for batch jobs | ~1-5% of input | Extracted JSON files |

---

## Input documents — `ocr-documents`

This is where your PDFs and TIFFs live. The batch script reads from a
mounted directory — how that directory gets populated depends on your
infrastructure.

### PoC (5 sample docs)

Skip the PVC entirely. Upload files directly to the setup workspace
(Step 2) via Jupyter's file upload button. Use the workspace's local
storage at `/home/jovyan/sample_docs/`.

### Production

The ideal setup is a **direct mount** — the cluster admin creates a
PersistentVolume backed by the source storage (NFS, CIFS/SMB). No copy
needed. The batch workspace reads files over the network from their
original location.

Ask your cluster admin:

> "Can you create a PV pointing to the imaging data share
> (e.g. `nfs-server:/imaging_archive`)? We need read-only access from
> our RunAI project."

Once the PV exists, create a Data Volume in the RunAI UI:

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. **Scope:** Your project
3. **PVC name:** Use the existing PVC backed by the NFS mount
4. **Data volume name:** `ocr-documents`
5. **Mount path:** `/data/documents`

If a direct mount isn't possible (data lives on a completely disconnected
system), you'll need to copy the data onto a cluster PVC. Options include
pulling from inside a workspace (`curl`, `wget`, `rclone` for S3) or
having the data team push to a staging location the cluster can access.

### Ongoing ingestion (~10K docs/month)

With a direct NFS mount, new documents appear automatically as the source
system writes them. The batch script's `--resume` flag means you can
re-run against the same input directory — it skips already-processed files
and only extracts the new ones.

For non-mounted setups, set up a periodic sync (cron job, scheduled
rsync, or rclone) to keep the cluster PVC in sync with the source.

---

## Output storage — `ocr-extracted`

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Configure:
   - **Scope:** Your project
   - **PVC name:** `ocr-extracted` (creates a new PVC)
   - **Data volume name:** `ocr-extracted`
   - **Size:** 100Gi to start (JSON is much smaller than the source docs)
3. Mount path: `/data/extracted`

---

## Shared models PVC — download Qwen3-VL-32B

The vLLM server needs `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` (~20 GB) on the
shared models PVC at `/models/.cache/huggingface`. If the model isn't
there yet, you need to download it before deploying the inference job.

**If you already have the WattBot shared models PVC** (`shared-models`),
reuse it — just add the VL model. If starting fresh, see
[rag_app/docs/setup-shared-models.md](../../rag_app/docs/setup-shared-models.md)
for full PVC creation instructions.

### Download the model

1. In the RunAI UI, go to **Workloads**
2. Switch to the **`shared-models`** project (or whichever project owns
   the PVC with write access)
3. Find **`update-shared-models`** and **Start** it
4. Once running, **Connect** > open a terminal
5. Run:

```bash
python /models/provision_shared_models.py download QuantTrio/Qwen3-VL-32B-Instruct-AWQ
```

This downloads ~64 GB to the PVC. Takes a few minutes depending on
network speed.

6. Verify the model is there:

```bash
python /models/provision_shared_models.py list
# Should show QuantTrio/Qwen3-VL-32B-Instruct-AWQ in the list
```

7. **Stop** the `update-shared-models` workspace when done.

> **Don't have access to the `shared-models` project?** Ask the PVC
> admin to download the model for you, or see
> [managing-models.md](../../rag_app/docs/managing-models.md)
> for alternative approaches.

The model only needs to be downloaded once. All inference jobs mount the
PVC read-only and load the weights from there.
