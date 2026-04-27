# Setup Storage

> **Step 0** in the [deployment guide](../README.md).

Before deploying any workloads, set up the storage. Two spots need to
exist:

| Path | What lives here | RunAI concept |
|------|-----------------|---------------|
| `/data/documents/` | Input PDFs and TIFFs | **Data Source** (NFS or PVC) |
| `/models/` | Qwen3-VL-32B weights (~20 GB) | **Data Volume** (shared from `shared-models` project) |

The notebook pipeline writes its output (per-doc JSON and chunk folders)
into the workspace's persistent volume at `/ocr/` — no separate output
Data Source is required.

## Data Source vs Data Volume

These are two different RunAI concepts and they are easy to mix up:

- **Data Source** — the primitive. "Storage X is mountable into
  workloads." Types include PVC, NFS, S3, Git, ConfigMap, HostPath,
  Secret. Used for OCR's input documents path.
- **Data Volume** — a *cross-scope sharing layer* wrapped around a
  populated PVC Data Source. Sharers mount it read-only; only the origin
  project keeps RW. Used here only for the cluster-wide model weights —
  see [`rag_app/docs/setup-shared-models.md`](../../rag_app/docs/setup-shared-models.md).

For OCR's `/data/documents` you want a **Data Source**. Data Volume
comes up only for the shared model weights.

---

## Input documents — pick one path

### Path A: PoC, ≤ ~5 GB and a few hundred files

Skip the Data Source entirely. Use the workspace's inline volume and
upload through Jupyter:

1. When creating the `ocr-setup` workspace (Step 1), under **Data &
   storage** click **+ Volume** and add a fresh persistent volume:
   - **Storage class:** `local-path`
   - **Container path:** `/ocr`
   - **Persistency:** Persistent
2. Wait for the workspace to reach `Running` and open the **Jupyter**
   tool link.
3. **Drag PDFs/TIFFs from your laptop's file browser straight into the
   Jupyter file browser pane.** They land in `/ocr/` on the persistent
   volume and survive workspace restarts.

Browser uploads start choking around 5 GB or a few hundred files. For
larger inputs, use Path B or Path C.

### Path B: Lab data lives on a network share the cluster can mount (NFS)

> **Will NFS even work here?** Probably not for a quick PoC if the
> cluster sits behind a DoIT-style firewall. Cluster nodes need outbound
> TCP to port `2049` on the NFS server (plus `111` and a few dynamic
> ports for NFSv3). University firewalls usually block that between a
> lab subnet and the datacenter subnet by default.
>
> Before going down this path, ask the cluster admin to test from a
> node:
>
> ```
> nc -zv <nfs-host> 2049
> ```
>
> If that times out, the cluster can't reach your share. Either request
> a firewall exception from DoIT (production timeline, often weeks) or
> use Path C instead (same-day).
>
> Some institutions also forbid NFS exports from end-user lab machines
> as a security policy — confirm before you set up the export.

If reachability checks out:

1. RunAI UI > **Data & Storage** > **Data Sources** > **+ NEW DATA
   SOURCE** > **NFS**
2. Fill in:
   - **Scope:** Your project
   - **Name:** `ocr-documents`
   - **NFS server:** hostname or IP of your NFS host
   - **NFS path:** the export, e.g. `/srv/imaging-archive`
   - **Container path:** `/data/documents`
   - **Prevent data modification:** ✓ (read-only)
3. Click **CREATE DATA SOURCE**.

Attach `ocr-documents` to any workspace under Data & storage during
workload creation. New files added on the source share appear
automatically inside the workload — no copy step.

### Path C: Lab data has to be copied onto cluster storage

Use this when NFS isn't reachable, when DoIT won't open the firewall, or
when the lab's data isn't on a network share to begin with. The flow is:
provision a fresh PVC, then populate it from inside a workspace using
HTTPS-based tools (which the firewall almost always allows outbound).

1. **Create the input PVC.** RunAI UI > **Data Sources** > **+ NEW DATA
   SOURCE** > **PVC** > **New PVC**:
   - **Scope:** Your project
   - **Name:** `ocr-documents`
   - **Storage class:** ask the cluster admin which classes are
     installed and which support `ReadWriteMany`. Pick an RWX class if
     multiple workloads will read it concurrently.
   - **Access mode:** `Read-write by many nodes` (RWX) if available,
     otherwise `Read-write by one node` (RWO).
   - **Claim size:** dataset size + ~20% headroom
   - **Volume mode:** Filesystem
   - **Container path:** `/data/documents`
2. **Stage the lab data once** somewhere the cluster can reach over
   HTTPS. Anything works — institutional Box, OneDrive, Google Drive,
   AWS S3, Azure Blob, etc. From your lab machine:
   ```
   aws s3 sync ./lab-data s3://staging/ocr-input/
   ```
   This part is outside RunAI.
3. **Create the `ocr-setup` workspace** (Step 1) with `ocr-documents`
   attached at `/data/documents`.
4. **Open Jupyter > Terminal** in the running workspace and pull the
   data into the PVC:
   ```
   aws s3 sync s3://staging/ocr-input /data/documents/
   # or
   rclone copy box:lab-data /data/documents/
   # or, if a campus-reachable lab host has SSH:
   rsync -avP user@lab-host.example.edu:/srv/data/ /data/documents/
   ```
5. The terminal can run for hours unattended. Close the browser; the
   workspace pod keeps copying. Reconnect later to check progress.

---

## Shared model weights — `/models`

The vLLM server needs `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` (~20 GB)
cached at `/models/.cache/huggingface`. This is mounted as a **Data
Volume** (not a Data Source) — one PVC in the `shared-models` project,
shared cluster-wide read-only via Data Volume.

**If WattBot is already deployed on this cluster**, the `shared-models`
Data Volume exists. You only need to add the VL model to it:

1. RunAI UI > **Workloads**
2. Switch to the **`shared-models`** project (or whichever project owns
   the PVC with write access)
3. Find **`update-shared-models`** and click **Start**
4. Once running, **Connect** > open a terminal
5. Run:

```
python /models/provision_shared_models.py download QuantTrio/Qwen3-VL-32B-Instruct-AWQ
```

This downloads ~20 GB to the PVC. Takes a few minutes depending on
network speed.

6. Verify:

```
python /models/provision_shared_models.py list
# Should list QuantTrio/Qwen3-VL-32B-Instruct-AWQ
```

7. **Stop** the `update-shared-models` workspace when done.

**If `shared-models` doesn't exist yet on this cluster**, follow
[`rag_app/docs/setup-shared-models.md`](../../rag_app/docs/setup-shared-models.md)
end-to-end — that doc covers creating the PVC, populating it, and
sharing it cluster-wide as a Data Volume.

> **Don't have access to the `shared-models` project?** Ask its admin
> to download the model for you, or see
> [`rag_app/docs/managing-models.md`](../../rag_app/docs/managing-models.md)
> for alternative approaches.

The model only needs to be downloaded once. All inference jobs mount the
Data Volume read-only and load weights from there.

---

## One-time questions for your cluster admin

If you can answer these once, almost all storage work afterward stays in
the RunAI UI:

1. **What StorageClasses are installed?** (`kubectl get sc` on the
   admin's side.) Which ones support `ReadWriteMany`? This determines
   what shows up in the *New PVC* dropdown and which access modes
   actually work.
2. **Is there an NFS or SMB CSI driver installed, or only the in-tree
   NFS volume support?** Affects whether the *NFS Data Source* type is
   usable directly.
3. **Can cluster nodes route to `<your-lab-NFS-host>:2049`?** If yes,
   Path B is on the table. If no, you're on Path C until DoIT opens
   the firewall.
4. **Are S3 Data Sources enabled?** If yes, you may be able to skip the
   "stage in S3 then pull into a PVC" step entirely and mount the
   bucket directly. Form fields vary by RunAI version — confirm before
   documenting any specific bucket.
