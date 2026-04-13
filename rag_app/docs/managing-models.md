# Managing Models

Model weights live on a shared PVC mounted at `/models/.cache/huggingface/`.
If you haven't created your own PVC yet, see
**[Setup Shared Models PVC](setup-shared-models.md)** first.

---

## vLLM compatibility

Not every HuggingFace model works with vLLM. Before choosing a new LLM,
check the [vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models/).
Well-supported families include:

- **Qwen** (Qwen2, Qwen2.5, Qwen3, Qwen3.5) — first-class support,
  [official deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- **Llama** (Llama 2, Llama 3, Llama 3.1, Llama 4)
- **Mistral / Mixtral**
- **Gemma** (Gemma 2)
- **Phi** (Phi-3, Phi-4)

Models that use non-standard architectures or custom generation code
(e.g., some multimodal or retrieval-augmented models) may not be
supported. When in doubt, search the
[vLLM GitHub issues](https://github.com/vllm-project/vllm/issues) for
the model name.

**Quantization:** vLLM supports AWQ and GPTQ quantized models out of
the box (pass `--quantization awq` or `--quantization gptq`). FP8
quantized models (e.g., `Qwen3-8B-FP8`) work on Ada Lovelace / Hopper
GPUs natively, and on Ampere GPUs via FP8 Marlin (vLLM v0.9.0+).

---

## Adding or updating models on the admin's shared PVC

The admin's PVC (`shared-model-repository`) lives in the **`shared-models`**
project. If you have access to that project, you can write to it directly
using the pre-built **`update-shared-models`** workspace:

1. In the RunAI UI, go to **Workloads**
2. Switch to the **`shared-models`** project
3. Find **`update-shared-models`** and **Start** it
4. Once running, **Connect** > open a terminal
5. Run the provisioning script directly from the PVC:

```bash
# List models and confirm PVC is writable
python /models/provision_shared_models.py list

# Download a new model
python /models/provision_shared_models.py download <org>/<model-name>
# e.g. python /models/provision_shared_models.py download Qwen/Qwen2.5-14B-Instruct

# Download OpenScholar 8B (Llama 3.1 8B fine-tuned for scientific synthesis)
python /models/provision_shared_models.py download OpenSciLM/Llama-3.1_OpenScholar-8B

# Download specific files (e.g. Jina V4 adapters)
python /models/provision_shared_models.py download jinaai/jina-embeddings-v4 --include "adapters/*"

# Verify a model has all required files
python /models/provision_shared_models.py verify jinaai/jina-embeddings-v4
```

6. **Stop** the workspace when done to free resources

> **Tip:** The provisioning script lives on the PVC at
> `/models/provision_shared_models.py` so it persists across workspace
> restarts — no need to re-upload it each time. To update the script,
> copy the latest version from the repo:
> ```bash
> cp scripts/provision_shared_models.py /models/provision_shared_models.py
> ```

> **Important:** The `update-shared-models` workspace uses `local-path`
> (RWO) storage, so **only one workspace can mount the PVC read-write at
> a time**. Make sure no other workspace in `shared-models` is running
> with the same PVC before starting.

If you **don't** have access to the `shared-models` project, either ask
an admin to add models on your behalf, or create your own PVC — see
[Setup Shared Models PVC](setup-shared-models.md).

## Adding a new model to your own PVC

If you created your own PVC (via [Setup Shared Models](setup-shared-models.md)),
re-start your provisioning Workspace and use the same script:

```bash
python /models/provision_shared_models.py download <org>/<model-name>
```

## Swapping the LLM (e.g., Qwen 7B → Llama 3 8B)

1. Make sure the new model is on the PVC (see above)
2. In the RunAI UI, edit the `wattbot-chat` job's arguments: change `--model`
3. Restart the job (the app auto-detects the model name from vLLM)
4. Restart both jobs

No code changes needed. The embedding model and vector DB are unchanged.

## Swapping the embedding model

Changing the embedding model requires rebuilding the vector index:

1. Download the new model to the PVC (see above)
2. Update the index build config and re-run [Step 0e](setup-workspace.md#0e-build-the-vector-index-writes-to-ppvc)
3. Update `wattbot-embedding` env vars (`EMBEDDING_MODEL`, `EMBEDDING_DIM`)
4. Restart the embedding server

---

## Models currently on the admin's shared PVC

Based on `ls /models/.cache/huggingface/`:

| Model | Est. Size | Notes |
|-------|-----------|-------|
| `Qwen/Qwen1.5-110B-Chat` | ~207 GB | Legacy model |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | |
| `Qwen/Qwen2.5-72B-Instruct` | ~135 GB | |
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~57 GB | MoE |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | ~152 GB | MoE (active ~3B) |
| `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` | ~76 GB | MoE FP8 |
| `Qwen/Qwen3.5-35B-A3B` | ~67 GB | MoE (active ~3B) |
| `OpenSciLM/Llama-3.1_OpenScholar-8B` | ~16 GB | Llama 3.1 8B fine-tuned for scientific synthesis |
| `jinaai/jina-embeddings-v4` | ~7 GB | Used by embedding server. Includes `adapters/` |

---

## Why the shared PVC is read-only

RunAI **Data Volumes** are read-only by design when shared across
projects. This is not a bug or misconfiguration — it's how RunAI
ensures data integrity:

> "Shared data volumes are mounted with read-only permissions. Any
> modifications must be made by writing to the **original PVC** used
> to create the data volume."
> — [RunAI Data Volumes docs](https://run-ai-docs.nvidia.com/saas/workloads-in-nvidia-run-ai/assets/data-volumes)

The lifecycle is:

1. A **Data admin** creates a **PVC data source** (writable)
2. They populate it with model weights from a Workspace
3. They wrap it in a **Data Volume** and share it across projects
4. All consumers (including your workspaces) get **read-only** access

To write to the existing `shared-models` PVC, you need access to the
**`shared-models` project** and must use a workspace that mounts the
**data source** (not the data volume). See
[Adding or updating models on the admin's shared PVC](#adding-or-updating-models-on-the-admins-shared-pvc)
above. Alternatively, create your own PVC — see below.

### How the embedding server handles read-only PVCs

The `embedding_server.py` script creates a writable overlay
automatically — no manual workaround needed:

1. Creates a writable cache at `/tmp/hf_home`
2. Symlinks model weight directories from the PVC (read-only is fine
   for reading weights)
3. Creates writable `refs/`, `.no_exist/` directories for HF metadata
4. Redirects xet logging and pip cache to `/tmp`
5. Auto-downloads missing Jina V4 adapters to `/tmp` on cold start
   (no longer needed — adapters are now on the PVC)

**Result:** Zero "Read-only file system" errors in logs.

### Cache directory structure

```
/models/                              ← shared-models PVC mount (read-only)
└── .cache/
    └── huggingface/                  ← HF_HOME points here
        ├── models--jinaai--jina-embeddings-v4/
        │   ├── snapshots/
        │   │   └── <commit-hash>/    ← model weights + config
        │   │       ├── model-00001-of-00002.safetensors
        │   │       ├── config.json
        │   │       ├── tokenizer.json
        │   │       ├── adapters/     ← LoRA adapters (if present)
        │   │       └── ...
        │   └── refs/
        │       └── main              ← commit hash pointer
        ├── models--Qwen--Qwen2.5-7B-Instruct/
        │   └── snapshots/...
        └── ...
```

---

## Creating your own shared models PVC

See **[Setup Shared Models PVC](setup-shared-models.md)** for the
complete guide. In short:

1. Create a PVC data source in your project (you own it, you can write to it)
2. Spin up a provisioning Workspace, download models with `huggingface-cli`
3. Optionally wrap as a Data Volume for cross-project sharing
4. Stop the Workspace — re-start anytime to add/update models
