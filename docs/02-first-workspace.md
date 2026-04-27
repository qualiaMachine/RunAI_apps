# 02 — Your First Workspace

> **Step 2** in the [New User Guide](README.md). Read [00 Overview](00-overview.md)
> first if you haven't.

By the end of this doc you'll have a Jupyter workspace running on the
cluster, this repo cloned inside it, and a small text-generation
example loaded from the shared model weights — proving end-to-end that
the cluster, the workspace, and the shared-models Data Volume all work
together.

This is intentionally minimal — no fractional GPU tricks, no vLLM
servers, no autoscaling. Just one workspace and one Python cell that
actually loads a model.

## Prerequisite

You need a project on the cluster (see [01 Access](README.md) once
that doc exists; for now ask your DoIT contact). You also need the
`shared-models` Data Volume to be available on the cluster — confirm
with `Data & Storage > Data Volumes` in the RunAI UI. If you don't see
it, your cluster hasn't been provisioned with shared models yet; see
[`rag_app/docs/setup-shared-models.md`](../rag_app/docs/setup-shared-models.md)
*(advanced)*.

## Step A. Create the workspace

1. RunAI UI > **Workloads** > **+ NEW WORKLOAD** > **Workspace**
2. **Project:** select your project.
3. **Template:** click **Start from scratch**. The other tiles are
   preconfigured templates — they'll skip past the fields below and
   leave you guessing what got set. Build it up by hand the first
   time so the moving parts are visible.
4. **Workspace name:** `first-workspace`
5. **Environment image** — Custom image:
   - **Image URL:** `nvcr.io/nvidia/pytorch:25.02-py3`
6. **Tools** — add Jupyter on port 8888.
7. **Runtime settings:**
   - **Command:** `bash`
   - **Arguments:**
     ```
     -c "pip install --no-cache-dir transformers accelerate; jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*'"
     ```
   - **Environment variables:**

     | Name | Value |
     |------|-------|
     | `HF_HOME` | `/models/.cache/huggingface` |
     | `HF_HUB_CACHE` | `/models/.cache/huggingface` |
     | `HF_HUB_OFFLINE` | `1` |

     `HF_HUB_OFFLINE=1` makes sure the workspace never silently
     downloads a model — if the cache isn't where it should be, you
     get a clear error instead of a multi-GB surprise download.
8. **Compute resources:**
   - **GPU devices:** `1`
   - **GPU fractioning:** Enabled — `25%` (≈20 GB on an 80 GB H100,
     enough to load Qwen2.5-7B in bf16)
9. **Data & storage:**
   - **+ Data Volume** > pick `shared-models`, mount path `/models`,
     read-only.
   - *(optional)* **+ Volume** > `local-path`, container path
     `/work`, persistent. This is where you'll save notebooks.
10. **CREATE WORKSPACE**.

Wait for the status to flip to `Running`. With image caching this is
~30 seconds; cold-pulling the PyTorch image is ~3 minutes the first
time.

## Step B. Open Jupyter and clone the repo

1. Click the workspace name, then click the **Jupyter** tool link.
2. In Jupyter, open a Terminal (File > New > Terminal).
3. Clone the repo into the persistent volume so it survives restarts:
   ```
   cd /work
   git clone https://github.com/qualiaMachine/RunAI_apps.git
   ls /work/RunAI_apps   # README.md, ocr_app/, rag_app/, ...
   ```

Now back to the file browser, navigate into `/work/RunAI_apps/` —
you'll see all the docs and code from the repo. Notebooks under
`ocr_app/notebooks/` and `rag_app/` will run from here once their
Data Sources are attached, but that's the job of those apps' own
deployment guides.

## Step C. Load a shared model and generate

Create a new notebook in `/work/` (Jupyter > File > New > Notebook,
pick the Python 3 kernel).

### Cell 1 — list what's on the shared-models volume

```python
from pathlib import Path

cache = Path("/models/.cache/huggingface")
for d in sorted(cache.glob("models--*")):
    # HF cache layout: models--<org>--<name>
    print(d.name.replace("models--", "").replace("--", "/"))
```

You should see a list of HuggingFace repo IDs that are pre-cached on
the cluster — likely some Qwen models, Jina V4, possibly a Qwen-VL,
etc. If the cell prints nothing, the `shared-models` Data Volume
didn't mount; check `Data & Storage > Data Volumes` in the RunAI UI.

> **Need a model that isn't listed?** Email Chris or Mike at DoIT
> (see [01 Access](01-access.md)) — adding a model to `shared-models`
> is a quick admin task, much cheaper than every workload downloading
> its own copy. Tell them the HuggingFace repo ID and your rough
> timeline.

For the rest of this walkthrough we'll use `Qwen/Qwen2.5-7B-Instruct`
— if your output above doesn't include it, swap in any other text
generation model from the list (the loading code below works
unchanged).

### Cell 2 — load the model

```python
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert os.environ.get("HF_HUB_OFFLINE") == "1", (
    "HF_HUB_OFFLINE not set — re-create the workspace with the env var"
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
print(model.dtype, next(model.parameters()).device)
```

First run takes ~30–60 seconds — `transformers` reads the safetensors
shards from the read-only PVC into GPU RAM. After that the weights
stay loaded as long as the kernel is alive.

### Cell 3 — generate

```python
messages = [
    {"role": "system", "content": "You are a concise research assistant."},
    {"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"},
]
inputs = tok.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda:0")

out = model.generate(inputs, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))
```

If you get a coherent answer, everything is working: the workspace can
schedule on a GPU, the shared-models Data Volume mounted correctly,
and offline-mode reads succeed. You're ready for the actual app
deployments.

## Step D. Stop the workspace

GPUs are scarce. When you're not using the workspace, stop it from the
RunAI UI — the volume and the cloned repo persist, and you can Start
it back up later. Don't leave it running idle.

## What this exercise does and doesn't show

**Does:**
- Cluster scheduling actually works for your project
- The `shared-models` Data Volume mounts correctly read-only
- `HF_HUB_OFFLINE=1` prevents accidental downloads
- Your account has GPU quota

**Doesn't:**
- Test multi-user concurrency or share the loaded model with anyone
  else — for that, the next step is to host it as an endpoint. See
  [03 Share a Model as a vLLM Endpoint](03-share-as-endpoint.md)
- Demonstrate sharing data with other projects (that's the
  [04 Storage](04-storage.md) walkthrough)
- Cover anything OCR-specific (vision-language models, chunking,
  prompts — see [`ocr_app/README.md`](../ocr_app/README.md))

Once you understand both the direct-load pattern (above) and the
endpoint pattern (03), head to [05 Examples](05-examples.md) and pick
whichever app matches what you're trying to build.
