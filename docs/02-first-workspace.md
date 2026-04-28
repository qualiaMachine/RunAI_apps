# 02 — Your First Workspace

> **Step 2** in the [New User Guide](../README.md#new-user-guide). Read [00 Overview](00-overview.md)
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

You need a project on the cluster (see [01 Access](01-access.md) once
that doc exists; for now ask your DoIT contact). You also need the
`shared-models` Data Volume to be available on the cluster — confirm
with `Data & Storage > Data Volumes` in the RunAI UI. If you don't see
it, talk to cluster admin.

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
     -c "curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp; mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null; ln -sf /tmp/RunAI_apps /work/repo; pip install --no-cache-dir transformers accelerate; jupyter-lab --ip=0.0.0.0 --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/work"
     ```

     Yes, this is annoying. Most of these are RunAI / proxy /
     headless-container glue that has nothing to do with your actual
     work, and you'll need essentially the same string for every
     workspace you create. Once you have one workspace dialed in,
     save it as a Workload Template (RunAI UI > **Workload manager**
     > **Templates** > **+ NEW TEMPLATE** from your existing
     workspace) so future workspaces start with all of this
     pre-filled and you only edit what differs.

     What that string actually does, piece by piece:

     | Chunk | Why it's there |
     |-------|----------------|
     | `-c "..."` | Tells `bash` to run the rest as a shell command, then exit. The whole arguments value is one string. |
     | `curl -sL https://github.com/.../main.tar.gz \| tar xz -C /tmp` | Pull the current `main` branch as a tarball and unpack it under `/tmp`. Faster and lighter than `git clone` (no `.git` history) and doesn't require git on the image. |
     | `mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null` | GitHub's tarball unpacks to `<repo>-<branch>/`. Rename to a stable path. The redirect swallows the harmless "directory already exists" error on subsequent restarts. |
     | `ln -sf /tmp/RunAI_apps /work/repo` | Drop a symlink into the persistent volume so `repo/` shows up in Jupyter's file browser next to your notebooks. The actual code lives in ephemeral `/tmp` and refreshes from GitHub on every restart — no stale local copy to worry about. |
     | `pip install --no-cache-dir transformers accelerate` | The PyTorch base image doesn't include HuggingFace `transformers`. `--no-cache-dir` skips writing wheel caches inside the pod (the GPU image is already big). pip installs go to the pod's ephemeral system Python, so this re-runs each restart — that's fine, the wheels are cached on the node. |
     | `;` | Run the next command after the previous one finishes, regardless of exit status. |
     | `jupyter-lab` | Start JupyterLab as the long-running foreground process. |
     | `--ip=0.0.0.0` | Bind to all interfaces so RunAI's proxy can reach the server from outside the pod. The default (`localhost`) only accepts connections from inside the container. |
     | `--allow-root` | The container runs as root by default; Jupyter refuses to start as root unless you say it's fine. |
     | `--ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}` | RunAI proxies your notebook at a path like `/<project>/<workload-name>/...`. Jupyter has to know its own base path or static asset URLs and websocket reconnects break. The two env vars are auto-set by RunAI inside the pod. |
     | `--ServerApp.token=''` | Disable Jupyter's own login token — RunAI's portal already authenticated you, and a token here would just block the proxy. |
     | `--ServerApp.allow_origin='*'` | Allow cross-origin requests. RunAI's proxy and Jupyter end up on different origins; without this, the browser blocks the websocket. |
     | `--notebook-dir=/work` | Open Jupyter's file browser at `/work` so you land directly on the persistent volume (where the cloned repo lives). |
     
   - **Environment variables:**

     | Name | Value | Why |
     |------|-------|-----|
     | `HF_HOME` | `/models/.cache/huggingface` | HuggingFace cache root. Default is `~/.cache/huggingface` inside the pod's ephemeral disk; pointing it at the mounted `shared-models` volume is what makes `transformers.from_pretrained(...)` find the pre-cached weights. |
     | `HF_HUB_CACHE` | `/models/.cache/huggingface` | More specific override for the hub-cache path used by `huggingface_hub`. Different transformers versions respect different vars; setting both `HF_HOME` and `HF_HUB_CACHE` is belt-and-suspenders so every code path lands at the same directory. |
     | `HF_HUB_OFFLINE` | `1` | Forbid network downloads. If the model isn't in the cache, you get a fast, loud error instead of a silent multi-GB download to ephemeral disk that vanishes on restart. |

          
   - **Set the container's working directory:** *(leave empty)* 
     
8. **Compute resources:**
   - **GPU devices:** `1`
   - **GPU fractioning:** Enabled — `25%` (≈20 GB on an 80 GB H100,
     enough to load Qwen2.5-7B in bf16)
9. **Data & storage:**
   - **+ Data Volume** > pick `shared-models`, mount path `/models`,
     read-only.
   - **+ Volume** > `local-path`, container path `/work`, persistent.
     This is where you'll save notebooks. The repo itself lives in
     ephemeral `/tmp` and is symlinked into `/work/repo` by the
     runtime args — without `/work` your own notebook files would
     also vanish on restart.
10. **CREATE WORKSPACE**.

Wait for the status to flip to `Running`. With image caching this is
~30 seconds; cold-pulling the PyTorch image is ~3 minutes the first
time.

## Step B. Open Jupyter

1. Click the workspace name, then click the **Jupyter** tool link.
2. In the file browser you should see `repo/` (the symlink to the
   unpacked tarball) alongside whatever notebooks you'll create. If
   `repo/` is missing, check the workload's **Logs** tab for `curl`
   or `tar` errors (most likely a network issue reaching GitHub).

Navigate into `/work/repo/` —
you'll see all the docs and code from the repo. Notebooks under
`ocr_app/notebooks/` and `rag_app/` will run from here once their
Data Sources are attached, but that's the job of those apps' own
deployment guides.

> **Getting your own data into the workspace.** For small,
> ad-hoc files there are two easy routes that don't require any
> Data Source setup:
>
> 1. **Drag and drop** files from your laptop's file browser
>    straight into the Jupyter file browser pane (drop them anywhere
>    under `/work/`). They land on the persistent volume and survive
>    Stop/Start.
> 2. **Pull from a URL** in a Jupyter terminal:
>    ```
>    cd /work
>    curl -LO https://example.org/some-dataset.tar.gz
>    # or wget, git clone, aws s3 cp, rclone copy, ...
>    ```
>
> Browser uploads start choking around a few GB; for anything
> bigger, or for a dataset you need shared across multiple
> workloads, use a Data Source — see [04 Storage](04-storage.md).
>
> **Heads up — what happens if you delete the workspace.** The
> `/work` volume here was created inline as part of *this* workspace,
> so when the workspace gets deleted the volume goes with it (and
> anything you dropped into `/work` is gone). To make data survive
> workspace deletion — i.e., live independently and be reusable from
> a future workspace — you have to register it as a standalone
> **Data Source** in your project. [04 Storage](04-storage.md)
> walks through that exact promotion: workspace creates a PVC,
> register it as a Data Source, optionally promote to a cluster-wide
> Data Volume.

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
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda:0")

out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> Recent `transformers` versions return a `BatchEncoding` (a dict
> with `input_ids` and `attention_mask`) from `apply_chat_template`,
> not a plain tensor — so `return_dict=True` makes that explicit, and
> we unpack with `**inputs` into `model.generate`. The decode line
> indexes into `inputs["input_ids"]` for the prompt length so we
> only print the newly generated tokens.

If you get a coherent answer, everything is working: the workspace can
schedule on a GPU, the shared-models Data Volume mounted correctly,
and offline-mode reads succeed. You're ready for the actual app
deployments.

## Step D. Stop the workspace

GPUs are scarce. When you're not using the workspace, **Stop** it
from the RunAI UI — your `/work` volume and any notebooks you saved
there survive stop/start cycles, so you can pick up where you left
off later. The repo at `/work/repo` re-pulls from GitHub on every
start (it lives in ephemeral `/tmp` under the symlink), so it always
reflects the latest `main`.

> **Stop ≠ Delete.** If you **Delete** the workspace, the inline
> `/work` volume goes with it and your notebooks are gone. To keep
> work across a delete-and-recreate cycle, move it into a standalone
> PVC Data Source first — see [04 Storage](04-storage.md).

Don't leave the workspace running idle either — Stop is reversible,
the GPU goes back into the project's pool for others to use, and
your data is still there when you Start again.

## Step E. Try repeating the above steps with your own GitHub code repository.

The whole walkthrough hinges on one URL — the
`curl -sL https://github.com/qualiaMachine/RunAI_apps/.../main.tar.gz`
in your runtime args. Swap that for any GitHub repo of your own and
the rest of the workspace config is unchanged: `/work` still
persists your notebooks, `shared-models` still mounts the same
weights, the `bash -c` boilerplate still glues Jupyter into the
RunAI proxy.

A good way to internalize the pattern:

1. **Stop and Edit** your existing workspace.
2. Replace the GitHub URL in the runtime args with one of your own
   repos (use the `archive/refs/heads/<branch>.tar.gz` form). Update
   the `mv /tmp/<repo>-<branch> /tmp/<repo>` rename to match the
   tarball's top-level directory, and the `ln -sf` target to a
   matching name under `/work/`.
3. **Start** the workspace and confirm `/work/<your-repo>` shows up
   in Jupyter.

Once you've done that successfully, you can build any new workload
by editing `runtime args + image + GPU + Data Sources` and leaving
the rest of the boilerplate alone. That's the whole job.

## Next

This works fine for one researcher poking at a model. But that GPU
fraction is pinned to your workspace for as long as it's running,
and any colleague who wants to use the same model has to load their
own copy onto another GPU fraction. Doesn't scale past you.

The fix is to host the model **once** as an autoscaling Inference
workload that other notebooks call over HTTP — your workspace can
then drop to **0 GPU** entirely. Head to
[03 Share a Model as a vLLM Endpoint](03-share-as-endpoint.md) to
convert this exact Qwen2.5-7B example into that pattern.
