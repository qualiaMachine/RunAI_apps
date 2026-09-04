# Submitting Workloads via the RunAI CLI

The rest of these docs are UI-first on purpose — the `runai` CLI isn't
installed on every cluster this repo targets, and the web UI is the
supported path. This guide covers the CLI as an **alternative** for
scriptable, repeatable submissions. Everything here was verified against
the DoIT AI cluster (`doit-ai-cluster`, control plane at
`deepthought.doit.wisc.edu`) with CLI v2.

When the CLI is worth it:

- Submitting the same workload shape repeatedly with small variations
  (new model, different GPU fraction) — a shell command beats 10 UI
  screens
- Quick inventory/status checks (`inference list`) without opening a
  browser
- A committed, reviewable record of exactly how a workload was
  launched (this doc's submit blocks double as that record)

The UI remains better for: templates, one-off exploration, and anything
where you want to see the form's current field set.

## Install (Windows / Git Bash)

1. In the RunAI web UI, click the **Help (?)** icon > **Researcher
   Command Line Interface**
2. Select the cluster, then your OS. Check your architecture first if
   unsure: `uname -m` in Git Bash — `x86_64` = amd64 (virtually all
   UW-issued machines), `aarch64` = arm64
3. Download the binary. **It ships without a file extension** (e.g.
   `runai-cli-amd64`) — rename it per the panel's instructions:

```bash
mkdir -p ~/runai && cd ~/runai
mv ~/Downloads/runai-cli-amd64 ./runai-cli-amd64.exe
./runai-cli-amd64.exe version
```

(Optionally move it to a folder on `PATH` and call it `runai`; running
`./runai-cli-amd64.exe` from its folder works fine.)

Linux/Mac: the panel gives a one-line installer command instead —
copy-paste it and skip the rename.

## Configure (one-time)

```bash
# Windows only — Linux/Mac installers set this automatically.
# The control-plane URL is the RunAI web console's own address
# (scheme + host from your browser's address bar, no path).
./runai-cli-amd64.exe config set --cp-url https://deepthought.doit.wisc.edu

# Browser SSO round-trip (NetID). Must be on campus network/VPN.
./runai-cli-amd64.exe login

# Default project, so submits don't need -p every time
./runai-cli-amd64.exe project set shared-models

# Sanity checks
./runai-cli-amd64.exe project list     # shows projects + GPU quota/allocation
./runai-cli-amd64.exe workload list    # everything in the default project
```

## Everyday commands

```bash
./runai-cli-amd64.exe inference list             # serving endpoints + status + GPU
./runai-cli-amd64.exe workload list              # inference + workspaces + trainings
./runai-cli-amd64.exe inference describe <name>  # one workload's full detail
./runai-cli-amd64.exe inference submit --help    # ALWAYS check before submitting —
                                                 # flag names drift between CLI versions
```

## Check quota before submitting

`project list` shows **GPU Quota** vs **Allocated GPUs** per project.
Inference workloads here run non-preemptible: a submission that would
push allocation past quota is refused or left pending. Add up your
`--gpu-portion-request` against the headroom first, and either shrink
the request or raise the project quota (org admin) before submitting.

## Submission examples (verified on this cluster)

Both follow the standard pattern from
[Share a model as a vLLM endpoint](03-share-as-endpoint.md): stock vLLM
image, model weights from the shared PVC (`HF_HUB_OFFLINE=1` makes a
missing model fail loudly instead of downloading to ephemeral disk),
public HTTP serving endpoint on 8000. Download the model to the PVC
first (see `rag_app/docs/managing-models.md`).

**Embedding server** (OpenAI-compatible `/v1/embeddings`):

```bash
./runai-cli-amd64.exe inference submit qwen3-embedding-8b \
  -i vllm/vllm-openai:latest --image-pull-policy IfNotPresent \
  --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.25 \
  --existing-pvc=claimname=shared-model-repository-project-3w4iu,path=/models \
  --serving-port=container=8000,protocol=http \
  -e HF_HOME=/models/.cache/huggingface -e HF_HUB_CACHE=/models/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -- Qwen/Qwen3-Embedding-8B --task embed
```

**Reranker** (vLLM score task, Jina-compatible `/rerank`):

```bash
./runai-cli-amd64.exe inference submit bge-reranker-v2-m3 \
  -i vllm/vllm-openai:latest --image-pull-policy IfNotPresent \
  --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.10 \
  --existing-pvc=claimname=shared-model-repository-project-3w4iu,path=/models \
  --serving-port=container=8000,protocol=http \
  -e HF_HOME=/models/.cache/huggingface -e HF_HUB_CACHE=/models/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -- BAAI/bge-reranker-v2-m3 --task score
```

Anatomy, piece by piece:

| Chunk | Why |
|-------|-----|
| positional name (`qwen3-embedding-8b`) | Workload name — also becomes the endpoint URL: `https://<name>-runai-<project>.deepthought.doit.wisc.edu` |
| `--gpu-request-type portion --gpu-portion-request N` | Fractional GPU. Size to model weights + headroom (0.10 ≈ 9.6 GB of a 96 GB RTX Pro 6000; `provision_shared_models.py vram <model>` prints the sizing table) |
| `--existing-pvc=claimname=…,path=/models` | Shared model weights. Claim name is per-project — this is `shared-models`'s claim |
| `--serving-port=container=8000,protocol=http` | Must match the port the server listens on (vLLM default 8000) |
| `-- <model> --task …` | Everything after `--` goes to the image's `vllm serve` entrypoint |
| *(no `--priority`)* | Defaults apply; add `--priority very-high` for standing services per house convention |

Watch it come up, then smoke-test:

```bash
./runai-cli-amd64.exe inference list    # Initializing -> Running

curl -s https://qwen3-embedding-8b-runai-shared-models.deepthought.doit.wisc.edu/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-8B","input":"hello world"}' | head -c 200

curl -s https://bge-reranker-v2-m3-runai-shared-models.deepthought.doit.wisc.edu/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-reranker-v2-m3","query":"what is a GPU?","documents":["GPUs accelerate computation","bananas are yellow"]}'
```

## Gotchas learned the hard way

- **Don't request CPU or memory.** `--cpu-core-request` /
  `--cpu-memory-request` were accepted in Aug 2026 and are now rejected
  at admission: the Knative revision fails in ~5 seconds, no pod is
  ever scheduled, and there is nothing in the logs to explain it. Omit
  both and let cluster defaults apply. Worth checking first whenever a
  submit dies suspiciously fast.
- **Flag drift:** these commands are validated against the CLI version
  current as of Aug 2026. Run `inference submit --help` after any CLI
  update and re-check.
- **`--task embed` / `--task score`:** vLLM's task flags have been
  renamed before (`--runner pooling` in newer versions). If the
  workload dies at startup with an unrecognized-argument error, this is
  the first suspect — check the workload log.
- **Failed containers show no logs in the UI** if they crash-loop
  faster than the log viewer attaches. For debugging, append
  `|| sleep 3600` inside a `bash -c` command so the container stays up
  with the error readable — then remove it once healthy. (Not needed
  for plain vLLM args-only submissions, which log before dying.)
- **The workload name is the URL.** Rename = new URL = every client
  breaks. Choose names you can live with.
