# Working in this repo

## Docs must track code

Whenever code changes in this repo, check whether docs still describe
it correctly. If they don't, update them in the same change — don't
leave drift for later.

High-signal pairings to double-check:

| If you touched... | Re-read these docs |
|-------------------|--------------------|
| `ocr_app/app.py`, `ocr_app/scripts/ocr_server.py`, `ocr_app/scripts/batch_extract.py` | `ocr_app/README.md`, `ocr_app/.env.example` (per-page path is not deployment-tested — no `docs/*.md` describe it) |
| `ocr_app/scripts/chunk_extract.py`, `merge.py`, `doc_prompt.py` or either notebook under `ocr_app/notebooks/` | `ocr_app/README.md`, `ocr_app/docs/setup-workspace.md`, `ocr_app/docs/deploy-vllm.md` |
| Startup args in any RunAI arguments block (image URL, pip install list, env vars, ports) | The matching `ocr_app/docs/*.md` and `rag_app/docs/*.md` step |
| Model name defaults (`VLM_MODEL`, `LLM_MODEL`) | The relevant `docs/*.md` env-var table and `.env.example` |
| `rag_app/app.py`, `rag_app/scripts/*` | `rag_app/README.md` and matching `rag_app/docs/*.md` |
| `scripts/*` at the repo root | Top-level `README.md` "Shared utilities" section |
| `scripts/fastapi_example.py` | `docs/06-fastapi-app.md` (Step B arguments block, env-var table, endpoint table) |

Anything in `rag_app/vendor/` is upstream — leave vendor docs alone.

## Deployment is UI-first

RunAI deployment is driven through the web UI, not the CLI. Docs should
prefer "click X > Y > Z" instructions over `runai submit` / `runai logs`
commands. The `runai` CLI isn't installed on every cluster this repo
targets.

## No yaml references yet

`deploy/runai_jobs.yaml` files exist as notes to self but are not the
supported deployment path yet. Don't reference them in user-facing docs.
