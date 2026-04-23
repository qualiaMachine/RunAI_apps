# Troubleshooting

Common issues when deploying the OCR extraction pipeline on RunAI.

---

## vLLM won't start / OOM

- Check the **Logs** tab on the `qwen3--vl--32b--instruct-awq` workload
  in the RunAI UI
- Reduce KV cache: set `--max-model-len 4096`
- For 24GB GPUs: use `--quantization awq` and the 8B model (see the GPU
  sizing table in [deploy-vllm.md](deploy-vllm.md))
- Qwen3-VL-32B needs ~64GB in bfloat16; use the AWQ build for anything
  smaller

## Notebook can't reach vLLM

- Use FQDN: `http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1`
- Do **NOT** include a port number — Knative routes on port 80
- Test from a workspace terminal:
  `curl http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1/models`
- Short names like `qwen3--vl--32b--instruct-awq:8000` return envoy 404

## Notebook chunk extraction hits `finish_reason == "length"`

- The chunk's VLM output was truncated before the JSON was complete
- Lower `MAX_PAGES_PER_CHUNK` in the notebook (try 10–15)
- Or raise `VLLM_MAX_TOKENS` (make sure `--max-model-len` on the vLLM
  workload can accommodate input + requested output)
- `<stem>_chunks/` holds the raw per-chunk responses — inspect the
  truncated one to confirm which field was cut off

## Bad JSON output

- Re-run the chunk in isolation against a single page to narrow down
  whether the prompt or the merge step is at fault
- Check the per-chunk JSON in `<stem>_chunks/` for the specific chunk
  that produced the bad output

## Skip-existing not working

- The notebook pipeline uses per-doc `<stem>_extracted.json` markers via
  `SKIP_EXISTING = True` — delete the `_extracted.json` to force re-run
- Check file permissions on the output PVC

## Data volume not mounting

- Verify the Data Volume exists in **Data & Storage** > **Data Volumes**
- Check that the mount path matches what the notebook expects
- For read-only volumes, ensure the access mode is correct
- Check pod events: click the workload in the RunAI UI, open the
  **Events** tab

## Model not found on PVC

- The vLLM server runs with `HF_HUB_OFFLINE=1` — it won't download
  models at runtime
- Verify the model exists: from a workspace with the PVC mounted,
  `ls /models/.cache/huggingface/models--QuantTrio--Qwen3-VL-32B-Instruct-AWQ/`
- If missing, download it — see [Setup Data Volumes](setup-data-volumes.md)
