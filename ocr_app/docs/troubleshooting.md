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

## Extraction server / batch script / notebook can't reach vLLM

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

## Batch script hangs

- Check the vLLM workload's **Logs** tab for errors
- Try `--concurrency 1` to isolate the issue
- Check if vLLM is OOM — reduce concurrency or `--max-model-len`
- Long scanned pages take longer; bump `max_tokens` or the client
  timeout if docs have many high-resolution pages

## Bad JSON output

- Try a different format: `--format key_values` is more flexible than
  `--format award` for documents that don't match the award schema
- Use `--format text` first to see raw extraction, then pick a more
  specific format
- Check if the document type matches the format — don't use `award`
  for general correspondence
- For the notebook pipeline, re-run the chunk in isolation against a
  single page to narrow down whether the prompt or the merge step is at
  fault

## Resume not working

- The state file is at `<output-dir>/.batch_state`
- It tracks completed files by their full input path
- If you moved input files, paths won't match — delete `.batch_state`
  and re-run
- Check file permissions on the output PVC
- The notebook pipeline uses per-doc `<stem>_extracted.json` markers via
  `SKIP_EXISTING = True` — delete the `_extracted.json` to force re-run

## Data volume not mounting

- Verify the Data Volume exists in **Data & Storage** > **Data Volumes**
- Check that the mount path matches what the script expects
  (`/data/documents`, `/data/extracted`)
- For read-only volumes, ensure the access mode is correct
- Check pod events: click the workload in the RunAI UI, open the
  **Events** tab

## Model not found on PVC

- The vLLM server runs with `HF_HUB_OFFLINE=1` — it won't download
  models at runtime
- Verify the model exists: from a workspace with the PVC mounted,
  `ls /models/.cache/huggingface/models--QuantTrio--Qwen3-VL-32B-Instruct-AWQ/`
- If missing, download it — see [Setup Data Volumes](setup-data-volumes.md)
