# Troubleshooting

Common issues when deploying the OCR extraction pipeline on RunAI.

---

## vLLM won't start / OOM

- Check GPU memory with `--max-model-len 4096` to reduce KV cache
- For 24GB GPUs: add `--quantization awq`
- Check logs: `runai logs qwen3--vl--32b--instruct-awq`
- Qwen3-VL-32B needs ~64GB in bfloat16 — if your GPU is smaller,
  quantization is required

## Extraction server / batch script can't reach vLLM

- Use FQDN: `http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1`
- Do **NOT** include a port number — Knative routes on port 80
- Test: `curl http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1/models`
- If using short names like `qwen3--vl--32b--instruct-awq:8000`, you'll get envoy 404

## Batch script hangs

- Check vLLM logs for errors: `runai logs qwen3--vl--32b--instruct-awq`
- Try `--concurrency 1` to isolate the issue
- Check if vLLM is OOM — reduce concurrency or `--max-model-len`
- Increase timeout if docs are very long (large scanned pages take longer)

## Bad JSON output

- Try a different format: `--format key_values` is more flexible than
  `--format award` for documents that don't match the award schema
- Use `--format text` first to see raw extraction, then pick a more
  specific format
- Check if the document type matches the format — don't use `award`
  for general correspondence

## Resume not working

- The state file is at `<output-dir>/.batch_state`
- It tracks completed files by their full input path
- If you moved input files, paths won't match — delete `.batch_state`
  and re-run
- Check file permissions on the output PVC

## Data volume not mounting

- Verify the Data Volume exists in **Data & Storage** > **Data Volumes**
- Check that the mount path matches what the script expects
  (`/data/documents`, `/data/extracted`)
- For read-only volumes, ensure the access mode is correct
- Check pod events: `runai describe job <name>`

## Model not found on PVC

- The vLLM server runs with `HF_HUB_OFFLINE=1` — it won't download
  models at runtime
- Verify the model exists: from a workspace with the PVC mounted,
  `ls /models/.cache/huggingface/models--QuantTrio--Qwen3-VL-32B-Instruct-AWQ/`
- If missing, download it — see [Setup Data Volumes](setup-data-volumes.md)
