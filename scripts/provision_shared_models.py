#!/usr/bin/env python3
"""Provision models on the shared PVC for RunAI inference jobs.

Run this from a workspace with WRITE access to /models (typically the
workspace that created the shared-models data source).

Usage:
    python scripts/provision_shared_models.py list
    python scripts/provision_shared_models.py download jinaai/jina-embeddings-v4
    python scripts/provision_shared_models.py download jinaai/jina-embeddings-v4 --include "adapters/*"
    python scripts/provision_shared_models.py verify jinaai/jina-embeddings-v4
"""

import argparse
import os
import sys
from pathlib import Path

PVC_CACHE = os.environ.get("SHARED_MODELS_PATH", "/models/.cache/huggingface")


def cmd_list(args):
    """List models currently cached on the shared PVC."""
    if not os.path.isdir(PVC_CACHE):
        print(f"ERROR: {PVC_CACHE} does not exist. Is the PVC mounted?")
        sys.exit(1)

    # Check if writable
    test_file = os.path.join(PVC_CACHE, ".write_test")
    writable = False
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        writable = True
    except OSError:
        pass
    print(f"PVC path: {PVC_CACHE} ({'writable' if writable else 'READ-ONLY'})")
    print()

    models = sorted(
        e for e in os.listdir(PVC_CACHE)
        if e.startswith("models--") and os.path.isdir(os.path.join(PVC_CACHE, e))
    )

    if not models:
        print("No models found.")
        return

    for model_dir_name in models:
        model_name = model_dir_name.replace("models--", "").replace("--", "/")
        model_path = os.path.join(PVC_CACHE, model_dir_name)

        # Count snapshots
        snap_dir = os.path.join(model_path, "snapshots")
        snapshots = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []

        # Compute size
        total_bytes = 0
        for root, _dirs, files in os.walk(model_path):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp):
                    total_bytes += os.path.getsize(fp)

        size_gb = total_bytes / (1024 ** 3)

        # Check for adapters in latest snapshot
        has_adapters = False
        if snapshots:
            latest = os.path.join(snap_dir, snapshots[-1])
            has_adapters = os.path.isdir(os.path.join(latest, "adapters"))

        adapters_str = " [+adapters]" if has_adapters else ""
        print(f"  {model_name} ({size_gb:.1f} GB, {len(snapshots)} snapshot(s)){adapters_str}")

        if snapshots:
            latest = os.path.join(snap_dir, snapshots[-1])
            files = sorted(os.listdir(latest))
            # Show key files
            safetensors = [f for f in files if f.endswith(".safetensors")]
            other = [f for f in files if not f.endswith(".safetensors")]
            if safetensors:
                print(f"    weights: {len(safetensors)} safetensors file(s)")
            dirs = [f for f in other if os.path.isdir(os.path.join(latest, f))]
            if dirs:
                print(f"    dirs: {', '.join(dirs)}")


def cmd_download(args):
    """Download a model to the shared PVC."""
    # Verify PVC is writable
    test_file = os.path.join(PVC_CACHE, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError:
        print(f"ERROR: {PVC_CACHE} is read-only.")
        print("You need write access to the PVC. Use the workspace that")
        print("created the shared-models data source.")
        sys.exit(1)

    os.environ["HF_HOME"] = PVC_CACHE
    os.environ["HF_HUB_CACHE"] = PVC_CACHE

    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": args.model,
        "cache_dir": PVC_CACHE,
    }
    if args.include:
        kwargs["allow_patterns"] = args.include
    if args.revision:
        kwargs["revision"] = args.revision
    if args.token:
        kwargs["token"] = args.token

    print(f"Downloading {args.model} to {PVC_CACHE}...")
    if args.include:
        print(f"  Include patterns: {args.include}")

    path = snapshot_download(**kwargs)
    print(f"Downloaded to: {path}")

    # Sizing report so the GPU fraction / --max-model-len can be chosen
    # right away, while you're looking at this terminal.
    print()
    print_vram_report(args.model)


def _latest_snapshot(model_id):
    """Return (snapshot_dir, snapshot_name) for a cached model, or exit."""
    model_dir = os.path.join(PVC_CACHE, f"models--{model_id.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_dir):
        print(f"ERROR: {model_id} not found at {model_dir}")
        sys.exit(1)
    snapshots = sorted(os.listdir(snap_dir))
    if not snapshots:
        print(f"ERROR: No snapshots found in {snap_dir}")
        sys.exit(1)
    return os.path.join(snap_dir, snapshots[-1]), snapshots[-1]


def print_vram_report(model_id, gpu_vram=96.0):
    """Estimate serving VRAM: weights (exact, from file sizes) + KV-cache
    guidance for sizing --max-model-len and the GPU fraction.

    Works without a GPU (e.g. in the provisioning workspace): weights VRAM
    equals the weight-file bytes on disk, and KV cost per token comes from
    config.json. If nvidia-smi is available, live GPU usage is shown too.
    """
    import json
    import subprocess

    snapshot, snap_name = _latest_snapshot(model_id)

    # ── Weights: sum of weight-file bytes (== VRAM to hold them) ──
    weight_bytes = 0
    for root, _dirs, files in os.walk(snapshot):
        for f in files:
            if f.endswith((".safetensors", ".bin")):
                weight_bytes += os.path.getsize(os.path.join(root, f))
    weights_gib = weight_bytes / 1024**3

    print(f"Model:    {model_id}  (snapshot {snap_name})")
    print(f"Weights:  {weights_gib:.1f} GiB on disk == VRAM to load them")

    # ── KV cache per token, from config.json ──
    cfg_path = os.path.join(snapshot, "config.json")
    kv_per_token = None
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # VL models nest the LM config under text_config
        cfg = cfg.get("text_config", cfg)
        try:
            layers = cfg["num_hidden_layers"]
            n_heads = cfg["num_attention_heads"]
            kv_heads = cfg.get("num_key_value_heads", n_heads)
            head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
            # K + V, bf16/fp16 (2 bytes). --kv-cache-dtype fp8 halves this.
            kv_per_token = 2 * layers * kv_heads * head_dim * 2
        except KeyError:
            pass
        quant = cfg.get("quantization_config", {}).get("quant_method")
        if quant:
            print(f"Note:     pre-quantized ({quant}) — disk size already reflects it")

    if kv_per_token is None:
        print("KV cache: config.json missing attention fields — no estimate")
        print("          (embedding/reranker-style models have no meaningful KV cache)")
    else:
        print(f"KV cache: {kv_per_token / 1024:.1f} KiB per token "
              f"({kv_per_token * 10_000 / 1024**3:.2f} GiB per 10k tokens, bf16)")
        print()
        print(f"Token budget by GPU fraction ({gpu_vram:.0f} GiB card, ~90% usable by")
        print("vLLM, minus weights and ~1.5 GiB CUDA overhead). Total tokens bound")
        print("--max-model-len x concurrent sequences. NOTE: inside a fractional")
        print("container, nvidia-smi reports the fraction as the device total.")
        for frac in (0.10, 0.15, 0.25, 0.50, 0.75, 1.00):
            budget = frac * gpu_vram * 0.90 - weights_gib - 1.5
            if budget <= 0:
                print(f"  {frac:.2f} GPU: weights don't fit")
            else:
                tokens = int(budget * 1024**3 / kv_per_token)
                print(f"  {frac:.2f} GPU: ~{budget:.1f} GiB free -> ~{tokens:,} tokens")
        print()
        print("Rules of thumb: if vLLM dies with 'KV cache ... is too small',")
        print("lower --max-model-len or raise the fraction; --kv-cache-dtype")
        print("fp8_e4m3 roughly doubles the token budget.")

    # ── Live reading, when a GPU exists (not in the provisioning workspace) ──
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"\nnvidia-smi (this container): {out.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\n(no GPU in this container — figures above are computed from files)")


def cmd_verify(args):
    """Verify a model has all expected files on the PVC."""
    model_dir_name = f"models--{args.model.replace('/', '--')}"
    model_path = os.path.join(PVC_CACHE, model_dir_name)

    if not os.path.isdir(model_path):
        print(f"ERROR: {args.model} not found at {model_path}")
        sys.exit(1)

    snap_dir = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snap_dir):
        print(f"ERROR: No snapshots directory at {snap_dir}")
        sys.exit(1)

    snapshots = sorted(os.listdir(snap_dir))
    if not snapshots:
        print(f"ERROR: No snapshots found in {snap_dir}")
        sys.exit(1)

    latest = os.path.join(snap_dir, snapshots[-1])
    files = sorted(os.listdir(latest))

    print(f"Model: {args.model}")
    print(f"Snapshot: {snapshots[-1]}")
    print(f"Files ({len(files)}):")

    issues = []

    # Check for essential files
    has_config = "config.json" in files
    has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
    has_tokenizer = any("tokenizer" in f for f in files)
    has_adapters = os.path.isdir(os.path.join(latest, "adapters"))

    for f in files:
        fp = os.path.join(latest, f)
        if os.path.isdir(fp):
            sub_count = len(os.listdir(fp))
            print(f"  {f}/ ({sub_count} items)")
        elif os.path.islink(fp):
            print(f"  {f} -> {os.readlink(fp)}")
        else:
            size_mb = os.path.getsize(fp) / (1024 ** 2)
            print(f"  {f} ({size_mb:.1f} MB)")

    print()
    print(f"  config.json:  {'OK' if has_config else 'MISSING'}")
    print(f"  weights:      {'OK' if has_weights else 'MISSING'}")
    print(f"  tokenizer:    {'OK' if has_tokenizer else 'MISSING'}")
    print(f"  adapters/:    {'OK' if has_adapters else 'NOT PRESENT (may need download)'}")

    if not has_config:
        issues.append("Missing config.json")
    if not has_weights:
        issues.append("Missing model weights (.safetensors or .bin)")

    # Model-specific checks
    if "jina-embeddings-v4" in args.model and not has_adapters:
        issues.append(
            "Jina V4 adapters/ directory missing. "
            "Run: python scripts/provision_shared_models.py download "
            f"{args.model} --include 'adapters/*'"
        )

    if issues:
        print(f"\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print(f"\nAll checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage models on the shared RunAI PVC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List models on the shared PVC")

    dl = sub.add_parser("download", help="Download a model to the shared PVC")
    dl.add_argument("model", help="HuggingFace model ID (e.g. jinaai/jina-embeddings-v4)")
    dl.add_argument("--include", nargs="+", help="Only download files matching these patterns")
    dl.add_argument("--revision", help="Git revision (branch, tag, or commit hash)")
    dl.add_argument("--token", help="HuggingFace token (for gated models)")

    vf = sub.add_parser("verify", help="Verify a model has all required files")
    vf.add_argument("model", help="HuggingFace model ID")

    vr = sub.add_parser("vram", help="Estimate serving VRAM (weights + KV-cache guidance)")
    vr.add_argument("model", help="HuggingFace model ID")
    vr.add_argument("--gpu-vram", type=float, default=96.0,
                    help="GPU VRAM in GiB for the fraction table "
                         "(default: 96, RTX Pro 6000)")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "vram":
        print_vram_report(args.model, gpu_vram=args.gpu_vram)


if __name__ == "__main__":
    main()
