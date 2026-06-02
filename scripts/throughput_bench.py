#!/usr/bin/env python3
"""Quick throughput benchmark for vLLM endpoints.

Usage
-----
# Compare two models served on the RunAI cluster (VPN required):
python scripts/throughput_bench.py \
    --url1 https://qwen3--vl--32b--instruct-awq-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --url2 https://qwen3--vl--32b--instruct-fp8-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --max-tokens 256 --runs 3

# Single model quick test:
python scripts/throughput_bench.py \
    --url1 https://qwen3--vl--32b--instruct-awq-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --max-tokens 256 --runs 3

# Also works with localhost when running vLLM locally:
python scripts/throughput_bench.py \
    --url1 http://localhost:8000/v1 --max-tokens 256 --runs 3

Model names are auto-detected from the /models endpoint. No API key required
for RunAI Knative endpoints — any string works.
"""

import argparse
import time

from openai import OpenAI

# ── helpers ──────────────────────────────────────────────────────────────────

def detect_model(client: OpenAI) -> str:
    models = client.models.list()
    return models.data[0].id if models.data else ""


def bench_one(client: OpenAI, model: str, prompt: str, max_tokens: int,
              runs: int) -> dict:
    """Run *runs* completions and return timing stats."""
    results = []
    for i in range(runs):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.perf_counter() - t0

        usage = resp.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        results.append({
            "run": i + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 3),
            "tok_per_s": round(completion_tokens / elapsed, 1),
            "ttft_estimate_s": round(elapsed - (completion_tokens / (completion_tokens / elapsed)) if completion_tokens else elapsed, 3),
        })
        print(f"  run {i+1}/{runs}: {completion_tokens} tokens in {elapsed:.2f}s "
              f"→ {completion_tokens/elapsed:.1f} tok/s")

    avg_tok_s = sum(r["tok_per_s"] for r in results) / len(results)
    avg_elapsed = sum(r["elapsed_s"] for r in results) / len(results)
    return {
        "model": model,
        "runs": results,
        "avg_tok_per_s": round(avg_tok_s, 1),
        "avg_elapsed_s": round(avg_elapsed, 3),
    }


BENCH_PROMPT = (
    "Write a detailed explanation of how transformer attention mechanisms work, "
    "including multi-head attention, scaled dot-product attention, and the role "
    "of query, key, and value matrices. Be thorough."
)


def main():
    ap = argparse.ArgumentParser(description="vLLM throughput benchmark")
    ap.add_argument("--url1", required=True, help="Base URL for model 1 (e.g. http://localhost:8000/v1)")
    ap.add_argument("--model1", default=None, help="Model name (auto-detected if omitted)")
    ap.add_argument("--url2", default=None, help="Base URL for model 2")
    ap.add_argument("--model2", default=None, help="Model 2 name (auto-detected if omitted)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt", default=BENCH_PROMPT)
    args = ap.parse_args()

    # ── Model 1 ──────────────────────────────────────────────────────────────
    client1 = OpenAI(base_url=args.url1, api_key="unused")
    model1 = args.model1 or detect_model(client1)
    print(f"\n{'='*60}")
    print(f"Model 1: {model1}")
    print(f"URL:     {args.url1}")
    print(f"{'='*60}")
    stats1 = bench_one(client1, model1, args.prompt, args.max_tokens, args.runs)

    # ── Model 2 (optional) ──────────────────────────────────────────────────
    stats2 = None
    if args.url2:
        client2 = OpenAI(base_url=args.url2, api_key="unused")
        model2 = args.model2 or detect_model(client2)
        print(f"\n{'='*60}")
        print(f"Model 2: {model2}")
        print(f"URL:     {args.url2}")
        print(f"{'='*60}")
        stats2 = bench_one(client2, model2, args.prompt, args.max_tokens, args.runs)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<50} {'Avg tok/s':>10} {'Avg time':>10}")
    print(f"  {'-'*50} {'-'*10} {'-'*10}")
    print(f"  {stats1['model']:<50} {stats1['avg_tok_per_s']:>10.1f} {stats1['avg_elapsed_s']:>9.3f}s")
    if stats2:
        print(f"  {stats2['model']:<50} {stats2['avg_tok_per_s']:>10.1f} {stats2['avg_elapsed_s']:>9.3f}s")
        speedup = stats2["avg_tok_per_s"] / stats1["avg_tok_per_s"] if stats1["avg_tok_per_s"] else 0
        if speedup > 1:
            print(f"\n  → Model 2 is {speedup:.2f}x faster")
        elif speedup > 0:
            print(f"\n  → Model 1 is {1/speedup:.2f}x faster")


if __name__ == "__main__":
    main()
