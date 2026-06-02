#!/usr/bin/env python3
"""Concurrent throughput benchmark for vLLM endpoints.

Measures how throughput scales with concurrent users by sending parallel
requests at increasing concurrency levels.

Usage
-----
# Single model, sweep concurrency 1→16:
python scripts/throughput_concurrent.py \
    --url1 https://qwen3--vl--32b--instruct-awq-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --concurrency 1,2,4,8,16

# Compare two models:
python scripts/throughput_concurrent.py \
    --url1 https://qwen3--vl--32b--instruct-awq-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --url2 https://qwen3--vl--32b--instruct-8bit-runai-shared-models.deepthought.doit.wisc.edu/v1 \
    --concurrency 1,2,4,8,16

Model names are auto-detected. No API key required for RunAI Knative endpoints.
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI, OpenAI

PROMPT = (
    "Write a detailed explanation of how transformer attention mechanisms work, "
    "including multi-head attention, scaled dot-product attention, and the role "
    "of query, key, and value matrices. Be thorough."
)


def detect_model(url: str) -> str:
    client = OpenAI(base_url=url, api_key="not-used")
    models = client.models.list()
    return models.data[0].id if models.data else "unknown"


async def single_request(client: AsyncOpenAI, model: str, prompt: str,
                         max_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.perf_counter() - t0
    toks = resp.usage.completion_tokens
    return {"toks": toks, "elapsed": elapsed, "tok_s": toks / elapsed}


async def bench_concurrency(url: str, model: str, concurrency: int,
                            max_tokens: int, prompt: str) -> dict:
    client = AsyncOpenAI(base_url=url, api_key="not-used")
    tasks = [
        single_request(client, model, prompt, max_tokens)
        for _ in range(concurrency)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_start

    total_toks = sum(r["toks"] for r in results)
    avg_per_user = sum(r["tok_s"] for r in results) / len(results)
    total_tok_s = total_toks / wall_elapsed

    return {
        "concurrency": concurrency,
        "wall_time_s": round(wall_elapsed, 2),
        "total_toks": total_toks,
        "total_tok_s": round(total_tok_s, 1),
        "per_user_tok_s": round(avg_per_user, 1),
        "avg_latency_s": round(sum(r["elapsed"] for r in results) / len(results), 2),
    }


async def bench_model(url: str, model: str, concurrency_levels: list[int],
                       max_tokens: int, prompt: str) -> list[dict]:
    print(f"\nModel: {model}")
    print(f"URL:   {url}")
    print(f"{'Conc':>6} {'Total tok/s':>12} {'Per-user tok/s':>15} "
          f"{'Avg latency':>12} {'Wall time':>10}")
    print(f"{'─'*6} {'─'*12} {'─'*15} {'─'*12} {'─'*10}")

    results = []
    for c in concurrency_levels:
        r = await bench_concurrency(url, model, c, max_tokens, prompt)
        results.append(r)
        print(f"{c:>6} {r['total_tok_s']:>12.1f} {r['per_user_tok_s']:>15.1f} "
              f"{r['avg_latency_s']:>11.2f}s {r['wall_time_s']:>9.2f}s")

    return results


def print_comparison(results1: list[dict], model1: str,
                     results2: list[dict], model2: str):
    print(f"\n{'='*70}")
    print("COMPARISON: Total tok/s at each concurrency level")
    print(f"{'='*70}")
    print(f"{'Conc':>6} {model1[:25]:>28} {model2[:25]:>28}  {'Ratio':>7}")
    print(f"{'─'*6} {'─'*28} {'─'*28}  {'─'*7}")

    for r1, r2 in zip(results1, results2):
        ratio = r1["total_tok_s"] / r2["total_tok_s"] if r2["total_tok_s"] else 0
        print(f"{r1['concurrency']:>6} {r1['total_tok_s']:>27.1f} "
              f"{r2['total_tok_s']:>27.1f}  {ratio:>6.2f}x")


async def main_async(args):
    levels = [int(x) for x in args.concurrency.split(",")]
    model1 = detect_model(args.url1)

    print(f"\n{'='*70}")
    print(f"Concurrent throughput benchmark")
    print(f"Max tokens per request: {args.max_tokens}")
    print(f"Concurrency levels: {levels}")
    print(f"{'='*70}")

    results1 = await bench_model(args.url1, model1, levels, args.max_tokens,
                                  args.prompt)

    results2 = None
    if args.url2:
        model2 = detect_model(args.url2)
        results2 = await bench_model(args.url2, model2, levels,
                                      args.max_tokens, args.prompt)
        print_comparison(results1, model1, results2, model2)

    print(f"\n{'='*70}")
    print("KEY")
    print(f"{'='*70}")
    print("  Total tok/s    = all tokens generated / wall clock time (aggregate throughput)")
    print("  Per-user tok/s = avg individual stream speed (what each user experiences)")
    print("  Avg latency    = avg time to get a complete response back")


def main():
    ap = argparse.ArgumentParser(description="Concurrent vLLM throughput benchmark")
    ap.add_argument("--url1", required=True)
    ap.add_argument("--url2", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,2,4,8,16",
                    help="Comma-separated concurrency levels (default: 1,2,4,8,16)")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
