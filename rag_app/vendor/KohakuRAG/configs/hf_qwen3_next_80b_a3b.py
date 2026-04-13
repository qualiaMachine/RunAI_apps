"""
WattBot Evaluation Config - Qwen3-Next 80B-A3B Instruct (Local HF, MoE)

Qwen3-Next's high-sparsity Mixture-of-Experts model: 80B total parameters
with only ~3B active per token. Features hybrid attention (Gated DeltaNet +
Gated Attention) and 512 experts (10 activated + 1 shared per token).
At 4-bit NF4 quantization it needs ~40GB VRAM (fits on 2x RTX Pro 6000).
Performs on par with Qwen3-235B-A22B on many benchmarks.

Apache 2.0 license, ungated on HuggingFace.

NOTE: Requires transformers from main branch:
    pip install git+https://github.com/huggingface/transformers.git@main

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen3_next_80b_a3b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen3_next_80b_a3b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen3-Next 80B-A3B Instruct (local MoE)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen3-Next-80B-A3B-Instruct"
hf_max_new_tokens = 512
hf_temperature = 0.2

# Embedding settings (Jina v4 - must match index)
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

# Retrieval settings
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C→Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 2
max_concurrent = 1
