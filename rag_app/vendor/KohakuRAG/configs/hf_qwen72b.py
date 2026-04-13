"""
WattBot Evaluation Config - Qwen 2.5 72B Instruct (Local HF)

Qwen 2.5 72B is the largest dense model in the Qwen 2.5 family.
With 4-bit NF4 quantization it fits in ~40GB VRAM.
device_map="auto" handles multi-GPU sharding automatically.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen72b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen72b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 2.5 72B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-72B-Instruct"
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
max_concurrent = 1  # Very large model - conservative concurrency
