"""
WattBot Evaluation Config - Qwen 2.5 32B Instruct (Local HF)

Qwen 2.5 32B. Requires ~20GB VRAM (4-bit NF4).
device_map="auto" handles GPU sharding if needed.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen32b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen32b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 2.5 32B Instruct (local)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen2.5-32B-Instruct"
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
max_concurrent = 1  # Large model - conservative concurrency
