"""
WattBot Evaluation Config - Qwen 1.5 110B Chat (Local HF)

Qwen 1.5 110B is the largest dense model in the Qwen 1.5 family (110B
parameters). It is the predecessor to Qwen 2.5 and remains one of the
few open-weight dense models above 100B.
With 4-bit NF4 quantization it needs ~60GB VRAM.
device_map="auto" handles multi-GPU sharding automatically.

Tongyi Qianwen license (free for <100M MAU), ungated on HuggingFace.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_qwen1_5_110b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_qwen1_5_110b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - Qwen 1.5 110B Chat (local)
llm_provider = "hf_local"
hf_model_id = "Qwen/Qwen1.5-110B-Chat"
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
use_reordered_prompt = True  # C->Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 2
max_concurrent = 1  # Very large model - conservative concurrency
