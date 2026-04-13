"""
WattBot Evaluation Config - OpenScholar 8B (Local HF)

OpenScholar 8B: Llama 3.1 8B fine-tuned for scientific literature synthesis.
Trained by UW + AI2 on 130K instances of scientific RAG data.
Excels at citation-backed responses and multi-paper synthesis.

Paper: Asai et al. (2024) "OpenScholar: Synthesizing Scientific Literature
with Retrieval-Augmented LMs" (Nature)

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/hf_openscholar_8b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_openscholar_8b.csv"
metadata = "../../data/metadata.csv"

# LLM settings - OpenScholar 8B (Llama 3.1 8B fine-tuned for scientific synthesis)
llm_provider = "hf_local"
hf_model_id = "OpenSciLM/Llama-3.1_OpenScholar-8B"
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
