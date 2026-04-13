"""
WattBot Evaluation Config - Meta Llama 3.3 70B via AWS Bedrock

Larger model for comparison against smaller Llama 4 variants.

Usage:
    python scripts/run_experiment.py --config vendor/KohakuRAG/configs/bedrock_llama3_70b.py
"""

# Database settings
db = "../../data/embeddings/wattbot_titan_v2.db"
table_prefix = "wattbot_tv2"

# Input/output
questions = "../../data/train_QA.csv"
output = "../../artifacts/submission_bedrock_llama3_70b.csv"
metadata = "../../data/metadata.csv"

# LLM settings
llm_provider = "bedrock"
bedrock_model = "us.meta.llama3-3-70b-instruct-v1:0"
bedrock_region = "us-east-2"
# bedrock_profile — set via --profile CLI arg or AWS_PROFILE env var

# Embedding settings — must match the index that was used to build the DB.
# Option A: Local Jina V4 (requires torch + transformers — use on GPU servers)
#   embedding_model = "jinav4"
# Option B: Bedrock Titan V2 (torch-free — use on laptops with a Titan V2 index)
#   embedding_model = "bedrock"
embedding_model = "bedrock"
embedding_dim = 1024
bedrock_embedding_model = "amazon.titan-embed-text-v2:0"

# Retrieval settings (matched to local configs for fair comparison)
top_k = 8
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
use_reordered_prompt = True  # C->Q: place context before question

# Unanswerable detection
retrieval_threshold = 0.25

# Other settings
max_retries = 3
max_concurrent = 1
