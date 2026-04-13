"""Build index with Amazon Titan Text Embeddings V2 via Bedrock.

This produces a torch-free index that can be queried on a laptop using
only Bedrock API calls (no GPU, no torch).

Prerequisites:
    - AWS SSO session active: aws sso login --profile <name>
    - Corpus JSON files in data/corpus/ (or metadata.csv for auto-fetch)

Usage (from repo root):
    python vendor/KohakuRAG/scripts/wattbot_build_index.py \
        --config vendor/KohakuRAG/configs/bedrock_titan_v2/index.py

The resulting DB should be uploaded to S3 so laptop users can download it:
    aws s3 cp data/embeddings/wattbot_titan_v2.db \
        s3://wattbot-nils-kohakurag/indexes/wattbot_titan_v2.db \
        --profile <name>
"""

# Document and database settings (paths relative to repo root)
metadata = "data/metadata.csv"
docs_dir = "data/corpus"
db = "data/embeddings/wattbot_titan_v2.db"
table_prefix = "wattbot_tv2"
use_citations = False

# PDF fetching settings
pdf_dir = "data/pdfs"

# Bedrock Titan V2 embedding settings
embedding_model = "bedrock"
embedding_dim = 1024  # Titan V2 supports: 256, 384, 1024
bedrock_profile = None  # Override via AWS_PROFILE env var or --profile
bedrock_region = "us-east-2"

# Paragraph embedding mode
paragraph_embedding_mode = "averaged"
