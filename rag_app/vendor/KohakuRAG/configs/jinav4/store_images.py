"""Extract figures from PDFs using heuristic detection (no VLM).

Usage:
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/store_images.py
"""

from kohakuengine import Config

# Document and database settings (must match index.py)
docs_dir = "../../data/corpus"
pdf_dir = "../../data/pdfs"
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# VLM verification — disabled (heuristic-only)
vlm_verify = False


def config_gen():
    return Config.from_globals()
