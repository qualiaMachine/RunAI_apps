"""Extract figures from PDFs with VLM verification using local Qwen2.5-VL.

Runs wattbot_store_images.py with VLM-based figure verification enabled.
The VLM (Qwen2.5-VL-72B, 4-bit) is loaded from the shared models PVC,
verifies each extracted crop is a real figure, and generates descriptions.

Usage:
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/store_images_vlm.py

After this, rebuild the text index and build the image index:
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
    kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
"""

from kohakuengine import Config

# Document and database settings (must match index.py)
docs_dir = "../../data/corpus"
pdf_dir = "../../data/pdfs"
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"

# VLM verification — enabled
vlm_verify = True
vlm_provider = "hf_local"
vlm_local_model = "Qwen/Qwen2.5-VL-72B-Instruct"
vlm_local_dtype = "4bit"
vlm_max_concurrent = 1  # GPU inference, keep at 1


def config_gen():
    return Config.from_globals()
