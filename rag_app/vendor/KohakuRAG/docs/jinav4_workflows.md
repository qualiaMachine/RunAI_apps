## JinaV4 Multimodal Workflows

This guide explains how to use the JinaV4 multimodal RAG workflows.

---

## Prerequisites

### 1. Install Dependencies
```bash
pip install -e .
# This installs: openrouter, torchvision, and other required packages
```

### 2. Set Environment Variables
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 3. Prepare Data
```bash
# You need:
- data/metadata.csv          # Document metadata
- data/test_Q.csv           # Questions to answer
- artifacts/docs_with_images/  # Parsed documents with images (JSONs)
- artifacts/raw_pdfs/       # Original PDFs (for caption extraction)
```

---

## Workflows

### Workflow 1: Full Pipeline (with Captioning)

**File:** `workflows/jinav4_pipeline.py`

**Steps:**
1. Add image captions (OpenRouter + Qwen3-VL-235B)
2. Build text index (JinaV4 multimodal embeddings)
3. Build image-only index (JinaV4 direct image embeddings)
4. Answer questions (OpenRouter + GPT5-nano)
5. Validate results

**Usage:**
```bash
python workflows/jinav4_pipeline.py
```

**Output:**
- Database: `artifacts/wattbot_jinav4.db`
- Predictions: `artifacts/jinav4_pipeline_preds.csv`
- Validation scores displayed

**When to use:**
- Fresh start with documents that don't have captions yet
- Want to re-caption images with better vision model
- Full end-to-end pipeline

---

### Workflow 2: Index-Only Pipeline (skip captioning)

**File:** `workflows/jinav4_pipeline_nocaption.py`

**Steps:**
1. Build text index (JinaV4 multimodal embeddings)
2. Render PDF pages as images and store in DB
3. Re-index to pick up page_image nodes with `image_storage_key`
4. Build image-only index (JinaV4 direct image embeddings)
5. Answer questions (OpenRouter + GPT5-nano)
6. Validate results

**Usage:**
```bash
python workflows/jinav4_pipeline_nocaption.py
```

**Output:**
- Database: `artifacts/wattbot_jinav4.db`
- Predictions: `artifacts/jinav4_nocap_preds.csv`

**When to use:**
- **WattBot deployment** — no vision LLM API key needed
- JinaV4 embeds page images directly (not captions)
- Want to quickly rebuild index with different JinaV4 settings
- Testing different embedding dimensions

> **Note:** This workflow uses `wattbot_store_images.py` to render full
> PDF pages, then JinaV4's `encode_image()` to embed them directly into
> the same vector space as text. No captioning step or external API
> required — the multimodal embedding handles cross-modal retrieval
> natively.

---

### Workflow 3: Ensemble Runner

**File:** `workflows/jinav4_ensemble_runner.py`

**Steps:**
1. Run multiple parallel inference jobs
2. Aggregate results using majority voting
3. Validate aggregated predictions

**Usage:**
```bash
python workflows/jinav4_ensemble_runner.py
```

**Output:**
- Individual predictions: `artifacts/ensemble/run{n}_preds.csv`
- Aggregated: `artifacts/jinav4_ensemble_preds.csv`

**Configuration options:**
- `num_runs`: Number of parallel inference runs
- `ref_mode`: Aggregation mode (`union`, `intersection`, etc.)
- `ignore_blank`: Filter out `is_blank` before voting

---

## Configuration Options

### Embedding Settings

**Matryoshka Dimensions:**
```python
embedding_dim = 1024  # Options: 128, 256, 512, 1024, 2048
```

Smaller = faster but less accurate
Larger = slower but more accurate

**Recommended:**
- Production: 1024 (good balance)
- Fast prototyping: 512
- Maximum quality: 2048

**Task Modes:**
```python
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"
```

- `"retrieval"`: Best for QA and search (recommended)
- `"text-matching"`: Best for semantic similarity
- `"code"`: Optimized for code search

### LLM Settings

**Models:**
```python
model = "openai/gpt-5-nano"  # Fast and cheap
# Or:
model = "anthropic/claude-3.5-sonnet"  # High quality
model = "google/gemini-2.0-flash-exp"  # Good balance
```

**Provider:**
```python
llm_provider = "openrouter"  # Default
# Or:
llm_provider = "openai"  # Direct OpenAI
```

### Retrieval Settings

```python
top_k = 16  # Text results per query
planner_max_queries = 3  # Number of query variations
deduplicate_retrieval = True  # Remove duplicates
rerank_strategy = "frequency"  # Rank by multi-query relevance
top_k_final = 24  # Final truncation after reranking

# Image retrieval
with_images = True
top_k_images = 4  # Images from dedicated search
```

### Aggregation Settings (for Ensemble)

```python
ref_mode = "union"  # How to aggregate ref_ids
# Options: "independent", "ref_priority", "answer_priority", "union", "intersection"

ignore_blank = False  # Filter out "is_blank" before voting
# Set True when some runs may fail due to rate limits
```

---

## Pipeline Details

There are two image processing paths. Choose based on your setup:

| Path | Requires | Image quality | Use case |
|------|----------|---------------|----------|
| **Page renders (recommended)** | GPU only | Full page images | WattBot deployment, no API key needed |
| **Extracted + captioned** | Vision LLM API | Individual figures with captions | Research, better per-figure retrieval |

---

### Stage 1a: Page Image Rendering (no API key needed)

**Script:** `scripts/wattbot_store_images.py`

**What it does:**
1. Renders each PDF page as a high-quality JPEG using PyMuPDF
2. Stores page images in the `image_blobs` table (ImageStore)
3. Adds `page_image` paragraphs with `image_storage_key` metadata to
   the corpus JSON files

**Config:** Uses the same config as the text index (`configs/jinav4/index.py`)

**Output:**
- Page images stored in `{db}::image_blobs` table
- Updated JSONs in `data/corpus/` with `page_image` paragraphs

> **Important:** After running this, you must **re-run the text indexer**
> (Stage 2) so the new `page_image` nodes with `image_storage_key`
> metadata are embedded into the vector store. Without re-indexing, the
> image index builder (Stage 3) won't find any images.

---

### Stage 1b: Image Captioning (optional, needs vision LLM)

**Script:** `scripts/wattbot_add_image_captions.py`

**What it does:**
1. Extracts individual images from PDF pages
2. Compresses images to reduce size
3. Generates captions using a vision model (e.g. Qwen3-VL)
4. Stores images in database (ImageStore) with `image_storage_key`
5. Updates JSON documents with image metadata and captions

**Config:**
```python
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
max_concurrent = 5  # Vision API concurrency
```

**Output:**
- Updated JSONs in `docs_with_images/` with image metadata
- Images stored in `{db}::image_blobs` table

> **Note:** This step is only needed if you want per-figure captions for
> text-based image search. For JinaV4 direct embedding, Stage 1a (page
> renders) is sufficient — JinaV4 understands images natively.

---

### Stage 2: Text Indexing

**Script:** `scripts/wattbot_build_index.py`

**What it does:**
1. Loads documents from `data/corpus/` (or `docs_with_images/`)
2. Builds hierarchical structure (document → section → paragraph → sentence)
3. Embeds all text nodes (including `page_image` caption nodes) using JinaV4
4. Stores in vector database

**JinaV4 Features:**
- **Unified embeddings**: Text and images in same vector space
- **Matryoshka**: Configurable dimensions (128-2048)
- **Task-specific**: Optimized for retrieval

**Config:**
```python
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"
```

**Output:**
- Vector database: `{db}::{prefix}_vec` table
- Node store: `{db}::{prefix}_nodes` table

> **Run this after Stage 1a or 1b** so nodes with `image_storage_key`
> exist in the DB for the image index builder to find.

---

### Stage 3: Image-Only Index

**Script:** `scripts/wattbot_build_image_index.py`

**What it does:**
1. Scans the node store for nodes with `attachment_type` = `"image"` or
   `"page_image"` that have `image_storage_key` metadata
2. Loads actual image bytes from ImageStore via `image_storage_key`
3. Embeds images directly using `JinaV4.encode_image()`
4. Creates a separate image-only vector table

**Config:**
```python
# JinaV4 direct embedding (recommended)
embedding_model = "jinav4"
embedding_dim = 1024  # Must match text index
embed_images_directly = True
```

**Output:**
- Image vector table: `{db}::{prefix}_images_vec`

**Benefits of JinaV4 Direct Embedding:**
- Better visual understanding (not limited by caption quality)
- Unified multimodal search (text and images share vector space)
- Can find images without good captions
- No external API or vision LLM required

**Common error — "No images loaded from ImageStore":**
This means nodes don't have `image_storage_key` metadata. Make sure you
ran Stage 1a (or 1b) **and then re-ran Stage 2** before this step.

---

### Stage 4: Question Answering

**Script:** `scripts/wattbot_answer.py`

**What it does:**
1. Loads questions from CSV
2. For each question:
   - Plans multiple query variations
   - Retrieves text snippets (top_k per query)
   - Retrieves images (top_k_images from image index)
   - Deduplicates and reranks
   - Generates structured answer using LLM
3. Saves answers to CSV

**Config:**
```python
llm_provider = "openrouter"
model = "openai/gpt-5-nano"
top_k = 16
top_k_final = 24
with_images = True
top_k_images = 4
```

---

### Stage 5: Validation

**Script:** `scripts/wattbot_validate.py`

**What it does:**
1. Compares predictions to ground truth
2. Scores: value (75%), ref (15%), NA (10%)
3. Shows per-question scores
4. Reports final WattBot score

**Output:**
```
Per-Question Scores
✓ [q001] val:1.000 ref:1.000 na:1.000 → final:1.000
✗ [q002] val:0.000 ref:1.000 na:1.000 → final:0.150

FINAL VALIDATION SUMMARY
🎯 FINAL WATTBOT SCORE: 0.9321
```

---

## Comparison: JinaV3 vs JinaV4

| Feature | JinaV3 | JinaV4 |
|---------|--------|--------|
| **Text Embedding** | ✅ | ✅ |
| **Image Embedding** | ❌ (via captions) | ✅ (direct) |
| **Multimodal** | ❌ | ✅ |
| **Matryoshka** | ❌ | ✅ (128-2048) |
| **Task Adapters** | ❌ | ✅ (retrieval/matching/code) |
| **Max Sequence** | 8192 tokens | 32,768 tokens |
| **Dimension** | 768 (fixed) | 128-2048 (flexible) |
| **Model Size** | ~560M params | ~4B params |

---

## Cost Considerations

### OpenRouter Pricing (approximate)

| Model | Cost per 1M tokens |
|-------|-------------------|
| gpt-5-nano | $0.05 |
| gpt-4o-mini | $0.15 |
| claude-3.5-sonnet | $3.00 |
| qwen3-vl-235b | $0.10 |

### Recommendations

**For Testing:**
```python
model = "openai/gpt-5-nano"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
```

**For Production:**
```python
model = "openai/gpt-4o-mini"  # Better quality, still affordable
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"  # Good vision quality
```

**For Maximum Quality:**
```python
model = "anthropic/claude-3.5-sonnet"
vision_model = "anthropic/claude-3.5-sonnet"  # Vision-capable
```

---

## Troubleshooting

### Issue: "OPENROUTER_API_KEY is required"

**Solution:**
```bash
export OPENROUTER_API_KEY="your-key-here"
# Or add to .env file
echo "OPENROUTER_API_KEY=your-key" >> .env
```

### Issue: "truncate_dim must be one of [128, 256, 512, 1024, 2048]"

**Solution:**
Update your config:
```python
embedding_dim = 1024  # Use valid Matryoshka dimension
```

### Issue: "No images loaded from ImageStore"

**Cause:** Nodes don't have `image_storage_key` metadata. This happens
when the image index builder runs before the text index has been rebuilt
after storing images.

**Solution:**
Run the 3-step image pipeline in order:
```bash
cd vendor/KohakuRAG
# 1. Store page images and update corpus JSONs
kogine run scripts/wattbot_store_images.py --config configs/jinav4/index.py
# 2. Re-index to embed page_image nodes with image_storage_key
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
# 3. Build image vector index
kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
```

### Issue: JinaV4 model loading fails

**Solution:**
Ensure you have sufficient GPU memory (~8GB for JinaV4):
```python
# Use smaller dimension to reduce memory
embedding_dim = 512  # Instead of 1024

# Or use CPU (slower)
# JinaV4 will auto-detect CPU if no GPU available
```

---

## Example: Complete Workflow Run

```bash
# Step 1: Set API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Step 2: Run full pipeline (with captioning)
python workflows/jinav4_pipeline.py

# Output:
# =====================================
# Stage 1: Image Captioning
# =====================================
# Captioned 45 images from 15 documents
#
# =====================================
# Stage 2: Text Indexing (JinaV4)
# =====================================
# Using JinaV4 embeddings (dim=1024, task=retrieval)
# Indexed 15 documents with 3,245 nodes
#
# =====================================
# Stage 3: Image Indexing (JinaV4)
# =====================================
# Using JinaV4 direct image embedding
# Inserted 45 image embeddings
#
# =====================================
# Stage 4: Question Answering
# =====================================
# Answered 274 questions
#
# =====================================
# Stage 5: Validation
# =====================================
# FINAL WATTBOT SCORE: 0.9234
```

---

## Advanced Usage

### Custom Embedding Dimension for Speed

```python
# configs/jinav4/index.py
embedding_dim = 512  # 2x faster than 1024

# configs/jinav4/answer.py
embedding_dim = 512  # MUST match index
```

### Mix and Match Providers

```python
# Use OpenRouter for main LLM, OpenAI for planner
llm_provider = "openrouter"
model = "anthropic/claude-3.5-sonnet"

# Planner can still use OpenAI if you set:
# (requires custom setup - not yet implemented)
```

### Test Single Question

```bash
kogine run scripts/wattbot_answer.py \\
    --config configs/jinav4/answer.py \\
    --override single_run_debug=True \\
    --override question_id=q001
```

---

## Performance Tips

### 1. Use Appropriate Dimensions

| Dimension | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| 128 | ⚡⚡⚡ | ⭐⭐ | Fast prototyping |
| 256 | ⚡⚡ | ⭐⭐⭐ | Balanced testing |
| 512 | ⚡⚡ | ⭐⭐⭐⭐ | Production (fast) |
| 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Production (recommended) |
| 2048 | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

### 2. Optimize Retrieval

```python
# For speed
top_k = 8
planner_max_queries = 1  # Disable multi-query
top_k_final = 10

# For quality
top_k = 20
planner_max_queries = 3
rerank_strategy = "frequency"
top_k_final = 30
```

### 3. Batch Processing

```python
# Increase concurrency for faster processing
max_concurrent = 20  # For LLM calls
```

---

## Architecture Diagram

```
┌─────────────────────────────────────┐
│  Documents (docs_with_images/)      │
│  - Text content                     │
│  - Images with captions             │
└────────────┬────────────────────────┘
             │
             ↓
    ┌────────────────────┐
    │  JinaV4 Embedder   │
    │  (Multimodal)      │
    └────────┬───────────┘
             │
      ┌──────┴──────┐
      ↓             ↓
┌─────────┐   ┌──────────────┐
│ Text    │   │ Images       │
│ Index   │   │ Index        │
│ (1024D) │   │ (1024D)      │
└────┬────┘   └──────┬───────┘
     │               │
     │    Question   │
     │       ↓       │
     │  ┌─────────┐ │
     └→ │ Planner │ ←┘
        └────┬────┘
             ↓
     ┌───────────────┐
     │ Multi-Query   │
     │ Retrieval     │
     └───────┬───────┘
             ↓
     ┌───────────────┐
     │ Dedup +       │
     │ Rerank        │
     └───────┬───────┘
             ↓
     ┌───────────────┐
     │ OpenRouter    │
     │ (GPT5-nano)   │
     └───────┬───────┘
             ↓
        ┌─────────┐
        │ Answer  │
        └─────────┘
```

---

## FAQ

**Q: Can I use JinaV4 for text-only (no images)?**

A: Yes! Just set `with_images=False`. JinaV4 works great for text-only too.

**Q: Do I need to rebuild the index if I change `top_k_final`?**

A: No! `top_k_final` is a runtime parameter, not an indexing parameter.

**Q: What's the difference between JinaV4 caption vs direct image embedding?**

A:
- **Caption**: Embed text description of image (faster, lower quality)
- **Direct**: Embed actual image pixels (slower, higher quality, better recall)

**Q: Can I mix JinaV3 and JinaV4?**

A: No. The dimensions are different (768 vs 1024), so you need to rebuild the index.

**Q: How do I know if JinaV4 is working?**

A: Check the logs:
```
Using JinaV4 embeddings (dim=1024, task=retrieval)
```

**Q: Why use OpenRouter instead of OpenAI directly?**

A:
- Access to 300+ models from one API
- Often cheaper pricing
- Automatic fallback and routing
- Support for open-source models

---

## Next Steps

After running the workflow:

1. **Check validation score:**
   ```
   FINAL WATTBOT SCORE: 0.XXXX
   ```

2. **Analyze failed questions:**
   ```bash
   kogine run scripts/wattbot_validate.py \\
       --config configs/jinav4/answer.py \\
       --override show_errors=20 \\
       --override verbose=True
   ```

3. **Tune parameters:**
   - Adjust `top_k`, `top_k_final`
   - Try different `rerank_strategy`
   - Experiment with `embedding_dim`

4. **Compare with baseline:**
   ```bash
   # Run JinaV3 baseline
   python workflows/text_pipeline.py

   # Compare scores
   ```

---

## Summary

### Quick Start — Page Renders + JinaV4 Direct Embedding (no API key)
```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
kogine run scripts/wattbot_store_images.py --config configs/jinav4/index.py
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py   # re-index
kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
```

### Full Pipeline (With Captioning — needs vision LLM API)
```bash
export OPENROUTER_API_KEY="your-key"
python workflows/jinav4_pipeline.py
```

### Custom Config
```bash
kogine run scripts/wattbot_answer.py --config configs/jinav4/answer.py
```

**Key Benefits:**
- 🎯 Multimodal search (text + images)
- ⚡ Flexible dimensions (128-2048)
- 💰 Cost-effective (GPT5-nano default)
- 🔧 Highly configurable
- 📊 Complete validation

Enjoy your multimodal RAG pipeline! 🚀
