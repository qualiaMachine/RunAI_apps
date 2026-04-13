#!/usr/bin/env python3
"""Batch document extraction — process directories of PDFs/TIFFs at scale.

Walks an input directory, extracts structured JSON from each document using
the hybrid pipeline (text extraction for digital PDFs, VLM for scans/TIFFs),
and writes one JSON file per input document to the output directory.

Designed for high throughput:
- Concurrent LLM/VLM requests (async + semaphore)
- Text extraction is done locally (no GPU, instant)
- Resumable — tracks completed files in a state file
- Writes output incrementally (one JSON per doc as it completes)

Usage:
    # Process all PDFs in a directory
    python ocr_app/scripts/batch_extract.py \
        --input-dir /data/documents \
        --output-dir /data/extracted \
        --format award

    # Resume a failed run
    python ocr_app/scripts/batch_extract.py \
        --input-dir /data/documents \
        --output-dir /data/extracted \
        --format award \
        --resume

    # Process only TIFFs, with 8 concurrent LLM requests
    python ocr_app/scripts/batch_extract.py \
        --input-dir /data/scans \
        --output-dir /data/extracted \
        --format key_values \
        --extensions .tiff .tif \
        --concurrency 8

Environment variables:
    LLM_BASE_URL    - vLLM / Ollama endpoint (default: http://localhost:8000/v1)
    LLM_MODEL       - Model name (default: auto-detected)
    VLM_BASE_URL    - Vision model endpoint (default: same as LLM_BASE_URL)
    VLM_MODEL       - Vision model name (default: Qwen/Qwen2.5-VL-7B-Instruct)
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

# ---------------------------------------------------------------------------
# Import prompt templates from the server module
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

# Import prompts directly to avoid starting the FastAPI app
from ocr_server import TEXT_PROMPTS, VLM_PROMPTS, OutputFormat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
MIN_TEXT_LENGTH = int(os.environ.get("MIN_TEXT_LENGTH", "50"))

SUPPORTED_EXTENSIONS = {".pdf", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


# ---------------------------------------------------------------------------
# LLM / VLM calls (reused from server, but with a shared httpx client)
# ---------------------------------------------------------------------------

async def detect_model(client: httpx.AsyncClient, base_url: str) -> str:
    try:
        resp = await client.get(f"{base_url}/models")
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return ""


async def llm_parse(client: httpx.AsyncClient, text: str, prompt: str,
                    max_tokens: int, model: str) -> str:
    full_prompt = f"{prompt}\n\n---\nDOCUMENT TEXT:\n---\n{text}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = await client.post(f"{LLM_BASE_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def vlm_ocr(client: httpx.AsyncClient, image: Image.Image, prompt: str,
                  max_tokens: int, model: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    base_url = VLM_BASE_URL or LLM_BASE_URL
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = await client.post(f"{base_url}/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from all pages of a PDF. Returns list of {page, text, has_text}."""
    import fitz
    doc = fitz.open(str(pdf_path))
    results = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        results.append({
            "page": i,
            "text": text,
            "has_text": len(text) >= MIN_TEXT_LENGTH,
        })
    doc.close()
    return results


def render_pdf_page(pdf_path: Path, page_idx: int) -> Image.Image:
    """Render a PDF page to an image at 2x resolution."""
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[page_idx]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


async def process_pdf(client: httpx.AsyncClient, pdf_path: Path, fmt: OutputFormat,
                      max_tokens: int, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Process a single PDF — extract text or VLM OCR per page."""
    pages = extract_pdf_pages(pdf_path)
    text_prompt = TEXT_PROMPTS[fmt]
    vlm_prompt = VLM_PROMPTS[fmt]

    page_results = []
    digital_count = 0
    scanned_count = 0

    for page_info in pages:
        async with semaphore:
            t0 = time.time()
            if page_info["has_text"]:
                digital_count += 1
                result_text = await llm_parse(
                    client, page_info["text"], text_prompt, max_tokens, model
                )
                method = "text_extraction"
            else:
                scanned_count += 1
                img = render_pdf_page(pdf_path, page_info["page"])
                result_text = await vlm_ocr(
                    client, img, vlm_prompt, max_tokens, VLM_MODEL
                )
                method = "vlm_ocr"
            elapsed_ms = (time.time() - t0) * 1000

        page_results.append({
            "page": page_info["page"] + 1,
            "text": result_text,
            "method": method,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    return {
        "source_file": str(pdf_path),
        "format": fmt.value,
        "total_pages": len(pages),
        "digital_pages": digital_count,
        "scanned_pages": scanned_count,
        "pages": page_results,
    }


async def process_image(client: httpx.AsyncClient, img_path: Path, fmt: OutputFormat,
                        max_tokens: int, semaphore: asyncio.Semaphore) -> dict:
    """Process a single image/TIFF — always VLM OCR."""
    img = Image.open(str(img_path))
    if img.mode != "RGB":
        img = img.convert("RGB")

    vlm_prompt = VLM_PROMPTS[fmt]

    async with semaphore:
        t0 = time.time()
        result_text = await vlm_ocr(client, img, vlm_prompt, max_tokens, VLM_MODEL)
        elapsed_ms = (time.time() - t0) * 1000

    return {
        "source_file": str(img_path),
        "format": fmt.value,
        "total_pages": 1,
        "digital_pages": 0,
        "scanned_pages": 1,
        "pages": [{
            "page": 1,
            "text": result_text,
            "method": "vlm_ocr",
            "elapsed_ms": round(elapsed_ms, 2),
        }],
    }


# ---------------------------------------------------------------------------
# State tracking (for resume)
# ---------------------------------------------------------------------------

def load_completed(state_file: Path) -> set[str]:
    """Load set of completed file paths from state file."""
    if not state_file.exists():
        return set()
    with open(state_file) as f:
        return set(line.strip() for line in f if line.strip())


def mark_completed(state_file: Path, file_path: str):
    """Append a completed file path to the state file."""
    with open(state_file, "a") as f:
        f.write(file_path + "\n")


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

async def run_batch(args: argparse.Namespace):
    global LLM_MODEL

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = OutputFormat(args.format)
    extensions = set(args.extensions) if args.extensions else SUPPORTED_EXTENSIONS
    max_tokens = args.max_tokens
    concurrency = args.concurrency

    # Discover files
    files = sorted(
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        print(f"No files found in {input_dir} with extensions {extensions}")
        return

    # Resume support
    state_file = output_dir / ".batch_state"
    completed = load_completed(state_file) if args.resume else set()
    remaining = [f for f in files if str(f) not in completed]

    print(f"[batch] Found {len(files)} files, {len(completed)} already completed, "
          f"{len(remaining)} to process", flush=True)
    print(f"[batch] Output: {output_dir}", flush=True)
    print(f"[batch] Format: {fmt.value}, Concurrency: {concurrency}", flush=True)

    if not remaining:
        print("[batch] Nothing to do — all files already processed.")
        return

    # Setup
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Auto-detect model
        if not LLM_MODEL:
            LLM_MODEL = await detect_model(client, LLM_BASE_URL)
            if LLM_MODEL:
                print(f"[batch] Auto-detected model: {LLM_MODEL}", flush=True)
            else:
                print(f"[batch] ERROR: Could not detect model at {LLM_BASE_URL}", flush=True)
                return

        print(f"[batch] LLM: {LLM_MODEL} at {LLM_BASE_URL}", flush=True)
        vlm_url = VLM_BASE_URL or LLM_BASE_URL
        print(f"[batch] VLM: {VLM_MODEL} at {vlm_url}", flush=True)

        # Process files
        total_t0 = time.time()
        success_count = 0
        error_count = 0

        for i, file_path in enumerate(remaining):
            progress = f"[{i + 1}/{len(remaining)}]"
            try:
                t0 = time.time()

                if file_path.suffix.lower() == ".pdf":
                    result = await process_pdf(
                        client, file_path, fmt, max_tokens, LLM_MODEL, semaphore
                    )
                elif file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    result = await process_image(
                        client, file_path, fmt, max_tokens, semaphore
                    )
                else:
                    print(f"{progress} SKIP {file_path.name} (unsupported type)", flush=True)
                    continue

                elapsed = time.time() - t0

                # Write output JSON
                # Preserve subdirectory structure from input
                rel_path = file_path.relative_to(input_dir)
                out_path = output_dir / rel_path.with_suffix(".json")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                mark_completed(state_file, str(file_path))
                success_count += 1

                pages = result["total_pages"]
                digital = result["digital_pages"]
                scanned = result["scanned_pages"]
                print(f"{progress} OK {file_path.name} "
                      f"({pages}p, {digital}d/{scanned}s, {elapsed:.1f}s) "
                      f"-> {out_path.name}", flush=True)

            except Exception as e:
                error_count += 1
                print(f"{progress} ERROR {file_path.name}: {e}", flush=True)

        # Summary
        total_elapsed = time.time() - total_t0
        total_done = success_count + len(completed)
        print(f"\n[batch] Done. {success_count} succeeded, {error_count} failed, "
              f"{total_done}/{len(files)} total complete.", flush=True)
        print(f"[batch] Total time: {total_elapsed:.1f}s "
              f"({success_count / max(total_elapsed, 0.001):.1f} docs/sec)", flush=True)

        if error_count > 0:
            print(f"[batch] Re-run with --resume to retry {error_count} failed files.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch document extraction — process directories of PDFs/TIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing input documents")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output JSON files")
    parser.add_argument("--format", default="award",
                        choices=[f.value for f in OutputFormat],
                        help="Output format / document type (default: award)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per LLM response (default: 4096)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent LLM/VLM requests (default: 4)")
    parser.add_argument("--extensions", nargs="+",
                        help="File extensions to process (default: all supported)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run (skip completed files)")
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
