#!/usr/bin/env python3
"""Document extraction server — hybrid text + VLM pipeline.

For digital PDFs: extracts text directly (PyMuPDF), then uses an LLM to
parse it into structured JSON.

For scanned PDFs / TIFFs / images: uses a Vision Language Model (Qwen2.5-VL)
to OCR and structure in one shot.

This hybrid approach is much faster and cheaper for digital documents
(no GPU needed for text extraction), while still handling scans correctly.

Launch:
    python ocr_app/scripts/ocr_server.py

Environment variables:
    LLM_BASE_URL    - vLLM / Ollama OpenAI-compatible endpoint for text parsing
                      (default: http://localhost:8000/v1)
    LLM_MODEL       - Model name for text parsing (default: auto-detected from endpoint)
    VLM_BASE_URL    - vLLM endpoint for vision model (scans/images only)
                      (default: same as LLM_BASE_URL)
    VLM_MODEL       - Vision model for scans (default: Qwen/Qwen2.5-VL-7B-Instruct)
    OCR_PORT        - Server port (default: 8090)
    OCR_HOST        - Server host (default: 0.0.0.0)
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # auto-detect if empty
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "")  # defaults to LLM_BASE_URL
VLM_MODEL = os.environ.get("VLM_MODEL", "QuantTrio/Qwen3-VL-32B-Instruct-AWQ")
HOST = os.environ.get("OCR_HOST", "0.0.0.0")
PORT = int(os.environ.get("OCR_PORT", "8090"))

# Minimum characters of extracted text to consider a page "digital"
# (vs. a scanned page that happens to have a tiny watermark or header)
MIN_TEXT_LENGTH = int(os.environ.get("MIN_TEXT_LENGTH", "50"))

# Local model mode: set LLM_BASE_URL=local to load the model directly
# with transformers instead of calling a vLLM/Ollama endpoint.
USE_LOCAL_MODEL = LLM_BASE_URL.lower() == "local"

# Globals for local model (populated in lifespan if USE_LOCAL_MODEL)
_local_model = None
_local_processor = None


# ---------------------------------------------------------------------------
# LLM / VLM inference via OpenAI-compatible API
# ---------------------------------------------------------------------------

async def _detect_model(base_url: str) -> str:
    """Auto-detect the model name from a vLLM/Ollama endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return ""


def _local_run_vlm(messages: list, max_tokens: int) -> str:
    """Run inference on the locally loaded model (shared by text + VLM paths)."""
    import torch
    from qwen_vl_utils import process_vision_info

    text_input = _local_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _local_processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_local_model.device)
    with torch.no_grad():
        generated_ids = _local_model.generate(**inputs, max_new_tokens=max_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return _local_processor.batch_decode(trimmed, skip_special_tokens=True)[0]


async def _llm_parse(text: str, prompt: str, max_tokens: int) -> str:
    """Send extracted text to an LLM for structured parsing."""
    full_prompt = f"{prompt}\n\n---\nDOCUMENT TEXT:\n---\n{text}"

    if USE_LOCAL_MODEL:
        messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
        return _local_run_vlm(messages, max_tokens)

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _vlm_ocr(image: Image.Image, prompt: str, max_tokens: int) -> str:
    """Send an image to a VLM for OCR + structured extraction in one shot."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    if USE_LOCAL_MODEL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{b64}"},
            {"type": "text", "text": prompt},
        ]}]
        return _local_run_vlm(messages, max_tokens)

    base_url = VLM_BASE_URL or LLM_BASE_URL
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Text extraction (digital PDFs)
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_bytes: bytes, page_indices: list[int]) -> list[dict]:
    """Extract text from PDF pages. Returns list of {page, text, has_text}."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []

    for page_idx in page_indices:
        page = doc[page_idx]
        text = page.get_text("text").strip()
        results.append({
            "page": page_idx,
            "text": text,
            "has_text": len(text) >= MIN_TEXT_LENGTH,
        })

    doc.close()
    return results


def _render_pdf_page(pdf_bytes: bytes, page_idx: int) -> Image.Image:
    """Render a single PDF page to an image at 2x resolution."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def _get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Get total page count of a PDF."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    award = "award"
    budget = "budget"
    terms = "terms"
    table = "table"
    key_values = "key_values"
    markdown = "markdown"
    json = "json"
    text = "text"
    library = "library"


# Prompts for LLM text parsing (text already extracted, no image)
TEXT_PROMPTS = {
    OutputFormat.award: (
        "Parse the following document text from a grant award notice / notice of award "
        "(research and sponsored programs).\n\n"
        "Return a JSON object with these fields (omit any that are not present):\n"
        '  "document_type": type of document (e.g. "Notice of Award", "Award Letter", "Subaward Agreement"),\n'
        '  "award_number": grant/award/contract number,\n'
        '  "sponsor": funding agency or organization,\n'
        '  "pi": principal investigator name(s),\n'
        '  "co_pis": array of co-PI names if listed,\n'
        '  "institution": awardee institution/organization,\n'
        '  "department": department or unit,\n'
        '  "project_title": title of the funded project,\n'
        '  "award_amount": total award amount (preserve exact formatting),\n'
        '  "current_period_amount": current period funding if shown,\n'
        '  "project_start": project/budget period start date,\n'
        '  "project_end": project/budget period end date,\n'
        '  "budget_periods": array of {period, start, end, amount} if shown,\n'
        '  "fa_rate": F&A / indirect cost rate and base if shown,\n'
        '  "cfda_number": CFDA number if shown,\n'
        '  "award_type": e.g. "New", "Continuation", "Supplement", "No-Cost Extension",\n'
        '  "special_conditions": array of any special terms or conditions,\n'
        '  "contacts": array of {role, name, email, phone} for program officers or admin contacts,\n'
        '  "additional_fields": object with any other labeled fields not covered above\n\n'
        "Preserve ALL dollar amounts, dates, and reference numbers exactly as they appear. "
        "Output only valid JSON."
    ),
    OutputFormat.budget: (
        "Parse the following document text from a research grant budget page, "
        "budget justification, or financial summary.\n\n"
        "Return a JSON object with:\n"
        '  "award_number": grant/award number if shown,\n'
        '  "budget_period": period or fiscal year if shown,\n'
        '  "categories": array of objects, each with:\n'
        '    "category": budget category (e.g. "Senior Personnel", "Fringe Benefits", '
        '"Equipment", "Travel", "Participant Support", "Other Direct Costs", "Indirect Costs"),\n'
        '    "items": array of {description, amount} line items,\n'
        '    "subtotal": category subtotal if shown\n'
        '  "total_direct": total direct costs,\n'
        '  "fa_rate": F&A / indirect cost rate,\n'
        '  "fa_base": F&A base (MTDC, TDC, etc.),\n'
        '  "total_indirect": total indirect/F&A costs,\n'
        '  "total": total project costs,\n'
        '  "cost_sharing": cost sharing amount if any,\n'
        '  "notes": any footnotes or annotations\n\n'
        "CRITICAL: Preserve ALL dollar amounts exactly — do not round, drop commas, "
        "or reformat. Output only valid JSON."
    ),
    OutputFormat.terms: (
        "Parse the following document text from "
        "terms & conditions, award terms, policies, or compliance requirements.\n\n"
        "Return a JSON object with:\n"
        '  "document_title": title of the document,\n'
        '  "effective_date": effective date if shown,\n'
        '  "sections": array of objects, each with:\n'
        '    "number": section number if present,\n'
        '    "title": section heading,\n'
        '    "text": full text of the section,\n'
        '    "subsections": array of {number, title, text} if nested\n'
        '  "definitions": object of defined terms if present,\n'
        '  "references": array of referenced regulations, OMB circulars, CFR citations\n\n'
        "Preserve exact wording — do not paraphrase or summarize. "
        "Include all regulatory citations (2 CFR 200, etc.) exactly. "
        "Output only valid JSON."
    ),
    OutputFormat.table: (
        "Extract the table(s) from the following document text. "
        "Return each table as a Markdown table with proper column alignment. "
        "If there are multiple tables, separate them with a blank line. "
        "Preserve ALL numeric values exactly. Output only the table(s)."
    ),
    OutputFormat.key_values: (
        "Extract all labeled data points from the following document text as "
        "key-value pairs. Look for field labels, line items, reference numbers, "
        "dates, names, and their corresponding values.\n\n"
        "Return a JSON object where keys are the field/label names and values "
        "are their corresponding values. For nested sections, use nested objects. "
        "Preserve ALL values exactly. Output only valid JSON."
    ),
    OutputFormat.markdown: (
        "Format the following document text as clean Markdown. "
        "Use headings, lists, bold/italic, and code blocks where appropriate. "
        "Preserve tables as Markdown tables. Output only the Markdown."
    ),
    OutputFormat.json: (
        "Parse the following document text and return it as a JSON object. "
        "Structure the content logically with appropriate keys. "
        "For forms, use field names as keys and field values as values. "
        "For documents, use sections as keys. Output only valid JSON."
    ),
    OutputFormat.text: (
        "Clean up and return the following document text, preserving the "
        "original reading order and structure. Output only the text."
    ),
    OutputFormat.library: (
        "Parse the following text extracted from a scanned library document. "
        "This may be a book page, manuscript, sheet music, newspaper, map, "
        "pamphlet, or other archival material.\n\n"
        "Return a JSON object with these fields (omit any that are not present):\n"
        '  "page_type": "<book_page | manuscript | sheet_music | newspaper | '
        'map | photograph | illustration | form | correspondence | ephemera | mixed>",\n'
        '  "title": running title, chapter title, or piece title visible on this page,\n'
        '  "creator": author, composer, artist, or cartographer if shown,\n'
        '  "date": any date visible (publication, manuscript, postmark, etc.),\n'
        '  "language": primary language of the text,\n'
        '  "page_number": printed or stamped page/folio number,\n'
        '  "body_text": full verbatim text in reading order (ALL text on the page),\n'
        '  "headings": array of section or chapter headings,\n'
        '  "footnotes": array of footnote texts,\n'
        '  "marginalia": array of handwritten marginal notes (describe if illegible),\n'
        '  "tables": array of tables as 2D arrays or key-value objects,\n'
        '  "illustrations": array of {description, caption} for images/figures/plates,\n'
        '  "musical_notation": {present: boolean, description: string} if sheet music,\n'
        '  "stamps_marks": array of {text, type} for library stamps, bookplates, '
        'accession numbers, call numbers, barcodes,\n'
        '  "physical_notes": any visible damage, stains, tears, or binding artifacts,\n'
        '  "one_sentence_summary": brief description of page content\n\n'
        "RULES:\n"
        "- Transcribe ALL printed and typed text VERBATIM — do not modernize spelling, "
        "correct errors, or expand abbreviations.\n"
        "- For sheet music: describe the notation (instrument, key, tempo markings, "
        "lyrics) but do not attempt to encode the music itself.\n"
        "- For handwriting: transcribe if legible, otherwise describe (e.g. "
        '"[illegible annotation in pencil, ~3 words]").\n'
        "- Preserve original line breaks within poetry or verse.\n"
        "- Capture ALL stamps, bookplates, and catalog markings — these are critical "
        "for provenance.\n"
        "- Output ONLY valid JSON."
    ),
}

# Prompts for VLM one-shot OCR (image input, for scans/TIFFs)
VLM_PROMPTS = {
    OutputFormat.award: (
        "Extract all information from this scanned grant award notice / notice of award. "
        "This is a research and sponsored programs document.\n\n"
        "Return a JSON object with these fields (omit any that are not present):\n"
        '  "document_type", "award_number", "sponsor", "pi", "co_pis", '
        '"institution", "department", "project_title", "award_amount", '
        '"current_period_amount", "project_start", "project_end", '
        '"budget_periods" (array of {period, start, end, amount}), '
        '"fa_rate", "cfda_number", "award_type", "special_conditions", '
        '"contacts" (array of {role, name, email, phone}), "additional_fields"\n\n'
        "Preserve ALL dollar amounts, dates, and reference numbers exactly as printed. "
        "Output only valid JSON."
    ),
    OutputFormat.budget: (
        "Extract the budget information from this scanned research grant document.\n\n"
        "Return a JSON object with: "
        '"award_number", "budget_period", '
        '"categories" (array of {category, items: [{description, amount}], subtotal}), '
        '"total_direct", "fa_rate", "fa_base", "total_indirect", "total", '
        '"cost_sharing", "notes"\n\n'
        "Preserve ALL dollar amounts exactly as printed. Output only valid JSON."
    ),
    OutputFormat.terms: (
        "Extract the terms and conditions from this scanned research and sponsored "
        "programs document.\n\n"
        "Return a JSON object with: "
        '"document_title", "effective_date", '
        '"sections" (array of {number, title, text, subsections}), '
        '"definitions", "references"\n\n'
        "Preserve exact wording — do not paraphrase. Output only valid JSON."
    ),
    OutputFormat.table: (
        "Extract the table(s) from this image. Return each table as a "
        "Markdown table with proper column alignment. "
        "Preserve ALL numeric values exactly. Output only the table(s)."
    ),
    OutputFormat.key_values: (
        "Extract all labeled data points from this image as key-value pairs. "
        "Return a JSON object where keys are field names and values are their values. "
        "Preserve ALL values exactly. Output only valid JSON."
    ),
    OutputFormat.markdown: (
        "Extract all text from this image and format it as clean Markdown. "
        "Preserve tables as Markdown tables. Output only the Markdown."
    ),
    OutputFormat.json: (
        "Extract all text from this image and return it as a JSON object. "
        "Structure the content logically with appropriate keys. "
        "Output only valid JSON."
    ),
    OutputFormat.text: (
        "Extract all text from this image exactly as it appears. "
        "Preserve the original reading order, line breaks, and structure. "
        "Output only the extracted text."
    ),
    OutputFormat.library: (
        "Extract all information from this scanned library document page. "
        "This may be a book page, manuscript, sheet music, newspaper, map, "
        "photograph, pamphlet, or other archival material.\n\n"
        "Return a JSON object with these fields (omit any not present):\n"
        '  "page_type": "<book_page | manuscript | sheet_music | newspaper | '
        'map | photograph | illustration | form | correspondence | ephemera | mixed>",\n'
        '  "title", "creator", "date", "language", "page_number",\n'
        '  "body_text": full verbatim text in reading order (ALL text on the page),\n'
        '  "headings": array of section/chapter headings,\n'
        '  "footnotes": array of footnote texts,\n'
        '  "marginalia": array of handwritten marginal notes,\n'
        '  "tables": array of tables as 2D arrays or key-value objects,\n'
        '  "illustrations": array of {description, caption},\n'
        '  "musical_notation": {present: boolean, description: string},\n'
        '  "stamps_marks": array of {text, type} for library stamps, bookplates, '
        'accession numbers, call numbers, barcodes,\n'
        '  "physical_notes": visible damage, stains, tears, binding artifacts,\n'
        '  "one_sentence_summary": brief description of page content\n\n'
        "RULES:\n"
        "- Transcribe ALL text VERBATIM — do not modernize spelling or correct errors.\n"
        "- For sheet music: describe notation (instrument, key, tempo, lyrics) "
        "but do not encode the music.\n"
        "- For handwriting: transcribe if legible, otherwise describe "
        '(e.g. "[illegible annotation in pencil, ~3 words]").\n'
        "- Preserve line breaks in poetry/verse.\n"
        "- Capture ALL stamps, bookplates, and catalog markings.\n"
        "- Output ONLY valid JSON."
    ),
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global LLM_MODEL, _local_model, _local_processor

    if USE_LOCAL_MODEL:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_name = VLM_MODEL
        print(f"[ocr_server] LOCAL MODE — loading {model_name} with transformers...", flush=True)
        _local_processor = AutoProcessor.from_pretrained(model_name)
        _local_model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        LLM_MODEL = model_name
        print(f"[ocr_server] Model loaded on {_local_model.device}", flush=True)
        print(f"[ocr_server] All pages rendered as images -> VLM extraction", flush=True)
    else:
        # Auto-detect model name from endpoint
        if not LLM_MODEL:
            LLM_MODEL = await _detect_model(LLM_BASE_URL)
            if LLM_MODEL:
                print(f"[ocr_server] Auto-detected LLM model: {LLM_MODEL}", flush=True)
            else:
                print(f"[ocr_server] WARNING: Could not detect model at {LLM_BASE_URL}", flush=True)
        print(f"[ocr_server] LLM: {LLM_MODEL} at {LLM_BASE_URL}", flush=True)
        vlm_url = VLM_BASE_URL or LLM_BASE_URL
        print(f"[ocr_server] VLM: {VLM_MODEL} at {vlm_url} (fallback for scans)", flush=True)

    print(f"[ocr_server] Min text length for digital detection: {MIN_TEXT_LENGTH}", flush=True)
    yield


app = FastAPI(
    title="Document Extraction Server",
    version="2.0.0",
    description=(
        "Document extraction: all pages rendered as images and sent to a VLM "
        "for structured JSON extraction. Supports PDFs, TIFFs, and images."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PageResult(BaseModel):
    """Result for a single page."""
    page: int
    text: str
    format: str
    method: str  # "text_extraction" or "vlm_ocr"
    elapsed_ms: float


class DocumentResponse(BaseModel):
    """Response for a document (single or multi-page)."""
    filename: str
    pages: list[PageResult]
    total_pages: int
    digital_pages: int
    scanned_pages: int
    total_elapsed_ms: float
    llm_model: str
    vlm_model: str


class ImageResponse(BaseModel):
    """Response for a single image."""
    text: str
    format: str
    method: str
    elapsed_ms: float
    vlm_model: str
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "vlm_model": VLM_MODEL,
        "llm_endpoint": LLM_BASE_URL,
    }


@app.get("/info")
async def info():
    return {
        "llm_model": LLM_MODEL,
        "vlm_model": VLM_MODEL,
        "llm_endpoint": LLM_BASE_URL,
        "vlm_endpoint": VLM_BASE_URL or LLM_BASE_URL,
        "min_text_length": MIN_TEXT_LENGTH,
        "formats": [f.value for f in OutputFormat],
        "pipeline": "all pages rendered as images -> VLM structured extraction",
    }


@app.post("/extract/pdf", response_model=DocumentResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.award),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(8192),
    pages: Optional[str] = Form(None),
):
    """Extract structured data from a PDF.

    All pages are rendered as images and sent to the VLM, which captures
    layout, tables, signatures, watermarks, and annotations that text
    extraction alone would miss.
    """
    import fitz  # PyMuPDF

    contents = await file.read()
    total_page_count = _get_pdf_page_count(contents)
    page_indices = _parse_pages(pages, total_page_count)

    vlm_prompt = prompt or VLM_PROMPTS[format]

    # Check text layer for informational logging
    extracted = _extract_pdf_text(contents, page_indices)

    results = []
    digital_count = 0
    scanned_count = 0
    total_t0 = time.time()

    for page_info in extracted:
        t0 = time.time()

        # All pages go through VLM as images
        img = _render_pdf_page(contents, page_info["page"])
        result_text = await _vlm_ocr(img, vlm_prompt, max_tokens)
        method = "vlm_image"

        if page_info["has_text"]:
            digital_count += 1
        else:
            scanned_count += 1

        elapsed_ms = (time.time() - t0) * 1000

        results.append(PageResult(
            page=page_info["page"] + 1,  # 1-indexed for display
            text=result_text,
            format=format.value,
            method=method,
            elapsed_ms=round(elapsed_ms, 2),
        ))

    total_elapsed = (time.time() - total_t0) * 1000

    return DocumentResponse(
        filename=file.filename or "unknown.pdf",
        pages=results,
        total_pages=len(results),
        digital_pages=digital_count,
        scanned_pages=scanned_count,
        total_elapsed_ms=round(total_elapsed, 2),
        llm_model=LLM_MODEL,
        vlm_model=VLM_MODEL,
    )


@app.post("/extract/image", response_model=ImageResponse)
async def extract_image(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.award),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(8192),
):
    """Extract structured data from an image (TIFF, PNG, JPG, etc.).

    Always uses VLM since images have no extractable text layer.
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    if img.mode != "RGB":
        img = img.convert("RGB")

    vlm_prompt = prompt or VLM_PROMPTS[format]

    t0 = time.time()
    result_text = await _vlm_ocr(img, vlm_prompt, max_tokens)
    elapsed_ms = (time.time() - t0) * 1000

    return ImageResponse(
        text=result_text,
        format=format.value,
        method="vlm_ocr",
        elapsed_ms=round(elapsed_ms, 2),
        vlm_model=VLM_MODEL,
        image_width=img.width,
        image_height=img.height,
    )


# ---------------------------------------------------------------------------
# Legacy OCR endpoints (kept for backwards compat)
# ---------------------------------------------------------------------------

@app.post("/ocr/upload")
async def ocr_upload(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.text),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(4096),
):
    """Legacy endpoint — routes to extract/image."""
    return await extract_image(file=file, format=format, prompt=prompt, max_tokens=max_tokens)


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.text),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(4096),
    pages: Optional[str] = Form(None),
):
    """Legacy endpoint — routes to extract/pdf."""
    return await extract_pdf(file=file, format=format, prompt=prompt, max_tokens=max_tokens, pages=pages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pages(pages_str: Optional[str], total_pages: int) -> list[int]:
    """Parse a page range string like '1-5' or '1,3,5' into 0-indexed list."""
    if not pages_str:
        return list(range(total_pages))

    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total_pages, int(end))
            indices.update(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                indices.add(idx)

    return sorted(indices)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
