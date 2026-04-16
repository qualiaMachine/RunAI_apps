"""Chunk-based RSP document extraction helpers.

Replaces the per-page sliding-window flow with a whole-document / chunked
flow. When the document fits in the VLM's context, it is sent in a single
call with all pages as images. When it doesn't, it is split into
overlapping chunks; each chunk is sent in one VLM call. Chunk results are
merged by ``scripts.merge.merge_chunks`` (dedupe + continuation stitch).

Design notes:
- Each chunk produces ONE doc-synthesis-style JSON (see ``doc_prompt.py``),
  not one JSON per page. Merge operates on chunk outputs, not page outputs.
- Overlap should be >= the longest atomic unit we expect (table / narrative).
  Five pages is a reasonable default for RSP; budget tables and
  publication lists occasionally exceed that and will rely on the
  continuation-flag fallback inside ``merge.merge_chunks``.
- Page numbers in the final Gemini-style JSON are tagged per CHUNK, not
  per page — ``PageRange: "PAGES 3-12"`` — since the VLM no longer produces
  per-page output. Finer-grained attribution would require asking the VLM
  to tag each item with an image index, which we are intentionally avoiding
  to keep the prompt focused on the extraction task.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------

def chunk_page_ranges(
    total_pages: int,
    max_pages_per_chunk: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Plan overlapping chunks that cover [0, total_pages).

    Returns a list of (start, end_exclusive) page indices. Each chunk has
    length <= ``max_pages_per_chunk``; consecutive chunks share ``overlap``
    pages. If the whole document fits in one chunk, returns a single
    [0, total_pages) range.

    Invariants:
    - Every page index appears in at least one chunk.
    - Adjacent chunks share exactly ``overlap`` pages (except possibly the
      last, which may have a larger overlap if the tail would otherwise
      be shorter than a full chunk).
    - No empty chunks.
    """
    if total_pages <= 0:
        return []
    if max_pages_per_chunk <= 0:
        raise ValueError("max_pages_per_chunk must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_pages_per_chunk:
        raise ValueError("overlap must be < max_pages_per_chunk")

    if total_pages <= max_pages_per_chunk:
        return [(0, total_pages)]

    stride = max_pages_per_chunk - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_pages:
        end = min(start + max_pages_per_chunk, total_pages)
        ranges.append((start, end))
        if end == total_pages:
            break
        start += stride

    # If the last two chunks have a large overlap (because the tail was
    # short), back-merge them so we don't run a near-duplicate chunk.
    if len(ranges) >= 2:
        a_start, a_end = ranges[-2]
        b_start, b_end = ranges[-1]
        if b_end - a_start <= max_pages_per_chunk:
            ranges[-2:] = [(a_start, b_end)]

    return ranges


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def build_chunk_messages(
    images: list,
    prompt: str,
    encode_image_b64: Callable,
    filename: Optional[str] = None,
    page_range_label: Optional[str] = None,
    links_per_page: Optional[list[list[dict]]] = None,
    is_first_chunk: bool = True,
    is_last_chunk: bool = True,
    pinned_images: Optional[list[tuple[Any, str]]] = None,
    forward_context: Optional[dict] = None,
) -> list[dict]:
    """Build the OpenAI-style ``messages`` payload for a chunk extraction.

    The VLM receives all chunk images in order, preceded by a context
    header that tells it which pages of the source document the chunk
    represents and whether it is the first/last chunk (used by the
    prompt to decide when continuation flags are meaningful).

    Args:
        images: list of PIL images in page order for this chunk.
        prompt: the doc-synthesis prompt text.
        encode_image_b64: callable that encodes a PIL image as a base64
            data URI (reuses the notebook's ``_encode_image_b64``).
        filename: source filename, surfaced to the model for context.
        page_range_label: human-readable label like "Pages 3-12 of 40".
        links_per_page: optional per-page hyperlinks extracted from the PDF.
        is_first_chunk / is_last_chunk: drive continuation-flag guidance.
        pinned_images: extra images shown before the chunk's own pages,
            labelled as PINNED CONTEXT. Typical use is pinning page 1
            (cover sheet) into every non-first chunk so the model always
            has access to title / funding agency / document type.
        forward_context: structured context from an earlier chunk
            (typically chunk 1's ``document_details``). Rendered as text
            so the model can anchor doc-level fields without guessing
            from a partial view.
    """
    header_parts: list[str] = []
    if filename:
        header_parts.append(f"Source filename: {filename}")
    if page_range_label:
        header_parts.append(f"Chunk page range: {page_range_label}")

    boundary_note = _continuation_hint(is_first_chunk, is_last_chunk)
    if boundary_note:
        header_parts.append(boundary_note)

    if forward_context:
        header_parts.append(_render_forward_context(forward_context))

    if links_per_page:
        flat = []
        for i, links in enumerate(links_per_page):
            for l in links or []:
                flat.append(f"  [page image {i+1}] {l.get('text','')} -> {l.get('url','')}")
        if flat:
            header_parts.append("Hyperlinks found in these pages:\n" + "\n".join(flat))

    prefix = "\n\n".join(header_parts)

    content: list[dict] = []

    # Pinned images first (e.g. page 1 cover sheet) so the model sees
    # them before the chunk's own pages.
    for img, label in pinned_images or []:
        content.append({"type": "text", "text": f"[PINNED CONTEXT — {label}]"})
        content.append({"type": "image", "image": encode_image_b64(img)})

    for i, img in enumerate(images):
        content.append({"type": "text", "text": f"[PAGE IMAGE {i+1} of {len(images)}]"})
        content.append({"type": "image", "image": encode_image_b64(img)})

    full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
    content.append({"type": "text", "text": full_prompt})

    return [{"role": "user", "content": content}]


def _render_forward_context(ctx: dict) -> str:
    """Format structured forward-context as a text block for the prompt.

    The model is told these are verified anchors from an earlier chunk —
    it should USE them rather than re-deriving doc-level metadata from
    its own (partial) view. Keeps the keys named exactly as they will
    be used in the output schema so the model doesn't need to translate.
    """
    lines = [
        "VERIFIED CONTEXT FROM EARLIER IN THIS DOCUMENT:",
        "The following fields were extracted from the document's opening pages.",
        "Treat them as authoritative — do NOT re-derive from your partial view.",
        "Copy them through into your document_details output unchanged.",
        "",
    ]
    dd = ctx.get("document_details") or {}
    for k, v in dd.items():
        if v not in (None, "", [], {}):
            lines.append(f"  document_details.{k}: {v!r}")
    if ctx.get("document_tags"):
        lines.append(f"  document_tags: {ctx['document_tags']!r}")
    if ctx.get("one_sentence_summary"):
        lines.append(f"  one_sentence_summary: {ctx['one_sentence_summary']!r}")
    return "\n".join(lines)


def _continuation_hint(is_first: bool, is_last: bool) -> str:
    """Tell the VLM when continuation flags should be considered."""
    if is_first and is_last:
        return (
            "You are seeing the ENTIRE document. All continuation flags "
            "(continues_from_previous_chunk, continues_to_next_chunk) "
            "must be false — there is nothing before or after."
        )
    if is_first:
        return (
            "You are seeing the FIRST chunk of a multi-chunk document. "
            "Nothing comes before the first image — set "
            "continues_from_previous_chunk to false for all items. "
            "Items that run off the LAST image you can see should have "
            "continues_to_next_chunk: true."
        )
    if is_last:
        return (
            "You are seeing the LAST chunk of a multi-chunk document. "
            "Nothing comes after the last image — set "
            "continues_to_next_chunk to false for all items. Items that "
            "were already in progress on the FIRST image you can see "
            "should have continues_from_previous_chunk: true."
        )
    return (
        "You are seeing a MIDDLE chunk of a multi-chunk document. Items "
        "already in progress on the first image should have "
        "continues_from_previous_chunk: true; items running off the last "
        "image should have continues_to_next_chunk: true."
    )


# ---------------------------------------------------------------------------
# Doc-level assembly from merged snake_case JSON
# ---------------------------------------------------------------------------

def assemble_document_from_merged(
    filename: str,
    merged: dict,
    chunk_results: list[dict],
    model_name: str,
    parse_filename_fn: Callable,
    prompt_text: Optional[str] = None,
    run_info: Optional[dict] = None,
) -> dict:
    """Turn merged chunk JSON into the Gemini-style PascalCase document.

    The ``merged`` input comes from ``scripts.merge.merge_chunks`` and uses
    the snake_case schema defined in ``doc_prompt.py``. This function
    produces the same PascalCase shape that the existing
    ``assemble_document()`` in the notebook produces, so downstream
    comparison tooling works unchanged.

    ``chunk_results`` is the per-chunk metadata (timings, page ranges,
    parse errors) — not the merged content. Used to build the coverage
    manifest.
    """
    file_meta = parse_filename_fn(filename)

    # --- Aggregate chunk-level metadata
    chunk_times_ms = [c.get("elapsed_ms", 0) for c in chunk_results]
    total_sec = round(sum(chunk_times_ms) / 1000, 1) if chunk_times_ms else 0.0
    total_pages = sum(c.get("page_end", 0) - c.get("page_start", 0) for c in chunk_results)

    parse_error_chunks = sum(1 for c in chunk_results if c.get("parse_error"))

    # --- Transform span fields to PascalCase + tag with chunk page ranges
    # With chunked extraction we can only attribute items to their chunk's
    # page range, not an exact page. Downstream tooling that relied on
    # exact PageNumber should treat PageRange as the closest equivalent.
    pascal_tables = []
    for t in merged.get("tables", []) or []:
        pascal_tables.append({
            "PageRange": t.get("_source_page_range", "UNKNOWN"),
            "PrecedingSectionHeader": t.get("preceding_section_header", ""),
            "TableClassification": t.get("table_classification", "Standard_Table"),
            "TableData": t.get("table_data", []),
        })

    pascal_narratives = []
    for n in merged.get("narrative_responses", []) or []:
        pascal_narratives.append({
            "SectionOrPage": n.get("_source_page_range", "UNKNOWN"),
            "PrecedingSectionHeader": n.get("preceding_section_header", ""),
            "PromptOrHeader": n.get("prompt_or_header", ""),
            "VerbatimText": n.get("verbatim_text", ""),
        })

    pascal_stakeholders = []
    for s in merged.get("stakeholders", []) or []:
        pascal_stakeholders.append({
            "PageRange": s.get("_source_page_range", "UNKNOWN"),
            "ContextSnippet": s.get("context_snippet", ""),
            "StakeholderRole": s.get("stakeholder_role", "Unknown"),
            "FullName": s.get("full_name", ""),
            "FirstName": s.get("first_name", ""),
            "LastName": s.get("last_name", ""),
            "Email": s.get("email", ""),
            "Phone": s.get("phone", ""),
            "Institution": s.get("institution", ""),
            "Department": s.get("department", ""),
            "PositionTitle": s.get("position_title", ""),
            "HighestEducation": s.get("highest_education", ""),
            "RawStakeholderText": s.get("raw_stakeholder_text", ""),
        })

    pascal_addresses = []
    for a in merged.get("addresses", []) or []:
        pascal_addresses.append({
            "PageRange": a.get("_source_page_range", "UNKNOWN"),
            "ContextSnippet": a.get("context_snippet", ""),
            "Addressee": a.get("addressee", ""),
            "CareOf": a.get("care_of"),
            "AddressLine1": a.get("address_line1", ""),
            "AddressLine2": a.get("address_line2", ""),
            "City": a.get("city", ""),
            "StateProvince": a.get("state_province", ""),
            "PostalCode": a.get("postal_code", ""),
            "Country": a.get("country", ""),
            "StakeholderType": a.get("stakeholder_type", "Unknown"),
            "RawAddressText": a.get("raw_address_text", ""),
        })

    dd = merged.get("document_details") or {}
    doc_details_pascal = {
        "ApplicationID": dd.get("application_id", ""),
        "ApplicationType": dd.get("application_type", ""),
        "Title": dd.get("title", ""),
        "RequestedAmount": dd.get("requested_amount"),
        "CompletedDate": dd.get("completed_date", ""),
        "SubDocumentType": dd.get("sub_document_type", ""),
    }

    sig = merged.get("signature_lines") or {}
    sig_info = {
        "PageRange": _first_chunk_range_with_signature(chunk_results),
        "HasSignatureLine": bool(sig.get("has_signature_line")),
        "HasValidSignature": bool(sig.get("has_valid_signature")),
    }

    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    timing = {
        "TotalElapsedSec": total_sec,
        "ChunkCount": len(chunk_results),
        "AvgChunkSec": round(sum(chunk_times_ms) / len(chunk_times_ms) / 1000, 2) if chunk_times_ms else 0,
        "MinChunkSec": round(min(chunk_times_ms) / 1000, 2) if chunk_times_ms else 0,
        "MaxChunkSec": round(max(chunk_times_ms) / 1000, 2) if chunk_times_ms else 0,
    }
    full_run_info = {
        "Model": model_name,
        "Timestamp": now_iso,
        "Timing": timing,
        "ExtractionMode": "chunked" if len(chunk_results) > 1 else "single_shot",
        **(run_info or {}),
        "ExtractionPrompt": prompt_text,
    }

    extraction_coverage = {
        "TotalPages": total_pages,
        "ChunkCount": len(chunk_results),
        "ChunksWithParseErrors": parse_error_chunks,
        "TableCount": len(pascal_tables),
        "NarrativeCount": len(pascal_narratives),
        "StakeholderCount": len(pascal_stakeholders),
        "AddressCount": len(pascal_addresses),
    }

    # One-sentence summary: merge.py already picked the longest chunk
    # summary as a placeholder. Keep a list (one per chunk) for parity
    # with the page-based notebook's OneSentenceNarrativeSummary field.
    per_chunk_summaries = []
    for c in chunk_results:
        per_chunk_summaries.append({
            "PageRange": c.get("page_range_label", ""),
            "Summary": ((c.get("extracted") or {}).get("one_sentence_summary") or "N/A"),
        })

    return {
        "Filename": filename,
        "FileNameMetaData": file_meta,
        "PageCount": total_pages,
        "ConfidencePercentage": merged.get("confidence_percentage", 0.0),
        "ConfidenceNarrative": merged.get("confidence_narrative", ""),
        "LLMModelAndVersion": model_name,
        "CurrentDateAndTime": now_iso,
        "HasAnnotation": bool(merged.get("has_annotation")),
        "HasWatermark": bool(merged.get("has_watermark")),
        "SignatureLines": sig_info,
        "DocumentTags": sorted(merged.get("document_tags") or []),
        "OneSentenceNarrativeSummary": per_chunk_summaries,
        "DocumentSummary": merged.get("one_sentence_summary", ""),
        "DocumentDetails": doc_details_pascal,
        "Stakeholders": pascal_stakeholders,
        "AddressesCollection": pascal_addresses,
        "TablesCollection": pascal_tables,
        "NarrativeResponses": pascal_narratives,
        "OtherMetadata": merged.get("other_metadata") or {},
        "ExtractionCoverage": extraction_coverage,
        "RunInfo": full_run_info,
    }


def _first_chunk_range_with_signature(chunk_results: list[dict]) -> Optional[str]:
    """Return the page range label of the first chunk that saw a signature."""
    for c in chunk_results:
        sig = ((c.get("extracted") or {}).get("signature_lines") or {})
        if sig.get("has_signature_line"):
            return c.get("page_range_label")
    return None


# ---------------------------------------------------------------------------
# Page-range tagging (post-merge, pre-assembly)
# ---------------------------------------------------------------------------

def tag_source_page_ranges(merged: dict, chunk_ranges: list[tuple[int, int]]) -> dict:
    """Add ``_source_page_range`` to every mergeable item.

    merge_chunks does not carry chunk-of-origin metadata on items (it
    collapses duplicates across chunks). To preserve the chunk attribution
    for the Gemini-style PascalCase output, we accept it as a best-effort
    "appeared first in chunk K" via chunk_ranges. This runs before
    ``assemble_document_from_merged``.

    Note: when an item appears in multiple chunks (dedupe hit), we cannot
    reconstruct which chunk's copy survived. We tag with "MULTIPLE" for
    such cases. Downstream consumers that want strict attribution should
    look at RunInfo.Timing.ChunkCount and the per-chunk audit file.

    For the minimum-viable path, this function is a no-op placeholder —
    items will get "UNKNOWN" in the PageRange field. A later iteration can
    plumb through richer attribution if needed.
    """
    # Intentionally no-op for now: the merge step doesn't preserve
    # chunk-of-origin. Leaving this function as the hook where such
    # plumbing would land so callers can depend on it being called.
    return merged
