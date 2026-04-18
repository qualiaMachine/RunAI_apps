"""Chunk-based RSP document extraction helpers.

When the whole document fits in the VLM's context, it is sent in a single
call with all pages as images. When it doesn't, it is split into
overlapping chunks; each chunk is sent in one VLM call and the per-chunk
results are merged by ``scripts.merge.merge_chunks``.

Design notes:
- Each chunk produces ONE doc-synthesis-style JSON (see ``doc_prompt.py``),
  not one JSON per page. ``merge_chunks`` operates on chunk outputs, not
  page outputs.
- Overlap should be >= the longest atomic unit we expect (table /
  narrative). Five pages is a reasonable default for RSP; long tables
  and publication lists occasionally exceed that and rely on the
  continuation-flag stitch fallback inside ``merge_chunks``.

The Gemini-style PascalCase assembly that used to live here was removed
when the merged output shape was flattened to mirror a chunk's extracted
JSON (see ``scripts/merge.py``). Consumers now read the merged JSON with
the same field names they'd read from a single-chunk extraction.
"""

from __future__ import annotations

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
        a_start, _ = ranges[-2]
        b_start, b_end = ranges[-1]
        if b_end - a_start <= max_pages_per_chunk:
            ranges[-2:] = [(a_start, b_end)]

    return ranges


# ---------------------------------------------------------------------------
# Chunk extraction message construction
# ---------------------------------------------------------------------------

def build_chunk_messages(
    images: list,
    prompt: str,
    encode_image_b64: Callable,
    filename: Optional[str] = None,
    links_per_page: Optional[list[list[dict]]] = None,
    is_first_chunk: bool = True,
    is_last_chunk: bool = True,
    pinned_images: Optional[list[tuple[Any, str]]] = None,
    forward_context: Optional[dict] = None,
    first_pdf_page: int = 1,
) -> list[dict]:
    """Build the OpenAI-style ``messages`` payload for a chunk extraction.

    The VLM receives all chunk images in order, preceded by a context
    header that tells it which pages of the source document the chunk
    represents and whether it is the first/last chunk (used by the
    prompt to decide when continuation flags are meaningful).
    """
    header_parts: list[str] = []
    if filename:
        header_parts.append(f"Source filename: {filename}")

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

    for img, label in pinned_images or []:
        content.append({"type": "text", "text": f"[PINNED CONTEXT — {label}]"})
        content.append({"type": "image", "image": encode_image_b64(img)})

    for i, img in enumerate(images):
        pdf_page = first_pdf_page + i
        content.append({
            "type": "text",
            "text": f"[PAGE IMAGE {i+1} of {len(images)} — PDF page {pdf_page}]",
        })
        content.append({"type": "image", "image": encode_image_b64(img)})

    full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
    content.append({"type": "text", "text": full_prompt})

    return [{"role": "user", "content": content}]


def _render_forward_context(ctx: dict) -> str:
    """Format structured forward-context as a text block for the prompt."""
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
