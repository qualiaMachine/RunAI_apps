"""Merge logic for chunk-level RSP extractions into a single document JSON.

Strategy: dedupe-first, stitch-as-fallback.

Chunks are produced by running the doc-synthesis prompt over overlapping
page ranges of a single document. Each chunk's JSON matches the schema in
``doc_prompt.py``. When chunks overlap by enough pages, any atomic unit
(table, narrative, stakeholder, address) that is shorter than the overlap
appears fully in at least one chunk. We dedupe those by fingerprint,
keeping the most complete copy. Items that span past the overlap still
fragment — those are handled by the continuation-flag stitch pass.

Three field categories, three merge strategies:

1. Span fields (``tables``, ``narrative_responses``): dedupe by
   fingerprint first; any leftover fragments (those with continuation
   flags set and no dedupe partner) are stitched in chunk order.

2. Identity fields (``stakeholders``, ``addresses``): dedupe by
   fingerprint. No stitching — an entity appearing in multiple chunks
   is the same entity, not a continuation.

3. Doc-level fields (``document_details``, ``document_tags``,
   ``one_sentence_summary``, ``signature_lines``, ``has_annotation``,
   ``has_watermark``, ``confidence_*``): aggregated by rule (union,
   null-coalesce, max, or pick-best-by-confidence).

The merge is deterministic and does not call the LLM. A separate
optional synthesis pass (not in this module) can be used to rewrite
``one_sentence_summary`` once the merged JSON is built.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    """Normalize a string for fingerprinting: lowercase, collapse whitespace."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _stakeholder_fingerprint(s: dict) -> str:
    """Identity key for a stakeholder.

    Tiered: email > phone > personal name > role+org. The role+org tier
    catches unnamed recurring references like "local environmental grant
    specialist" that appear dozens of times across a long document — each
    occurrence is the same role, not a separate person. An empty
    fingerprint means we really couldn't key the entry and it passes
    through unmerged.
    """
    email = _norm(s.get("email"))
    if email:
        return f"email:{email}"
    phone = _norm(s.get("phone"))
    if phone:
        return f"phone:{phone}"
    first = _norm(s.get("first_name"))
    last = _norm(s.get("last_name"))
    full = _norm(s.get("full_name"))
    inst = _norm(s.get("institution"))
    dept = _norm(s.get("department"))
    position = _norm(s.get("position_title"))
    role = _norm(s.get("stakeholder_role"))
    if last and first:
        return f"name:{last}|{first}|{inst}"
    if full:
        return f"name:{full}|{inst}"
    if position:
        return f"role:{position}|{inst}|{dept}"
    if role and role != "unknown" and inst:
        return f"orgrole:{role}|{inst}|{dept}"
    return ""


def _address_fingerprint(a: dict) -> str:
    """Identity key for an address: normalized postal code + line1 + city."""
    line1 = _norm(a.get("address_line1"))
    postal = _norm(a.get("postal_code"))
    city = _norm(a.get("city"))
    if not any([line1, postal, city]):
        return ""
    return f"addr:{postal}|{line1}|{city}"


def _table_fingerprint(t: dict) -> str:
    """Match key for tables.

    Disambiguation is: classification + page + header_signature + first_row.
    We prefer the printed ``visual_page_number`` for disambiguation over
    ``preceding_section_header``: the VLM regularly assigns different
    section headers to the same table across overlapping chunks (e.g.
    "Funding" vs "Section 1, Table 1…"), which broke dedupe. The printed
    page number is stable across chunks. When the printed page is missing,
    we fall back to the section header so same-shape tables in different
    sections (Year 1 vs Year 2 budgets) still stay separate.
    """
    cls = _norm(t.get("table_classification"))
    page = _norm(t.get("visual_page_number"))
    disambig = page if page else _norm(t.get("preceding_section_header"))
    header_sig = _norm(_table_header_signature(t))
    first_row = _norm(_table_first_row_text(t))
    return _hash(f"tbl:{cls}|{disambig}|{header_sig}|{first_row}")


def _table_header_signature(t: dict) -> str:
    """Derive a header signature from table_data based on classification."""
    cls = t.get("table_classification", "")
    data = t.get("table_data")
    if cls == "Standard_Table" and isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return "|".join(sorted(first.keys()))
    if cls == "Key_Value_Form" and isinstance(data, dict):
        return "|".join(sorted(data.keys())[:8])
    if cls == "Literal_Grid" and isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, list):
            return "|".join(str(c) for c in first)
    return ""


def _table_first_row_text(t: dict) -> str:
    """First row content, best-effort, for disambiguating same-header tables."""
    data = t.get("table_data")
    if isinstance(data, list) and data:
        row = data[0]
        if isinstance(row, dict):
            return "|".join(str(v) for v in row.values())
        if isinstance(row, list):
            return "|".join(str(c) for c in row)
    if isinstance(data, dict) and data:
        k = next(iter(data))
        return f"{k}={data[k]}"
    return ""


def _narrative_fingerprint(n: dict) -> str:
    """Match key for a narrative response.

    Same rationale as ``_table_fingerprint``: prefer the printed page for
    cross-chunk disambiguation since the VLM can label the same narrative
    with different ``preceding_section_header`` values in overlapping
    chunks. ``verbatim_text`` head chars + ``prompt_or_header`` keep
    truly distinct narratives on the same page apart.
    """
    page = _norm(n.get("visual_page_number"))
    disambig = page if page else _norm(n.get("preceding_section_header"))
    header = _norm(n.get("prompt_or_header"))
    head_chars = _norm(n.get("verbatim_text", ""))[:120]
    return _hash(f"narr:{disambig}|{header}|{head_chars}")


# ---------------------------------------------------------------------------
# Completeness comparison
# ---------------------------------------------------------------------------

def _table_size(t: dict) -> int:
    """Row count for a table, used to pick the most complete copy on dedupe."""
    data = t.get("table_data")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return 0


def _is_fragment(item: dict) -> bool:
    """An item with any continuation flag set is a fragment."""
    return bool(
        item.get("continues_from_previous_chunk")
        or item.get("continues_to_next_chunk")
    )


def _pick_more_complete(a: dict, b: dict, size_fn) -> dict:
    """Prefer non-fragments over fragments; then larger size; then first."""
    a_frag, b_frag = _is_fragment(a), _is_fragment(b)
    if a_frag != b_frag:
        return b if a_frag else a
    if size_fn(a) != size_fn(b):
        return a if size_fn(a) >= size_fn(b) else b
    return a


# ---------------------------------------------------------------------------
# Identity-field merge (stakeholders, addresses)
# ---------------------------------------------------------------------------

def _merge_identity(items_by_chunk: list[list[dict]], fingerprint_fn) -> list[dict]:
    """Dedupe identity-field items across chunks by fingerprint.

    When an item has no fingerprint (empty key), keep it as-is — we'd
    rather have a duplicate than silently collapse two unrelated entries.
    When fingerprints match, merge field-by-field using null-coalesce
    (prefer non-empty values; longer string wins on conflict).
    """
    out: list[dict] = []
    by_key: dict[str, int] = {}
    for chunk_items in items_by_chunk:
        for item in chunk_items or []:
            key = fingerprint_fn(item)
            if not key:
                out.append(item)
                continue
            if key in by_key:
                out[by_key[key]] = _coalesce_fields(out[by_key[key]], item)
            else:
                by_key[key] = len(out)
                out.append(dict(item))
    return out


def _coalesce_fields(a: dict, b: dict) -> dict:
    """Merge two dicts, preferring non-empty values. Longer strings win ties."""
    out = dict(a)
    for k, v_b in b.items():
        v_a = out.get(k)
        out[k] = _pick_value(v_a, v_b)
    return out


def _pick_value(a, b):
    """Pick the 'better' of two values: non-empty wins, then longer string."""
    if _is_empty(a):
        return b
    if _is_empty(b):
        return a
    if isinstance(a, str) and isinstance(b, str):
        return a if len(a) >= len(b) else b
    if isinstance(a, list) and isinstance(b, list):
        return a if len(a) >= len(b) else b
    return a


def _is_empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


# ---------------------------------------------------------------------------
# Span-field merge (tables, narrative_responses)
# ---------------------------------------------------------------------------

def _merge_span(
    items_by_chunk: list[list[dict]],
    fingerprint_fn,
    size_fn,
    concat_fn,
) -> list[dict]:
    """Merge span-type items: dedupe fully-contained copies, stitch fragments.

    Phase 1 (dedupe): group items by fingerprint across all chunks. Within
    each group, keep the most complete copy (non-fragment > fragment, then
    larger size). Items with no fingerprint partner are passed through.

    Phase 2 (stitch): for fragments that weren't paired by fingerprint,
    walk chunks in order and pair items that have matching
    ``continues_to_next_chunk``/``continues_from_previous_chunk`` flags.
    Match by fingerprint first, fall back to positional match within the
    fragment queue of each chunk boundary.
    """
    # --- Phase 1: fingerprint-based dedupe
    groups: dict[str, list[tuple[int, int, dict]]] = {}
    passthrough: list[tuple[int, int, dict]] = []
    for ci, chunk_items in enumerate(items_by_chunk):
        for ii, item in enumerate(chunk_items or []):
            key = fingerprint_fn(item)
            if not key:
                passthrough.append((ci, ii, item))
            else:
                groups.setdefault(key, []).append((ci, ii, item))

    kept: dict[tuple[int, int], dict] = {}  # (chunk_idx, item_idx) -> item
    for key, group in groups.items():
        if len(group) == 1:
            ci, ii, item = group[0]
            kept[(ci, ii)] = item
            continue
        # Pick best among copies
        best = group[0]
        for cand in group[1:]:
            if _pick_more_complete(cand[2], best[2], size_fn) is cand[2]:
                best = cand
        kept[(best[0], best[1])] = best[2]
    for ci, ii, item in passthrough:
        kept[(ci, ii)] = item

    # --- Phase 2: stitch fragments across adjacent chunks
    # Only consider items that survived Phase 1 and still carry continuation
    # flags. This catches spans longer than the chunk overlap, which no
    # single chunk saw completely.
    stitched_out: list[dict] = []
    consumed: set[tuple[int, int]] = set()

    # Build per-chunk ordered lists of surviving items (in original order)
    per_chunk: list[list[tuple[int, dict]]] = [[] for _ in items_by_chunk]
    for (ci, ii), item in kept.items():
        per_chunk[ci].append((ii, item))
    for lst in per_chunk:
        lst.sort()

    for ci in range(len(per_chunk)):
        for ii, item in per_chunk[ci]:
            if (ci, ii) in consumed:
                continue
            # Try to stitch forward across chunks while this item runs on
            cur = dict(item)
            cur_ci, cur_ii = ci, ii
            while cur.get("continues_to_next_chunk") and cur_ci + 1 < len(per_chunk):
                nxt_ci = cur_ci + 1
                partner = _find_stitch_partner(
                    cur, per_chunk[nxt_ci], consumed, nxt_ci, fingerprint_fn,
                )
                if partner is None:
                    break
                p_ii, p_item = partner
                cur = concat_fn(cur, p_item)
                consumed.add((nxt_ci, p_ii))
                cur_ci, cur_ii = nxt_ci, p_ii
                if not cur.get("continues_to_next_chunk"):
                    break
            consumed.add((ci, ii))
            stitched_out.append(cur)

    return stitched_out


def _find_stitch_partner(
    cur: dict,
    next_chunk_items: list[tuple[int, dict]],
    consumed: set,
    next_ci: int,
    fingerprint_fn,
) -> tuple[int, dict] | None:
    """Find the item in next_chunk that continues `cur`.

    Preferred match: same fingerprint + continues_from_previous_chunk=true.
    Fallback: first unconsumed item in the next chunk with
    continues_from_previous_chunk=true.
    """
    cur_fp = fingerprint_fn(cur)
    fallback: tuple[int, dict] | None = None
    for ii, item in next_chunk_items:
        if (next_ci, ii) in consumed:
            continue
        if not item.get("continues_from_previous_chunk"):
            continue
        if cur_fp and fingerprint_fn(item) == cur_fp:
            return (ii, item)
        if fallback is None:
            fallback = (ii, item)
    return fallback


def _concat_tables(a: dict, b: dict) -> dict:
    """Concat two table fragments. b continues a."""
    out = dict(a)
    a_data = a.get("table_data")
    b_data = b.get("table_data")
    if isinstance(a_data, list) and isinstance(b_data, list):
        out["table_data"] = list(a_data) + list(b_data)
    elif isinstance(a_data, dict) and isinstance(b_data, dict):
        merged = dict(a_data)
        merged.update(b_data)
        out["table_data"] = merged
    else:
        # Mixed/unexpected — keep a's shape, append b as-is if possible
        out["table_data"] = a_data
    out["continues_to_next_chunk"] = bool(b.get("continues_to_next_chunk"))
    out["continues_from_previous_chunk"] = bool(a.get("continues_from_previous_chunk"))
    return out


def _concat_narratives(a: dict, b: dict) -> dict:
    """Concat two narrative fragments. b continues a."""
    out = dict(a)
    a_txt = a.get("verbatim_text", "") or ""
    b_txt = b.get("verbatim_text", "") or ""
    # Join with a single space to avoid double-spacing; callers can reflow
    joiner = " " if a_txt and b_txt and not a_txt.endswith((" ", "\n")) else ""
    out["verbatim_text"] = f"{a_txt}{joiner}{b_txt}"
    out["continues_to_next_chunk"] = bool(b.get("continues_to_next_chunk"))
    out["continues_from_previous_chunk"] = bool(a.get("continues_from_previous_chunk"))
    return out


# ---------------------------------------------------------------------------
# Doc-level aggregation
# ---------------------------------------------------------------------------

def _agg_doc_details(chunks: list[dict]) -> dict:
    """Null-coalesce doc_details fields across chunks. Longer strings win."""
    out: dict = {}
    for c in chunks:
        d = c.get("document_details") or {}
        for k, v in d.items():
            out[k] = _pick_value(out.get(k), v)
    return out


def _agg_tags(chunks: list[dict]) -> list[str]:
    """Union of document_tags, preserving first-seen order."""
    seen: dict[str, None] = {}
    for c in chunks:
        for tag in c.get("document_tags") or []:
            if isinstance(tag, str):
                key = _norm(tag)
                if key not in seen:
                    seen[key] = None
    # Return the first-cased version of each tag
    out: list[str] = []
    seen_keys: set[str] = set()
    for c in chunks:
        for tag in c.get("document_tags") or []:
            if isinstance(tag, str):
                key = _norm(tag)
                if key in seen and key not in seen_keys:
                    out.append(tag)
                    seen_keys.add(key)
    return out


def _agg_signature(chunks: list[dict]) -> dict:
    """OR-aggregate: any chunk with a signature line means the doc has one."""
    return {
        "has_signature_line": any(
            (c.get("signature_lines") or {}).get("has_signature_line")
            for c in chunks
        ),
        "has_valid_signature": any(
            (c.get("signature_lines") or {}).get("has_valid_signature")
            for c in chunks
        ),
    }


def _agg_confidence(chunks: list[dict]) -> tuple[float, str]:
    """Mean of confidence_percentage across chunks + concatenated narrative."""
    vals = [
        c.get("confidence_percentage")
        for c in chunks
        if isinstance(c.get("confidence_percentage"), (int, float))
    ]
    pct = round(sum(vals) / len(vals), 2) if vals else 0.0
    narratives = [
        c.get("confidence_narrative", "")
        for c in chunks
        if c.get("confidence_narrative")
    ]
    return pct, " | ".join(narratives)


def _best_summary(chunks: list[dict]) -> str:
    """Pick the longest one_sentence_summary as a placeholder.

    A proper document-level summary should be produced by a downstream
    LLM synthesis pass over the merged JSON. We return the longest chunk
    summary here as a deterministic, non-LLM fallback.
    """
    best = ""
    for c in chunks:
        s = c.get("one_sentence_summary") or ""
        if isinstance(s, str) and len(s) > len(best):
            best = s
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_chunks(chunks: list[dict]) -> dict:
    """Merge a list of per-chunk extraction JSONs into one document JSON.

    Input order must be the page order of the chunks (chunk 0 = earliest
    pages). Each chunk must match the schema in ``doc_prompt.py``.
    """
    if not chunks:
        return {}
    if len(chunks) == 1:
        return _strip_continuation_flags(dict(chunks[0]))

    pct, narr = _agg_confidence(chunks)
    merged: dict = {
        "confidence_percentage": pct,
        "confidence_narrative": narr,
        "has_annotation": any(c.get("has_annotation") for c in chunks),
        "has_watermark": any(c.get("has_watermark") for c in chunks),
        "signature_lines": _agg_signature(chunks),
        "document_tags": _agg_tags(chunks),
        "one_sentence_summary": _best_summary(chunks),
        "document_details": _agg_doc_details(chunks),
        "stakeholders": _merge_identity(
            [c.get("stakeholders", []) for c in chunks],
            _stakeholder_fingerprint,
        ),
        "addresses": _merge_identity(
            [c.get("addresses", []) for c in chunks],
            _address_fingerprint,
        ),
        "tables": _merge_span(
            [c.get("tables", []) for c in chunks],
            _table_fingerprint,
            _table_size,
            _concat_tables,
        ),
        "narrative_responses": _merge_span(
            [c.get("narrative_responses", []) for c in chunks],
            _narrative_fingerprint,
            lambda n: len(n.get("verbatim_text") or ""),
            _concat_narratives,
        ),
        "other_metadata": _merge_other_metadata(chunks),
    }
    return _strip_continuation_flags(merged)


def _merge_other_metadata(chunks: list[dict]) -> dict:
    """Shallow-merge other_metadata across chunks; later chunks win on conflict."""
    out: dict = {}
    for c in chunks:
        m = c.get("other_metadata") or {}
        if isinstance(m, dict):
            out.update(m)
    return out


def _strip_continuation_flags(doc: dict) -> dict:
    """Remove continuation flags from the final merged output.

    These are a chunk-level internal signal; they should not leak into
    the final document JSON that downstream consumers see.
    """
    for item in doc.get("tables", []) or []:
        item.pop("continues_from_previous_chunk", None)
        item.pop("continues_to_next_chunk", None)
    for item in doc.get("narrative_responses", []) or []:
        item.pop("continues_from_previous_chunk", None)
        item.pop("continues_to_next_chunk", None)
    return doc


def merge_chunks_json(chunk_texts: list[str]) -> dict:
    """Convenience wrapper: parse raw VLM JSON strings, then merge.

    Chunks that fail to parse as JSON are skipped (with a note recorded
    in ``other_metadata.merge_errors``). This is a common failure mode
    when the VLM hits its max_tokens limit mid-JSON.
    """
    parsed: list[dict] = []
    errors: list[str] = []
    for i, raw in enumerate(chunk_texts):
        try:
            parsed.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError) as e:
            errors.append(f"chunk {i}: {e}")
    merged = merge_chunks(parsed)
    if errors:
        merged.setdefault("other_metadata", {})["merge_errors"] = errors
    return merged
