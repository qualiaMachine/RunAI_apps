"""Merge per-chunk extractions into a single doc-level JSON.

Design goal: the merged output should look almost identical to a single
chunk's extracted JSON, just with content unioned across chunks. No
PascalCase transform, no Gemini-style reshuffling — downstream consumers
should be able to treat the merged JSON with the same code that reads a
chunk JSON.

Shape of the merged output:

    {
      "one_sentence_summary": "",          # filled by pass-2 VLM synthesis
      "experiment": {...},                 # doc-level settings (from chunks[0])
      "confidence_percentage": <mean>,
      "confidence_narrative": "<concat>",
      "document_details": {...},           # null-coalesced across chunks
      "has_annotation": <any-true>,
      "has_watermark": <any-true>,
      "signature_lines": {
        "has_signature_line": <any-true>,
        "has_valid_signature": <any-true>,
      },
      "document_tags": [...],              # union
      "stakeholders": [...],               # deduped union
      "addresses": [...],                  # deduped union
      "tables": [...],                     # deduped + stitched union
      "narrative_responses": [...],        # deduped + stitched union
      "other_metadata": {...},             # shallow-merged
      "boundary_notes": [...],             # deterministic lint output
      "chunks": [                          # per-chunk sidecar
        {"chunk_index", "page_start", "page_end",
         "experiment": {...},
         "extracted": {
           "one_sentence_summary", "confidence_percentage", "confidence_narrative"}}
      ],
    }

``merge_chunks`` accepts either raw extracted dicts (backwards-compat for
tests) OR full chunk records (``{"extracted": {...}, "chunk_index":...,
"page_start":..., "page_end":..., "experiment":{...}}``). When records
are passed, ``experiment`` and ``chunks[]`` are populated; otherwise they
default to ``{}``/``[]``.

Dedup strategy for arrays:
    - identity fields (stakeholders, addresses): dedupe by fingerprint.
    - span fields (tables, narrative_responses): dedupe fully-contained
      copies by fingerprint, then stitch fragments via continuation flags.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable


# Settings-type keys we expect to be constant across chunks — copied to the
# doc-level ``experiment`` field. Per-chunk runtime keys (elapsed_ms,
# timestamp) live under ``chunks[i].experiment`` instead.
_PER_CHUNK_EXPERIMENT_KEYS = {"elapsed_ms", "timestamp"}


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _stakeholder_fingerprint(s: dict) -> str:
    """Tiered identity key: email > phone > personal name > role+org > raw text.

    The raw_stakeholder_text tail is a last-resort key for recurring
    unnamed references (a doc that repeats the same block of org contact
    text in every chunk) — far from perfect but better than letting
    dozens of copies through unkeyed.
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
    raw = _norm(s.get("raw_stakeholder_text"))
    if raw:
        return f"raw:{_hash(raw[:120])}"
    return ""


def _stakeholder_is_empty(s: dict) -> bool:
    """True when a stakeholder has nothing usable — drop rather than keep."""
    for field in (
        "email", "phone", "first_name", "last_name", "full_name",
        "institution", "department", "position_title",
        "raw_stakeholder_text", "context_snippet",
    ):
        if _norm(s.get(field)):
            return False
    return True


def _address_fingerprint(a: dict) -> str:
    line1 = _norm(a.get("address_line1"))
    postal = _norm(a.get("postal_code"))
    city = _norm(a.get("city"))
    if not any([line1, postal, city]):
        return ""
    return f"addr:{postal}|{line1}|{city}"


def _table_fingerprint(t: dict) -> str:
    """classification + page (fallback to section) + header_sig + first_row."""
    cls = _norm(t.get("table_classification"))
    page = _norm(t.get("visual_page_number"))
    disambig = page if page else _norm(t.get("preceding_section_header"))
    header_sig = _norm(_table_header_signature(t))
    first_row = _norm(_table_first_row_text(t))
    return _hash(f"tbl:{cls}|{disambig}|{header_sig}|{first_row}")


def _table_header_signature(t: dict) -> str:
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
    page = _norm(n.get("visual_page_number"))
    disambig = page if page else _norm(n.get("preceding_section_header"))
    header = _norm(n.get("prompt_or_header"))
    head_chars = _norm(n.get("verbatim_text", ""))[:120]
    return _hash(f"narr:{disambig}|{header}|{head_chars}")


# ---------------------------------------------------------------------------
# Completeness comparison
# ---------------------------------------------------------------------------

def _table_size(t: dict) -> int:
    data = t.get("table_data")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return 0


def _is_fragment(item: dict) -> bool:
    return bool(
        item.get("continues_from_previous_chunk")
        or item.get("continues_to_next_chunk")
    )


def _pick_more_complete(a: dict, b: dict, size_fn) -> dict:
    a_frag, b_frag = _is_fragment(a), _is_fragment(b)
    if a_frag != b_frag:
        return b if a_frag else a
    if size_fn(a) != size_fn(b):
        return a if size_fn(a) >= size_fn(b) else b
    return a


# ---------------------------------------------------------------------------
# Identity-field merge (stakeholders, addresses)
# ---------------------------------------------------------------------------

def _merge_identity(
    items_by_chunk: list[list[dict]],
    fingerprint_fn: Callable[[dict], str],
    empty_fn: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Dedupe identity items by fingerprint.

    Items with an empty fingerprint are kept (we'd rather have a duplicate
    than silently collapse low-confidence entries). When ``empty_fn`` is
    provided, items it flags as wholly empty are dropped before they reach
    the fingerprint step — useful for filtering out ``stakeholder_role:
    "Unknown"`` + all-empty-fields noise that the VLM sometimes emits on
    pages with just a page number reference.
    """
    out: list[dict] = []
    by_key: dict[str, int] = {}
    for chunk_items in items_by_chunk:
        for item in chunk_items or []:
            if empty_fn and empty_fn(item):
                continue
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
    out = dict(a)
    for k, v_b in b.items():
        out[k] = _pick_value(out.get(k), v_b)
    return out


def _pick_value(a, b):
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
    fingerprint_fn: Callable[[dict], str],
    size_fn: Callable[[dict], int],
    concat_fn: Callable[[dict, dict], dict],
) -> list[dict]:
    """Dedupe fully-contained copies, then stitch fragments across chunks."""
    groups: dict[str, list[tuple[int, int, dict]]] = {}
    passthrough: list[tuple[int, int, dict]] = []
    for ci, chunk_items in enumerate(items_by_chunk):
        for ii, item in enumerate(chunk_items or []):
            key = fingerprint_fn(item)
            if not key:
                passthrough.append((ci, ii, item))
            else:
                groups.setdefault(key, []).append((ci, ii, item))

    kept: dict[tuple[int, int], dict] = {}
    for group in groups.values():
        if len(group) == 1:
            ci, ii, item = group[0]
            kept[(ci, ii)] = item
            continue
        best = group[0]
        for cand in group[1:]:
            if _pick_more_complete(cand[2], best[2], size_fn) is cand[2]:
                best = cand
        kept[(best[0], best[1])] = best[2]
    for ci, ii, item in passthrough:
        kept[(ci, ii)] = item

    stitched_out: list[dict] = []
    consumed: set[tuple[int, int]] = set()

    per_chunk: list[list[tuple[int, dict]]] = [[] for _ in items_by_chunk]
    for (ci, ii), item in kept.items():
        per_chunk[ci].append((ii, item))
    for lst in per_chunk:
        lst.sort()

    for ci in range(len(per_chunk)):
        for ii, item in per_chunk[ci]:
            if (ci, ii) in consumed:
                continue
            cur = dict(item)
            cur_ci = ci
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
                cur_ci = nxt_ci
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
    fingerprint_fn: Callable[[dict], str],
) -> tuple[int, dict] | None:
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
        out["table_data"] = a_data
    out["continues_to_next_chunk"] = bool(b.get("continues_to_next_chunk"))
    out["continues_from_previous_chunk"] = bool(a.get("continues_from_previous_chunk"))
    return out


def _concat_narratives(a: dict, b: dict) -> dict:
    out = dict(a)
    a_txt = a.get("verbatim_text", "") or ""
    b_txt = b.get("verbatim_text", "") or ""
    joiner = " " if a_txt and b_txt and not a_txt.endswith((" ", "\n")) else ""
    out["verbatim_text"] = f"{a_txt}{joiner}{b_txt}"
    out["continues_to_next_chunk"] = bool(b.get("continues_to_next_chunk"))
    out["continues_from_previous_chunk"] = bool(a.get("continues_from_previous_chunk"))
    return out


# ---------------------------------------------------------------------------
# Doc-level aggregation
# ---------------------------------------------------------------------------

def _agg_doc_details(extracted_list: list[dict]) -> dict:
    out: dict = {}
    for e in extracted_list:
        d = e.get("document_details") or {}
        for k, v in d.items():
            out[k] = _pick_value(out.get(k), v)
    return out


def _agg_tags(extracted_list: list[dict]) -> list[str]:
    seen_keys: set[str] = set()
    out: list[str] = []
    for e in extracted_list:
        for tag in e.get("document_tags") or []:
            if isinstance(tag, str):
                key = _norm(tag)
                if key and key not in seen_keys:
                    out.append(tag)
                    seen_keys.add(key)
    return out


def _agg_signature(extracted_list: list[dict]) -> dict:
    return {
        "has_signature_line": any(
            (e.get("signature_lines") or {}).get("has_signature_line")
            for e in extracted_list
        ),
        "has_valid_signature": any(
            (e.get("signature_lines") or {}).get("has_valid_signature")
            for e in extracted_list
        ),
    }


def _agg_confidence(extracted_list: list[dict]) -> float:
    """Mean confidence across chunks. Per-chunk narratives live in chunks[]
    — concatenating 24 chunk narratives at the doc level produced a
    multi-paragraph blob of near-identical boilerplate, so the doc-level
    field is just the numeric mean now.
    """
    vals = [
        e.get("confidence_percentage")
        for e in extracted_list
        if isinstance(e.get("confidence_percentage"), (int, float))
    ]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _merge_other_metadata(extracted_list: list[dict]) -> dict:
    out: dict = {}
    for e in extracted_list:
        m = e.get("other_metadata") or {}
        if isinstance(m, dict):
            out.update(m)
    return out


_PAGE_NUMBER_FOOTER_PATTERNS = [
    # "50 | Page", "50 |Page", "50|Page"
    re.compile(r"^(\d+)\s*\|\s*page\b", re.IGNORECASE),
    # "Page 12", "Page 12 of 142"
    re.compile(r"^page\s+(\d+)", re.IGNORECASE),
    # "12 | 142", "12/142" (current/total) — take first
    re.compile(r"^(\d+)\s*[|/]\s*\d+$"),
]


def normalize_visual_page_number(raw):
    """Strip common footer artifacts from a printed-page-number string.

    The VLM sometimes captures footer decoration verbatim (``"50 | Page"``
    for PDF page 51, ``"Page 12 of 142"``, etc.) because the extraction
    prompt asks for the value to be verbatim. That breaks cross-chunk
    dedupe and page-sort ordering, so we normalize anything that matches
    a known footer pattern to just the page identifier.
    """
    if raw is None or raw == "":
        return raw
    s = str(raw).strip()
    for pat in _PAGE_NUMBER_FOOTER_PATTERNS:
        m = pat.match(s)
        if m:
            return m.group(1)
    return s


def _normalize_page_numbers_inplace(items: list[dict]) -> None:
    for it in items or []:
        page = it.get("visual_page_number")
        normed = normalize_visual_page_number(page)
        if normed != page:
            it["visual_page_number"] = normed


def _page_sort_key(item: dict) -> tuple:
    """Stable sort key based on ``visual_page_number``.

    Numeric pages sort numerically; roman-numeral / appendix-style pages
    (e.g. "iii", "A-5") sort after numeric pages lexicographically. Items
    with no printed page number sort last. Leading digits are used when a
    page looks like "50 something" so decoration doesn't poison the sort.
    """
    page = item.get("visual_page_number")
    if page is None or page == "":
        return (2, "")
    s = str(page).strip()
    try:
        return (0, int(s))
    except ValueError:
        pass
    m = re.match(r"^(\d+)", s)
    if m:
        return (0, int(m.group(1)))
    return (1, s.lower())


# ---------------------------------------------------------------------------
# Input-shape detection + sidecar construction
# ---------------------------------------------------------------------------

def _is_chunk_record(c: dict) -> bool:
    """A full chunk record has ``extracted`` as a dict sub-object.

    Raw extracted dicts have ``tables``/``narrative_responses`` at the top
    level. We never have both.
    """
    return (
        isinstance(c.get("extracted"), dict)
        and "tables" not in c
        and "narrative_responses" not in c
    )


def _doc_level_experiment(records: list[dict]) -> dict:
    """Copy chunks[0].experiment minus per-chunk runtime keys."""
    if not records:
        return {}
    exp = records[0].get("experiment") or {}
    return {k: v for k, v in exp.items() if k not in _PER_CHUNK_EXPERIMENT_KEYS}


def _build_chunks_sidecar(records: list[dict]) -> list[dict]:
    """Per-chunk summary + confidence + page range + runtime experiment."""
    out: list[dict] = []
    for c in records:
        extracted = c.get("extracted") or {}
        out.append({
            "chunk_index": c.get("chunk_index"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "experiment": c.get("experiment") or {},
            "extracted": {
                "one_sentence_summary": extracted.get("one_sentence_summary", ""),
                "confidence_percentage": extracted.get("confidence_percentage"),
                "confidence_narrative": extracted.get("confidence_narrative", ""),
            },
        })
    return out


def _is_empty_table(t: dict) -> bool:
    """A Standard_Table / Key_Value_Form / Literal_Grid with no rows/cells.

    The VLM sometimes tags a section header (e.g. 'Depreciation',
    'DONATED PROFESSIONAL LABOR') as a Standard_Table when there's no
    actual table below it, producing an entry with ``table_data: []``.
    These are pure noise — drop them in postprocess.
    """
    data = t.get("table_data")
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        return len(data) == 0
    return True


# ---------------------------------------------------------------------------
# Deterministic lint (boundary_notes)
# ---------------------------------------------------------------------------

def _lint_merged(merged: dict) -> list[str]:
    """Flag likely merge issues without calling the VLM.

    Empty-table entries are DROPPED in postprocess (see ``_is_empty_table``),
    not flagged here — the VLM emitting a header as a Standard_Table with
    no rows is a known, benign failure mode that we prefer to silently
    clean up rather than surface to the user.
    """
    notes: list[str] = []

    # Narratives with no text.
    for i, n in enumerate(merged.get("narrative_responses") or []):
        if not (n.get("verbatim_text") or "").strip():
            page = n.get("visual_page_number") or "?"
            header = n.get("prompt_or_header") or "(no header)"
            notes.append(
                f"narrative_responses[{i}]: empty verbatim_text on page "
                f"{page} under '{header}'"
            )

    # Unkeyed stakeholders (passed through without dedupe).
    unkeyed = [
        s for s in merged.get("stakeholders") or []
        if not _stakeholder_fingerprint(s)
    ]
    if unkeyed:
        notes.append(
            f"stakeholders: {len(unkeyed)} entries with no identifiable key "
            f"(no email, phone, name, position_title) — passed through unmerged"
        )

    # Items that still carry continuation flags (shouldn't after strip).
    stale_flags = sum(
        1 for t in (merged.get("tables") or [])
        if t.get("continues_from_previous_chunk") or t.get("continues_to_next_chunk")
    )
    if stale_flags:
        notes.append(
            f"tables: {stale_flags} items still carry continuation flags "
            f"after merge — stitch likely failed"
        )
    stale_flags_n = sum(
        1 for n in (merged.get("narrative_responses") or [])
        if n.get("continues_from_previous_chunk") or n.get("continues_to_next_chunk")
    )
    if stale_flags_n:
        notes.append(
            f"narrative_responses: {stale_flags_n} items still carry "
            f"continuation flags after merge"
        )

    return notes


def _strip_continuation_flags(doc: dict) -> dict:
    for item in doc.get("tables", []) or []:
        item.pop("continues_from_previous_chunk", None)
        item.pop("continues_to_next_chunk", None)
    for item in doc.get("narrative_responses", []) or []:
        item.pop("continues_from_previous_chunk", None)
        item.pop("continues_to_next_chunk", None)
    return doc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_chunks(chunks: list[dict], extraction_prompt: str | None = None) -> dict:
    """Merge per-chunk extraction records into one doc-level JSON.

    ``chunks`` accepts either:
      - Raw extracted dicts (the VLM's per-chunk JSON as-is), OR
      - Full chunk records with ``extracted`` plus ``chunk_index``,
        ``page_start``, ``page_end``, ``experiment``.

    When full records are passed, the merged output includes a populated
    ``experiment`` (from chunks[0]) and ``chunks[]`` sidecar. Raw-dict
    input produces an otherwise-identical merged doc with ``experiment``
    ``{}`` and ``chunks`` ``[]``.

    ``extraction_prompt``: the pass-1 prompt text used when extracting
    chunks. If provided, it's recorded under ``experiment.extraction_prompt``
    so the merged JSON captures exactly what the VLM was asked.

    Lint notes (``boundary_notes``) are always computed at the end.
    Stakeholders and addresses are sorted by ``visual_page_number`` so
    downstream readers get a natural top-of-doc-to-bottom reading order.
    """
    if not chunks:
        return {}

    records: list[dict] = []
    if all(_is_chunk_record(c) for c in chunks):
        records = chunks
        extracted_list = [c.get("extracted") or {} for c in chunks]
    else:
        extracted_list = chunks

    # Normalize VLM footer-text drift ("50 | Page" → "50") before anything
    # downstream sees the values. Both dedupe and page-sort rely on stable
    # page identifiers; the VLM sometimes captures the decoration verbatim.
    for e in extracted_list:
        for field in ("tables", "narrative_responses", "stakeholders", "addresses"):
            _normalize_page_numbers_inplace(e.get(field) or [])

    if len(extracted_list) == 1:
        # Single chunk: pass content straight through, no dedupe/stitch work.
        merged = dict(extracted_list[0])
        _strip_continuation_flags(merged)
        # Strip the per-chunk confidence_narrative from the doc-level view —
        # chunks[].extracted preserves it per-chunk.
        # Doc-level confidence_narrative is filled by pass-2 VLM synthesis
        # (a short summary of chunks[].extracted.confidence_narrative).
        merged["confidence_narrative"] = ""
        merged["experiment"] = _doc_level_experiment(records)
        if extraction_prompt:
            merged["experiment"]["extraction_prompt"] = extraction_prompt
        # Drop empty-noise stakeholders even on the single-chunk path.
        if merged.get("stakeholders"):
            merged["stakeholders"] = sorted(
                (s for s in merged["stakeholders"] if not _stakeholder_is_empty(s)),
                key=_page_sort_key,
            )
        if merged.get("addresses"):
            merged["addresses"] = sorted(merged["addresses"], key=_page_sort_key)
        if merged.get("tables"):
            merged["tables"] = [
                t for t in merged["tables"] if not _is_empty_table(t)
            ]
        merged["chunks"] = _build_chunks_sidecar(records)
        merged["boundary_notes"] = _lint_merged(merged)
        return merged

    experiment = _doc_level_experiment(records)
    if extraction_prompt:
        experiment["extraction_prompt"] = extraction_prompt

    stakeholders = _merge_identity(
        [e.get("stakeholders", []) for e in extracted_list],
        _stakeholder_fingerprint,
        empty_fn=_stakeholder_is_empty,
    )
    addresses = _merge_identity(
        [e.get("addresses", []) for e in extracted_list],
        _address_fingerprint,
    )

    merged = {
        "one_sentence_summary": "",  # filled by pass-2 VLM synthesis
        "experiment": experiment,
        "confidence_percentage": _agg_confidence(extracted_list),
        # Doc-level narrative is also filled by pass-2: synthesizing the
        # per-chunk narratives keeps this short instead of concatenating
        # ~24 near-identical boilerplate paragraphs.
        "confidence_narrative": "",
        "document_details": _agg_doc_details(extracted_list),
        "has_annotation": any(e.get("has_annotation") for e in extracted_list),
        "has_watermark": any(e.get("has_watermark") for e in extracted_list),
        "signature_lines": _agg_signature(extracted_list),
        "document_tags": _agg_tags(extracted_list),
        "stakeholders": sorted(stakeholders, key=_page_sort_key),
        "addresses": sorted(addresses, key=_page_sort_key),
        "tables": _merge_span(
            [e.get("tables", []) for e in extracted_list],
            _table_fingerprint,
            _table_size,
            _concat_tables,
        ),
        "narrative_responses": _merge_span(
            [e.get("narrative_responses", []) for e in extracted_list],
            _narrative_fingerprint,
            lambda n: len(n.get("verbatim_text") or ""),
            _concat_narratives,
        ),
        "other_metadata": _merge_other_metadata(extracted_list),
        "chunks": _build_chunks_sidecar(records),
    }
    _strip_continuation_flags(merged)
    # Drop empty-table entries (VLM sometimes tags a section heading as a
    # Standard_Table with no rows — pure noise). Done after dedupe so any
    # chunk that actually captured rows wins via _pick_more_complete.
    merged["tables"] = [t for t in (merged.get("tables") or []) if not _is_empty_table(t)]
    merged["boundary_notes"] = _lint_merged(merged)
    return merged


def merge_chunks_json(chunk_texts: list[str]) -> dict:
    """Convenience wrapper: parse raw VLM JSON strings, then merge.

    Chunks that fail to parse as JSON are skipped with a note recorded in
    ``other_metadata.merge_errors`` — a common failure mode when the VLM
    hits its max_tokens limit mid-JSON.
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
