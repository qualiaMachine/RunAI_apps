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
      "potential_issues": [...],           # deterministic lint output
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


# Fields that identify a stakeholder for the subset-compat dedup pass.
# Two entries merge only if, for every field here, their values are
# identical OR at least one side is empty. Any real disagreement
# (different email, different name, different department string) keeps
# them separate. Fields intentionally omitted — raw_stakeholder_text,
# context_snippet, visual_page_number — are informational or positional;
# they inform which entry survives but don't block merging.
_STAKEHOLDER_IDENTITY_FIELDS = (
    "stakeholder_role",
    "institution",
    "department",
    "email",
    "phone",
    "full_name",
    "first_name",
    "last_name",
    "position_title",
    "highest_education",
)


def _stakeholders_subset_compatible(a: dict, b: dict) -> bool:
    """True when a and b have no conflicting identity fields.

    Requires at least one field to match on both sides so wholly-empty
    pairs don't silently collapse — those carry no identity signal and
    ``_merge_identity`` already handled them via raw-text fingerprint.
    """
    any_match = False
    for field in _STAKEHOLDER_IDENTITY_FIELDS:
        av = _norm(a.get(field))
        bv = _norm(b.get(field))
        if av and bv:
            if av != bv:
                return False
            any_match = True
    return any_match


def _finalize_stakeholders(stakeholders: list[dict]) -> list[dict]:
    """Final post-processing pass to collapse compatible stakeholder dupes.

    Runs after ``_merge_identity`` to catch the common pattern where the
    same organization appears multiple times with slight field-population
    variations — e.g. one entry fills institution+department, a second
    has just institution, a third only raw_stakeholder_text. When no
    identity field conflicts, fold into the most-complete entry.

    Conservative by design: any conflict on an identity field
    (different email, different department string) preserves both
    entries. Stakeholders no longer carry any page metadata, so
    there's nothing here that could drift out of sync.
    """
    out: list[dict] = []
    for item in stakeholders or []:
        merged_idx = None
        for i, existing in enumerate(out):
            if _stakeholders_subset_compatible(existing, item):
                out[i] = _coalesce_fields(existing, item)
                merged_idx = i
                break
        if merged_idx is None:
            out.append(dict(item))
    return out


def _address_fingerprint(a: dict) -> str:
    line1 = _norm(a.get("address_line1"))
    postal = _norm(a.get("postal_code"))
    city = _norm(a.get("city"))
    if not any([line1, postal, city]):
        return ""
    return f"addr:{postal}|{line1}|{city}"


_SECTION_HEADER_NOISE = re.compile(
    r"^\**\s*(?:rev|new|updated|revised)\s*\**[\s:\-\u2013\u2014]*",
    re.IGNORECASE,
)


def _normalize_section_header(h) -> str:
    """Header text normalized for same-table collapse comparisons.

    Strips VLM-captured revision markers like ``**REV**`` / ``**NEW**``
    from the front — the VLM includes the marker in some chunks and
    drops it in others, causing two copies of the same table to look
    like distinct sections during dedup.
    """
    s = _norm(h)
    prev = None
    while s and s != prev:
        prev = s
        s = _SECTION_HEADER_NOISE.sub("", s).strip()
    return s


def _table_fingerprint(t: dict) -> str:
    """page (fallback to section) + header_sig + first_row.

    ``table_classification`` is intentionally excluded: the VLM sometimes
    emits the same table as ``Standard_Table`` in one chunk and
    ``Key_Value_Form`` in the next. Omitting classification here lets
    first-pass dedup collapse those pairs instead of deferring entirely
    to ``_collapse_same_page_duplicates``.
    """
    page = _norm(t.get("visual_page_number"))
    disambig = page if page else _normalize_section_header(t.get("preceding_section_header"))
    header_sig = _norm(_table_header_signature(t))
    first_row = _norm(_table_first_row_text(t))
    return _hash(f"tbl:{disambig}|{header_sig}|{first_row}")


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


_CITE_MARKER_RE = re.compile(r"\[cite:\s*\d+\]", re.IGNORECASE)

# Matches ANY [cite: ...] token including malformed bodies (e.g. "[cite: 世]"
# when the VLM tokenizer drifts and drops a CJK char where a digit should be).
_ANY_CITE_MARKER_RE = re.compile(r"\[cite:([^\]]*)\]", re.IGNORECASE)


def _strip_cite_markers(s: str) -> str:
    """Remove ``[cite: N]`` tokens the VLM injects into verbatim text.

    Cite numbers are scoped per-narrative-entry (numbered from 1 within
    each entry), so two chunks that capture the same section emit the
    same prose with *different* cite numbers embedded throughout. Leaving
    them in would defeat substring / fingerprint comparisons across
    chunks.
    """
    if not s:
        return s
    return _CITE_MARKER_RE.sub("", s)


def _strip_malformed_cite_markers(s: str) -> str:
    """Remove ``[cite: X]`` tokens whose body isn't a plain number.

    Well-formed markers like ``[cite: 3]`` are preserved for downstream
    citation tracking. Malformed markers like ``[cite: 世]`` or
    ``[cite: abc]`` are VLM tokenizer drift (usually a CJK char emitted
    where a digit belongs) and should be stripped from the stored output.
    """
    if not s:
        return s

    def repl(m):
        body = m.group(1).strip()
        return m.group(0) if body.isdigit() else ""

    return _ANY_CITE_MARKER_RE.sub(repl, s)


# Unicode ranges that are almost never legitimate in English-language grant
# admin documents. When the VLM's tokenizer drifts it occasionally substitutes
# a CJK or Cyrillic char where an ASCII one belongs (e.g. "Wis.牌" for
# "Wis. Stats", "8700-世" for "8700-349"). Flag these so they can be
# triaged in potential_issues — don't strip, since legit foreign-language
# quotations are rare but possible.
_EXOTIC_UNICODE_RE = re.compile(
    "["
    "\u0400-\u052f"        # Cyrillic + Cyrillic Supplement
    "\u0531-\u058f"        # Armenian
    "\u0590-\u05ff"        # Hebrew
    "\u0600-\u06ff"        # Arabic
    "\u3000-\u303f"        # CJK Symbols & Punctuation
    "\u3040-\u309f"        # Hiragana
    "\u30a0-\u30ff"        # Katakana
    "\u3400-\u4dbf"        # CJK Unified Ideographs Ext A
    "\u4e00-\u9fff"        # CJK Unified Ideographs
    "\uac00-\ud7af"        # Hangul
    "\uff00-\uffef"        # Halfwidth/Fullwidth forms
    "]"
)


def _find_exotic_unicode(s) -> str:
    """Return a sample exotic-unicode run if found, else empty string."""
    if not isinstance(s, str) or not s:
        return ""
    m = _EXOTIC_UNICODE_RE.search(s)
    return m.group(0) if m else ""


def _strip_exotic_unicode(s) -> str:
    """Drop exotic-unicode chars (CJK/Cyrillic/etc) from a string.

    These chars in English-language grant docs are VLM tokenizer drift
    — the model emitted a Han ideograph or Cyrillic letter where an
    ASCII char belonged (e.g. "1.\u724c" for "1.91", "highest_\u4e16ducation"
    for "highest_education"). Removing the glitched chars typically
    leaves the surrounding context intact and readable. We strip the
    char rather than try to guess what was meant; the lint pass still
    flags so a human can spot-check.
    """
    if not isinstance(s, str) or not s:
        return s
    return _EXOTIC_UNICODE_RE.sub("", s)


def _strip_exotic_unicode_inplace(items: list[dict]) -> None:
    """Walk every string value in a list of dicts and strip drift chars.

    Mutates each item in place. Skips dict KEYS (those are schema names;
    if they're glitched we want the lint to surface that — silently
    dropping a char from a key would change the field name and silently
    drop data downstream).
    """
    def _scrub(v):
        if isinstance(v, str):
            return _strip_exotic_unicode(v)
        if isinstance(v, list):
            return [_scrub(x) for x in v]
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        return v
    for it in items or []:
        if not isinstance(it, dict):
            continue
        for k in list(it.keys()):
            it[k] = _scrub(it[k])


def _walk_strings(value):
    """Yield every string leaf inside nested dicts/lists (for scanning)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_strings(v)


def _walk_dict_keys(value):
    """Yield every dict KEY found inside nested dicts/lists.

    Separate from ``_walk_strings`` because the VLM sometimes corrupts
    JSON keys themselves with CJK drift (e.g. emitting
    ``"visual_page世_number"`` instead of ``"visual_page_number"``). A
    corrupted key won't be caught by a value-only scan.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                yield k
            yield from _walk_dict_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_dict_keys(v)


def _narrative_fingerprint(n: dict) -> str:
    page = _norm(n.get("visual_page_number"))
    disambig = page if page else _norm(n.get("preceding_section_header"))
    header = _norm(n.get("prompt_or_header"))
    # Strip cite markers before hashing the head: two copies of the same
    # section have different cite numbers (per-entry scope) that would
    # otherwise make the hash diverge.
    head_chars = _norm(_strip_cite_markers(n.get("verbatim_text", "")))[:120]
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
    # "12/142", "12 | 142", "12 of 142", "12 — 142" (en/em-dash), and
    # "12\u201c142" (smart-quote slipped in because the extraction prompt
    # tells the VLM to use U+201C/U+201D inside string values and it
    # misapplied that rule to a page-number footer).
    re.compile(
        r"^(\d+)\s*(?:[|/,\-\u2013\u2014\u2018\u2019\u201c\u201d]|of)\s*\d+$",
        re.IGNORECASE,
    ),
    # Catch-all: leading digits, a run of non-word chars, trailing
    # digits, end. Covers exotic footer decoration we haven't named
    # explicitly. Won't match appendix-style "A-5" since those don't
    # start with a digit.
    re.compile(r"^(\d+)[^\w]+\d+$"),
]


def normalize_visual_page_number(raw, chunk_page_range=None):
    """Strip common footer artifacts from a printed-page-number string.

    The VLM sometimes captures footer decoration verbatim (``"50 | Page"``
    for PDF page 51, ``"Page 12 of 142"``, etc.) because the extraction
    prompt asks for the value to be verbatim. That breaks cross-chunk
    dedupe and page-sort ordering, so we normalize anything that matches
    a known footer pattern to just the page identifier.

    Sub-form collision guard: when ``chunk_page_range=(page_start,
    page_end)`` is supplied (0-based inclusive start, exclusive end — the
    shape used by ``chunk_page_ranges``) AND the normalized value is a
    plain int falling outside the 1-based printed range ``[page_start+1,
    page_end]``, the function returns ``None``. This catches the common
    failure where the VLM reads a sub-form's "Page X of N" footer (or a
    bare "X") from an attached document inside a larger award packet and
    emits it as the outer doc's ``visual_page_number``; without this
    guard, those sub-form pages collide with the enclosing doc's real
    pages during dedup and sort out-of-order. Non-numeric identifiers
    (roman numerals, "A-5", etc.) bypass the range check.
    """
    if raw is None or raw == "":
        return raw
    s = str(raw).strip()
    normalized = s
    for pat in _PAGE_NUMBER_FOOTER_PATTERNS:
        m = pat.match(s)
        if m:
            normalized = m.group(1)
            break
    if chunk_page_range is not None:
        start, end = chunk_page_range
        if start is not None and end is not None:
            try:
                n = int(normalized)
            except (TypeError, ValueError):
                return normalized
            if n < start + 1 or n > end:
                return None
    return normalized


def _normalize_page_numbers_inplace(items: list[dict], chunk_page_range=None) -> None:
    for it in items or []:
        page = it.get("visual_page_number")
        normed = normalize_visual_page_number(page, chunk_page_range)
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


def _table_content_tokens(t: dict) -> set[str]:
    """Bag of normalized cell tokens — classification-agnostic.

    Used for a secondary dedup pass that catches same-page duplicates the
    fingerprint misses when the VLM emits the same table under different
    classifications (e.g. Key_Value_Form in one chunk, Standard_Table in
    the next).
    """
    tokens: set[str] = set()
    data = t.get("table_data")
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                for k, v in row.items():
                    tokens.add(_norm(k))
                    tokens.add(_norm(v))
            elif isinstance(row, list):
                for c in row:
                    tokens.add(_norm(c))
    elif isinstance(data, dict):
        for k, v in data.items():
            tokens.add(_norm(k))
            tokens.add(_norm(v))
    tokens.discard("")
    return tokens


_CROSS_PAGE_GAP = 3
_CROSS_PAGE_MIN_TOKENS = 20
_CROSS_PAGE_COVERAGE = 0.95


def _page_as_int(raw) -> int | None:
    """Return the numeric part of a visual_page_number, or None if the
    identifier isn't numeric (e.g. 'A-5', 'iii'). Used for page-gap math
    only; non-numeric pages just skip the cross-page rule.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.match(r"^(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _collapse_same_page_duplicates(tables: list[dict]) -> list[dict]:
    """Drop near-duplicate tables whose content is substantially covered
    by a kept neighbor. Runs after sort; keeps the larger/non-fragment entry.

    Three-tier rule (checked against each prior kept table in reverse):
      1. Same page + headers agree (after ``_normalize_section_header``,
         or one is empty) + ≥80% token coverage of the smaller → collapse.
      2. Same page + ≥95% coverage + ≥10 tokens → collapse. Covers the
         common case where the VLM picks different ``preceding_section_header``
         text across overlapping chunks for the same table.
      3. Different page (gap ≤ 3, both numeric) + ≥95% coverage + ≥20
         tokens → collapse. Covers the case where a single logical table
         straddles a chunk boundary and the VLM records a different page
         number for each copy (chunk A emits it labeled p.108, chunk B
         labels the overlapping re-extraction p.112). The higher token
         floor and stricter coverage protect legitimately-different
         same-shape tables on nearby pages.

    Walk stops when the page gap exceeds the threshold (no reason to keep
    searching further back in the doc once we're past the window).

    The ≥10/≥20-token floors protect same-shape distinct tables (Year-1
    vs Year-2 budgets, repeated ranking rubrics) from collapsing; in
    practice those have diverging cell values that drop coverage below
    the threshold.
    """
    if not tables:
        return tables
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for t in tables:
        page = t.get("visual_page_number")
        page_n = _page_as_int(page)
        header = _normalize_section_header(t.get("preceding_section_header"))
        tokens = _table_content_tokens(t)
        absorbed = False
        for i in range(len(kept) - 1, -1, -1):
            other_page = kept[i].get("visual_page_number")
            other_page_n = _page_as_int(other_page)
            same_page = other_page == page
            # Page gap bail-out: once we're more than _CROSS_PAGE_GAP
            # pages back and both pages are numeric, stop searching.
            if not same_page and page_n is not None and other_page_n is not None:
                if abs(page_n - other_page_n) > _CROSS_PAGE_GAP:
                    break
            other_header = _normalize_section_header(kept[i].get("preceding_section_header"))
            other_tokens = kept_tokens[i]
            if not tokens or not other_tokens:
                continue
            overlap = len(tokens & other_tokens)
            small = min(len(tokens), len(other_tokens))
            if small == 0:
                continue
            coverage = overlap / small
            header_match = not header or not other_header or header == other_header
            if same_page:
                eligible = (
                    (header_match and coverage >= 0.8)
                    or (coverage >= 0.95 and small >= 10)
                )
            else:
                # Cross-page: require both pages to be numeric (gap
                # comparable) and apply the stricter threshold.
                if page_n is None or other_page_n is None:
                    continue
                eligible = (
                    coverage >= _CROSS_PAGE_COVERAGE
                    and small >= _CROSS_PAGE_MIN_TOKENS
                )
            if not eligible:
                continue
            winner = _pick_more_complete(kept[i], t, _table_size)
            if winner is t:
                kept[i] = t
                kept_tokens[i] = tokens
            absorbed = True
            break
        if not absorbed:
            kept.append(t)
            kept_tokens.append(tokens)
    return kept


def _table_column_signature(table: dict) -> tuple | None:
    """Normalized signature of a Standard_Table's column-header keys.

    Returns the sorted tuple of normalized row keys (their union across
    all rows, to tolerate minor row-shape variance the VLM occasionally
    produces). Returns None for non-Standard_Table inputs or empty rows.
    Used by the supertable collapser to detect adjacent tables that
    share the same column shape.
    """
    if table.get("table_classification") != "Standard_Table":
        return None
    rows = table.get("table_data")
    if not isinstance(rows, list) or not rows:
        return None
    keys: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            keys.update(row.keys())
    if not keys:
        return None
    return tuple(sorted(_norm(k) for k in keys))


def _rows_introduce_new_content(a: dict, b: dict) -> bool:
    """True when ``b`` has at least one row that's not already in ``a``.

    Used to block supertable merges when two same-shape tables have
    identical rows — that's a duplicate pattern, not a supertable.
    Real supertable subtables each carry distinct rows (different
    protocols per page, different categories per section, etc.).
    """
    a_rows = a.get("table_data") or []
    b_rows = b.get("table_data") or []
    if not isinstance(a_rows, list) or not isinstance(b_rows, list) or not b_rows:
        return False
    a_sigs = {
        json.dumps(r, sort_keys=True, ensure_ascii=False)
        for r in a_rows if isinstance(r, dict)
    }
    for r in b_rows:
        if isinstance(r, dict):
            if json.dumps(r, sort_keys=True, ensure_ascii=False) not in a_sigs:
                return True
    return False


def _supertable_mergeable(
    a: dict, b: dict, a_trailing_page: int | None, max_page_gap: int = 2,
) -> bool:
    """Predicate: should b's rows be appended into a as part of a supertable?

    Triggers when two adjacent Standard_Tables share an identical
    column-key signature AND sit within ``max_page_gap`` pages of each
    other (``visual_page_number`` parsed as int) AND b contributes at
    least one row that isn't already in a. ``a_trailing_page`` is the
    last absorbed page so a run like pp.80..84 keeps advancing the
    comparison baseline. Continuation-flagged tables are excluded.
    Tables with a missing or non-numeric ``visual_page_number`` on
    either side are conservatively left unmerged.
    """
    if a.get("continues_to_next_chunk") or b.get("continues_from_previous_chunk"):
        return False
    sig_a = _table_column_signature(a)
    sig_b = _table_column_signature(b)
    if sig_a is None or sig_a != sig_b:
        return False
    pa = a_trailing_page if a_trailing_page is not None else _page_as_int(a.get("visual_page_number"))
    pb = _page_as_int(b.get("visual_page_number"))
    if pa is None or pb is None:
        return False
    if not (0 <= (pb - pa) <= max_page_gap):
        return False
    return _rows_introduce_new_content(a, b)


def _collapse_supertable_runs(tables: list[dict]) -> list[dict]:
    """Merge adjacent Standard_Tables that look like one logical "supertable".

    Common pattern in long grant-admin docs: a section heading is
    followed by several subtables across consecutive pages, each with
    identical column headers but different ``preceding_section_header``
    values (the subtable subtitle). The VLM correctly emits each
    subtable separately; this pass folds them into a single table whose
    ``preceding_section_header`` is the supertable's heading (the first
    subtable's). Rows are appended in page order via ``_concat_tables``.

    Assumes the input is already page-sorted. The ``trailing`` list
    tracks the last absorbed page so a run at pp.80, 81, 82, 83, 84
    keeps advancing rather than re-comparing back to p.80.
    """
    if not tables:
        return tables
    out: list[dict] = []
    trailing: list[int | None] = []
    for t in tables:
        tp = _page_as_int(t.get("visual_page_number"))
        if out and _supertable_mergeable(out[-1], t, trailing[-1]):
            out[-1] = _concat_tables(out[-1], t)
            if tp is not None:
                trailing[-1] = tp
        else:
            out.append(t)
            trailing.append(tp)
    return out


def _reclassify_self_keyed_standard_table(table: dict) -> dict:
    """Convert ``Standard_Table`` to ``Literal_Grid`` when rows are self-keyed.

    The VLM occasionally emits a tabular section as a Standard_Table where
    every row is a dict with ``key == value`` for every entry — the model
    has stuffed cell text into both positions because there are no real
    column headers. These rows carry no schema information and would
    fail the homogeneous-keys check downstream. Convert to Literal_Grid
    (a 2D string array, one inner array per row) so the layout survives
    without forcing a phantom schema.

    Triggers only when ALL rows are dicts and EVERY cell satisfies
    ``str(key).strip() == str(value).strip()`` (after collapsing
    whitespace). One real header cell is enough to leave the table
    alone — better to keep an awkward Standard_Table than to throw away
    a real schema.
    """
    if table.get("table_classification") != "Standard_Table":
        return table
    rows = table.get("table_data")
    if not isinstance(rows, list) or not rows:
        return table
    for row in rows:
        if not isinstance(row, dict) or not row:
            return table
        for k, v in row.items():
            if _norm(k) != _norm(v):
                return table
    grid = [list(row.keys()) for row in rows]
    out = dict(table)
    out["table_classification"] = "Literal_Grid"
    out["table_data"] = grid
    return out


def _reclassify_array_valued_standard_table(table: dict) -> dict:
    """Convert ``Standard_Table`` to ``Literal_Grid`` when row values are arrays.

    Pattern seen in the WI DNR doc's APPENDIX K (List of Forms): the
    section title became the row key and the bulleted list of forms
    became a single array value, e.g.
    ``{"Application forms": ["Form A", "Form B", ...]}``. That isn't
    really a table row — it's a section header followed by a list.
    Convert to Literal_Grid where each form occupies its own row,
    preserving the section header as the leading cell of the first row
    in each group.
    """
    if table.get("table_classification") != "Standard_Table":
        return table
    rows = table.get("table_data")
    if not isinstance(rows, list) or not rows:
        return table
    for row in rows:
        if not isinstance(row, dict) or not row:
            return table
        for v in row.values():
            if not isinstance(v, list):
                return table
    grid: list[list[str]] = []
    for row in rows:
        for k, items in row.items():
            grid.append([str(k)])
            for item in items:
                grid.append([str(item)])
    out = dict(table)
    out["table_classification"] = "Literal_Grid"
    out["table_data"] = grid
    return out


def _dedupe_table_rows(table: dict) -> dict:
    """Drop fully-duplicate rows within a single table's ``table_data``.

    Applies to ``Standard_Table`` (list of dicts) and ``Literal_Grid``
    (list of lists). ``Key_Value_Form`` is a single dict, so row
    duplication is impossible. Row order is preserved; first occurrence
    of each row wins. Catches VLM runaway where a long table (e.g. a
    priority-waterbody ranking list spanning multiple pages) repeats
    rows after the model loses its place.
    """
    data = table.get("table_data")
    if not isinstance(data, list) or not data:
        return table
    seen: set = set()
    deduped: list = []
    for row in data:
        if isinstance(row, dict):
            key = tuple(sorted((str(k), str(v)) for k, v in row.items()))
        elif isinstance(row, list):
            key = ("_list",) + tuple(str(c) for c in row)
        else:
            key = ("_raw", str(row))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    if len(deduped) == len(data):
        return table
    out = dict(table)
    out["table_data"] = deduped
    return out


def _collapse_narrative_same_page_substrings(narratives: list[dict]) -> list[dict]:
    """Drop narratives whose ``verbatim_text`` is contained in another's text.

    Overlapping chunks often produce two near-identical entries for the
    same section: one prepends the section heading, the other picks up
    the tail of the previous paragraph, etc. ``_narrative_fingerprint``
    hashes the first 120 chars, so those leading-text variants survive
    as "distinct". A substring containment pass after dedup collapses
    the pair, keeping the longer text.

    Narratives no longer carry page metadata, so we run a single
    containment pass across the whole list (O(n²) in narrative count,
    fine for typical docs with up to low thousands of narratives).
    """
    if not narratives:
        return narratives
    drop: set[int] = set()

    # Strip cite markers before comparison: two copies of the same
    # section emit different cite numbers (per-entry scope) that
    # would otherwise break exact substring containment.
    items = [
        (i, _norm(_strip_cite_markers(n.get("verbatim_text", ""))))
        for i, n in enumerate(narratives)
    ]
    # Sort by length desc so shorter candidates are checked against
    # longer ones first.
    items.sort(key=lambda p: -len(p[1]))
    for j, (a_i, a_t) in enumerate(items):
        if a_i in drop or not a_t:
            continue
        for b_i, b_t in items[j + 1:]:
            if b_i in drop or not b_t:
                continue
            if len(a_t) >= len(b_t) and b_t in a_t:
                drop.add(b_i)

    if not drop:
        return narratives
    return [n for i, n in enumerate(narratives) if i not in drop]


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
# Deterministic lint (potential_issues)
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

    # Page values with non-ASCII characters that survived normalization —
    # usually smart-quote separators or other footer decoration the
    # normalizer patterns don't catch yet.
    for field in ("tables", "narrative_responses", "stakeholders", "addresses"):
        for i, item in enumerate(merged.get(field) or []):
            page = item.get("visual_page_number")
            if page is None:
                continue
            s = str(page)
            if any(ord(c) > 127 for c in s):
                notes.append(
                    f"{field}[{i}]: visual_page_number {s!r} contains "
                    f"non-ASCII characters — normalizer may need another "
                    f"footer pattern"
                )

    # Standard_Table entries with inconsistent key sets across rows.
    # The VLM sometimes misreads a column-span header as an extra row,
    # producing one row with a different schema than the rest.
    for i, t in enumerate(merged.get("tables") or []):
        if t.get("table_classification") != "Standard_Table":
            continue
        data = t.get("table_data")
        if not isinstance(data, list) or len(data) < 2:
            continue
        key_sets = [frozenset(r.keys()) for r in data if isinstance(r, dict)]
        if len(key_sets) < 2:
            continue
        counts: dict = {}
        for ks in key_sets:
            counts[ks] = counts.get(ks, 0) + 1
        if len(counts) == 1:
            continue
        most_common = max(counts, key=counts.get)
        odd = sum(c for ks, c in counts.items() if ks != most_common)
        page = t.get("visual_page_number") or "?"
        notes.append(
            f"tables[{i}] on page {page}: Standard_Table has "
            f"{odd}/{len(key_sets)} row(s) with inconsistent column keys "
            f"(VLM may have misread a header cell)"
        )

    # VLM tokenizer drift: CJK / Cyrillic / other non-Latin chars
    # appearing inside English-language grant text almost always indicate
    # a glitched token (e.g. "Wis.牌" where "Wis. Stats" was expected,
    # "[cite: 世]" where a digit was expected). Flag but don't auto-strip
    # — legit foreign-language quotes are rare but possible.
    #
    # Scans both cell VALUES and dict KEYS — the VLM occasionally emits
    # corrupted keys like "visual_page世_number" that a value-only scan
    # misses. A corrupted key is the stronger signal of drift since keys
    # should be stable schema names.
    exotic_hits: list[str] = []
    for i, n in enumerate(merged.get("narrative_responses") or []):
        ex = _find_exotic_unicode(n.get("verbatim_text") or "")
        if ex:
            page = n.get("visual_page_number") or "?"
            exotic_hits.append(
                f"narrative_responses[{i}] on page {page}: contains "
                f"exotic unicode {ex!r} (VLM token drift)"
            )
        # Key scan on the narrative entry itself — detects corrupted
        # field names like "visual_page世_number".
        for k in _walk_dict_keys(n):
            kex = _find_exotic_unicode(k)
            if kex:
                exotic_hits.append(
                    f"narrative_responses[{i}]: dict key {k!r} contains "
                    f"exotic unicode {kex!r} (VLM corrupted a JSON key "
                    f"— structural issue, worse than a value glitch)"
                )
                break
    for i, t in enumerate(merged.get("tables") or []):
        data = t.get("table_data")
        found = ""
        for s in _walk_strings(data):
            ex = _find_exotic_unicode(s)
            if ex:
                found = ex
                break
        if found:
            page = t.get("visual_page_number") or "?"
            exotic_hits.append(
                f"tables[{i}] on page {page}: cell contains exotic unicode "
                f"{found!r} (VLM token drift)"
            )
        # Key scan on the table entry (including nested row dicts in
        # table_data) — catches both outer-level corrupted keys and
        # corrupted column-header names.
        for k in _walk_dict_keys(t):
            kex = _find_exotic_unicode(k)
            if kex:
                page = t.get("visual_page_number") or "?"
                exotic_hits.append(
                    f"tables[{i}] on page {page}: dict key {k!r} contains "
                    f"exotic unicode {kex!r} (VLM corrupted a JSON key "
                    f"— structural issue, worse than a value glitch)"
                )
                break
    # Cap at 10 to avoid drowning the lint output on pathological docs.
    for note in exotic_hits[:10]:
        notes.append(note)
    if len(exotic_hits) > 10:
        notes.append(
            f"...{len(exotic_hits) - 10} additional exotic-unicode hits "
            f"suppressed"
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


def _strip_malformed_cites_from_narratives(narratives: list[dict]) -> None:
    """Rewrite verbatim_text to drop cite markers whose body isn't numeric.

    Mutates in place. Legitimate ``[cite: 3]`` markers are preserved; only
    VLM-glitched ones like ``[cite: 世]`` are removed.
    """
    for n in narratives or []:
        t = n.get("verbatim_text")
        if isinstance(t, str) and "[cite:" in t:
            cleaned = _strip_malformed_cite_markers(t)
            if cleaned != t:
                n["verbatim_text"] = cleaned


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

    Lint notes (``potential_issues``) are always computed at the end.
    Tables carry a ``visual_page_number`` (the page number printed on
    the page) and are sorted by it. Stakeholders, addresses, and
    narrative responses carry no page metadata and preserve the order
    they were emitted in by the VLM.
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
    # When full chunk records are available, each chunk's (page_start,
    # page_end) range is passed through so the normalizer can also null
    # out sub-form misreads (e.g., a bare "3" emitted for an attached
    # form when the chunk actually covers PDF pages 31-39).
    chunk_ranges: list = []
    if records:
        for r in records:
            ps, pe = r.get("page_start"), r.get("page_end")
            chunk_ranges.append((ps, pe) if ps is not None and pe is not None else None)
    else:
        chunk_ranges = [None] * len(extracted_list)
    # visual_page_number lives on TABLES ONLY. The VLM prompt no longer
    # asks for any page metadata on narratives / stakeholders / addresses,
    # but we defensively strip any stray copies (older cached outputs, or
    # a drifting VLM) so downstream lookups stay consistent. pdf_page_index
    # and chunk_relative_page_index are no longer emitted anywhere — strip
    # them wholesale to clean up any cached chunks produced by an older
    # pipeline version.
    for e, rng in zip(extracted_list, chunk_ranges):
        _normalize_page_numbers_inplace(e.get("tables") or [], rng)
        for field in ("tables", "narrative_responses", "stakeholders", "addresses"):
            _strip_exotic_unicode_inplace(e.get(field) or [])
            for it in e.get(field) or []:
                if isinstance(it, dict):
                    it.pop("pdf_page_index", None)
                    it.pop("chunk_relative_page_index", None)
        for field in ("narrative_responses", "stakeholders", "addresses"):
            for it in e.get(field) or []:
                if isinstance(it, dict):
                    it.pop("visual_page_number", None)

    if len(extracted_list) == 1:
        # Single chunk: no cross-chunk stitch needed, but still apply
        # within-chunk dedup + page-sort so output shape matches the
        # multi-chunk path (a single VLM call can still emit duplicate
        # stakeholders or out-of-order tables).
        merged = dict(extracted_list[0])
        # NOTE: continuation-flag stripping is deferred until after the
        # supertable collapser runs — that pass uses the flags to skip
        # tables already handled by the cross-chunk stitcher.
        # Strip the per-chunk confidence_narrative from the doc-level view —
        # chunks[].extracted preserves it per-chunk.
        # Doc-level confidence_narrative is filled by pass-2 VLM synthesis
        # (a short summary of chunks[].extracted.confidence_narrative).
        merged["confidence_narrative"] = ""
        merged["experiment"] = _doc_level_experiment(records)
        if extraction_prompt:
            merged["experiment"]["extraction_prompt"] = extraction_prompt
        # Stakeholders / addresses / narratives are kept in source order
        # (no page-based sort) since they no longer carry page metadata.
        merged["stakeholders"] = _finalize_stakeholders(_merge_identity(
            [merged.get("stakeholders") or []],
            _stakeholder_fingerprint,
            empty_fn=_stakeholder_is_empty,
        ))
        merged["addresses"] = _merge_identity(
            [merged.get("addresses") or []],
            _address_fingerprint,
        )
        merged["tables"] = _collapse_supertable_runs([
            _dedupe_table_rows(_reclassify_array_valued_standard_table(_reclassify_self_keyed_standard_table(t)))
            for t in _collapse_same_page_duplicates(sorted(
                (t for t in (merged.get("tables") or []) if not _is_empty_table(t)),
                key=_page_sort_key,
            ))
        ])
        merged["narrative_responses"] = _collapse_narrative_same_page_substrings(
            merged.get("narrative_responses") or []
        )
        _strip_malformed_cites_from_narratives(merged["narrative_responses"])
        _strip_continuation_flags(merged)
        merged["chunks"] = _build_chunks_sidecar(records)
        merged["potential_issues"] = _lint_merged(merged)
        return merged

    experiment = _doc_level_experiment(records)
    if extraction_prompt:
        experiment["extraction_prompt"] = extraction_prompt

    stakeholders = _finalize_stakeholders(_merge_identity(
        [e.get("stakeholders", []) for e in extracted_list],
        _stakeholder_fingerprint,
        empty_fn=_stakeholder_is_empty,
    ))
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
        # Stakeholders / addresses no longer carry page metadata; preserve
        # source (chunk) order rather than sorting by a field that's gone.
        "stakeholders": stakeholders,
        "addresses": addresses,
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
    # NOTE: continuation-flag stripping is deferred until after the
    # supertable collapser runs — that pass uses the flags to skip
    # tables already handled by the cross-chunk stitcher.
    # Drop empty-table entries (VLM sometimes tags a section heading as a
    # Standard_Table with no rows — pure noise). Done after dedupe so any
    # chunk that actually captured rows wins via _pick_more_complete.
    # Dedupe rows within each surviving table to clean up runaway-loop
    # repetition the VLM sometimes emits in long tabular sections.
    merged["tables"] = _collapse_supertable_runs([
        _dedupe_table_rows(_reclassify_self_keyed_standard_table(t))
        for t in _collapse_same_page_duplicates(sorted(
            (t for t in (merged.get("tables") or []) if not _is_empty_table(t)),
            key=_page_sort_key,
        ))
    ])
    merged["narrative_responses"] = _collapse_narrative_same_page_substrings(
        merged.get("narrative_responses") or []
    )
    _strip_malformed_cites_from_narratives(merged["narrative_responses"])
    _strip_continuation_flags(merged)
    merged["potential_issues"] = _lint_merged(merged)
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
