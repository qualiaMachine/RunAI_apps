"""Standalone tests for scripts.merge.merge_chunks.

Run:
    python -m ocr_app.tests.test_merge
    # or
    python ocr_app/tests/test_merge.py

Tests exercise:
    1. Single-chunk pass-through
    2. Identity dedupe (stakeholders / addresses)
    3. Span dedupe (full-copy wins over fragment)
    4. Continuation stitch (two-chunk table longer than overlap)
    5. Doc-level aggregation (tags, signature, confidence)
    6. Section-header disambiguation (same header, different sections)
    7. Fragment chain across 3 chunks

Each test prints PASS/FAIL to stdout and the script exits non-zero if
any test fails. No pytest dependency.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Make ocr_app/scripts importable when run directly
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from scripts.merge import (  # noqa: E402
    merge_chunks,
    merge_chunks_json,
    normalize_visual_page_number,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic chunks
# ---------------------------------------------------------------------------

def _chunk(**overrides) -> dict:
    base = {
        "confidence_percentage": 90,
        "confidence_narrative": "",
        "has_annotation": False,
        "has_watermark": False,
        "signature_lines": {"has_signature_line": False, "has_valid_signature": False},
        "document_tags": [],
        "one_sentence_summary": "",
        "document_details": {
            "application_id": "", "application_type": "", "title": "",
            "requested_amount": None, "completed_date": "", "sub_document_type": "",
        },
        "stakeholders": [],
        "addresses": [],
        "tables": [],
        "narrative_responses": [],
        "other_metadata": {},
    }
    base.update(overrides)
    return base


def _table(classification="Standard_Table", header="Year 1 Budget",
           rows=None, cfp=False, ctn=False,
           visual_page_number=None):
    return {
        "preceding_section_header": header,
        "visual_page_number": visual_page_number,
        "table_classification": classification,
        "continues_from_previous_chunk": cfp,
        "continues_to_next_chunk": ctn,
        "table_data": rows if rows is not None else [],
    }


def _narr(header="General Body Text", section="", text="",
          cfp=False, ctn=False):
    return {
        "preceding_section_header": section,
        "prompt_or_header": header,
        "continues_from_previous_chunk": cfp,
        "continues_to_next_chunk": ctn,
        "verbatim_text": text,
    }


def _stake(email="", first="", last="", full="", inst="", role="Unknown",
           position="", dept="", phone=""):
    return {
        "context_snippet": "",
        "stakeholder_role": role,
        "full_name": full,
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": phone,
        "institution": inst,
        "department": dept,
        "position_title": position,
        "highest_education": "",
        "raw_stakeholder_text": "",
    }


def _addr(line1="", postal="", city="", addressee=""):
    return {
        "context_snippet": "",
        "addressee": addressee,
        "care_of": None,
        "address_line1": line1,
        "address_line2": "",
        "city": city,
        "state_province": "",
        "postal_code": postal,
        "country": "",
        "stakeholder_type": "Unknown",
        "raw_address_text": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_chunk_passthrough():
    c = _chunk(
        stakeholders=[_stake(email="pi@u.edu", last="Smith")],
        tables=[_table(rows=[{"a": 1, "b": 2}])],
        document_tags=["IRB"],
    )
    out = merge_chunks([c])
    assert len(out["stakeholders"]) == 1
    assert len(out["tables"]) == 1
    # Continuation flags should be stripped from the final output
    assert "continues_from_previous_chunk" not in out["tables"][0]
    assert "continues_to_next_chunk" not in out["tables"][0]
    assert out["document_tags"] == ["IRB"]


def test_identity_dedupe_stakeholders():
    c1 = _chunk(stakeholders=[
        _stake(email="pi@u.edu", first="Jane", last="Smith", inst="U"),
    ])
    c2 = _chunk(stakeholders=[
        # Same person, slightly different info — should collapse
        _stake(email="pi@u.edu", full="Jane M. Smith", inst="University of U"),
    ])
    c3 = _chunk(stakeholders=[
        # Different person
        _stake(email="copi@u.edu", first="John", last="Doe"),
    ])
    out = merge_chunks([c1, c2, c3])
    assert len(out["stakeholders"]) == 2, f"expected 2 got {len(out['stakeholders'])}"
    # The merged Smith should have the longer institution string
    smiths = [s for s in out["stakeholders"] if s["email"] == "pi@u.edu"]
    assert len(smiths) == 1
    assert "University" in smiths[0]["institution"]


def test_identity_dedupe_addresses():
    c1 = _chunk(addresses=[_addr(line1="123 Main St", postal="53706", city="Madison")])
    c2 = _chunk(addresses=[_addr(line1="123 Main St", postal="53706", city="Madison")])
    c3 = _chunk(addresses=[_addr(line1="500 Lincoln Dr", postal="53706", city="Madison")])
    out = merge_chunks([c1, c2, c3])
    assert len(out["addresses"]) == 2


def test_span_dedupe_collapses_full_copies_in_overlap():
    # Realistic case: both chunks overlap on the pages containing the
    # full table, so both emit the SAME full copy. Dedupe should collapse
    # them via fingerprint (matching first-row hash + header).
    full_rows = [{"cat": "Personnel", "amt": 100000},
                 {"cat": "Equipment", "amt": 50000},
                 {"cat": "Travel", "amt": 10000}]
    c1 = _chunk(tables=[_table(rows=full_rows, header="Year 1 Budget")])
    c2 = _chunk(tables=[_table(rows=full_rows, header="Year 1 Budget")])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1, f"dedupe failed: {len(out['tables'])} tables"
    assert len(out["tables"][0]["table_data"]) == 3


def test_section_header_disambiguates_same_shape_tables():
    # Y1 and Y2 budgets have the same structure — must not collapse.
    row_shape = [{"cat": "Personnel", "amt": 100}, {"cat": "Equipment", "amt": 50}]
    c1 = _chunk(tables=[
        _table(rows=row_shape, header="Year 1 Budget"),
        _table(rows=row_shape, header="Year 2 Budget"),
    ])
    out = merge_chunks([c1])
    assert len(out["tables"]) == 2, f"expected Y1+Y2, got {len(out['tables'])}"


def test_continuation_stitch_across_chunks():
    # Long table: chunk 1 sees rows 1-3 (ends with ctn), chunk 2 sees
    # rows 4-6 (starts with cfp). With no dedupe overlap, merge must
    # stitch them.
    c1 = _chunk(tables=[_table(
        rows=[{"r": 1}, {"r": 2}, {"r": 3}],
        header="Long Budget",
        ctn=True,
    )])
    c2 = _chunk(tables=[_table(
        rows=[{"r": 4}, {"r": 5}, {"r": 6}],
        header="Long Budget",
        cfp=True,
    )])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1, f"stitch failed, got {len(out['tables'])} tables"
    rows = out["tables"][0]["table_data"]
    assert [r["r"] for r in rows] == [1, 2, 3, 4, 5, 6], f"rows: {rows}"


def test_three_chunk_fragment_chain():
    # A table spans chunks 1, 2, 3 with no overlap
    c1 = _chunk(tables=[_table(rows=[{"r": 1}], header="Publications", ctn=True)])
    c2 = _chunk(tables=[_table(rows=[{"r": 2}], header="Publications",
                               cfp=True, ctn=True)])
    c3 = _chunk(tables=[_table(rows=[{"r": 3}], header="Publications", cfp=True)])
    out = merge_chunks([c1, c2, c3])
    assert len(out["tables"]) == 1
    rows = out["tables"][0]["table_data"]
    assert [r["r"] for r in rows] == [1, 2, 3], f"rows: {rows}"


def test_narrative_stitch():
    c1 = _chunk(narrative_responses=[_narr(
        header="Specific Aims",
        text="Our lab studies the role of",
        ctn=True,
    )])
    c2 = _chunk(narrative_responses=[_narr(
        header="Specific Aims",
        text="protein folding in neurodegeneration.",
        cfp=True,
    )])
    out = merge_chunks([c1, c2])
    assert len(out["narrative_responses"]) == 1
    txt = out["narrative_responses"][0]["verbatim_text"]
    assert "role of" in txt and "protein folding" in txt, f"text: {txt!r}"


def test_doc_level_aggregation():
    c1 = _chunk(
        confidence_percentage=90,
        has_annotation=True,
        has_watermark=False,
        signature_lines={"has_signature_line": False, "has_valid_signature": False},
        document_tags=["IRB", "IACUC"],
        one_sentence_summary="Short.",
    )
    c2 = _chunk(
        confidence_percentage=80,
        has_annotation=False,
        has_watermark=True,
        signature_lines={"has_signature_line": True, "has_valid_signature": True},
        document_tags=["IRB", "Biosafety"],
        one_sentence_summary="A much longer descriptive summary of the document.",
    )
    out = merge_chunks([c1, c2])
    assert out["has_annotation"] is True
    assert out["has_watermark"] is True
    assert out["signature_lines"]["has_signature_line"] is True
    assert out["signature_lines"]["has_valid_signature"] is True
    # Tags union, order-preserving
    assert set(out["document_tags"]) == {"IRB", "IACUC", "Biosafety"}
    # Confidence averaged
    assert out["confidence_percentage"] == 85.0
    # Top-level summary and confidence_narrative are blank — pass 2 VLM
    # synthesis fills both later, avoiding a concat of 20+ chunk narratives.
    assert out["one_sentence_summary"] == ""
    assert out["confidence_narrative"] == ""


def test_doc_details_null_coalesce():
    c1 = _chunk(document_details={
        "application_id": "AWD-001", "title": "", "requested_amount": None,
        "application_type": "", "completed_date": "", "sub_document_type": "",
    })
    c2 = _chunk(document_details={
        "application_id": "AWD-001", "title": "Grant A", "requested_amount": 500000,
        "application_type": "New", "completed_date": "2025-01-01", "sub_document_type": "",
    })
    out = merge_chunks([c1, c2])
    dd = out["document_details"]
    assert dd["application_id"] == "AWD-001"
    assert dd["title"] == "Grant A"
    assert dd["requested_amount"] == 500000
    assert dd["application_type"] == "New"


def test_merge_chunks_json_handles_parse_errors():
    good = json.dumps(_chunk(document_tags=["A"]))
    bad = "{not valid json at all"
    out = merge_chunks_json([good, bad])
    assert out["document_tags"] == ["A"]
    assert "merge_errors" in out["other_metadata"]
    assert len(out["other_metadata"]["merge_errors"]) == 1


def test_visual_page_number_survives_dedupe_and_stitch():
    # Dedupe: two chunks both see the full table on page 7, keep visual_page_number
    rows = [{"r": 1}, {"r": 2}, {"r": 3}]
    c1 = _chunk(tables=[_table(rows=rows, header="Year 1 Budget", visual_page_number="7")])
    c2 = _chunk(tables=[_table(rows=rows, header="Year 1 Budget", visual_page_number="7")])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1
    assert out["tables"][0]["visual_page_number"] == "7"

    # Stitch: long table spans chunks — keep the START visual_page_number
    c3 = _chunk(tables=[_table(rows=[{"r": 1}, {"r": 2}],
                               header="Publications",
                               visual_page_number="10", ctn=True)])
    c4 = _chunk(tables=[_table(rows=[{"r": 3}, {"r": 4}],
                               header="Publications",
                               visual_page_number="15", cfp=True)])
    out = merge_chunks([c3, c4])
    assert len(out["tables"]) == 1
    assert out["tables"][0]["visual_page_number"] == "10", \
        f"expected start visual page 10, got {out['tables'][0]['visual_page_number']}"


def test_stakeholder_role_dedupe_unnamed():
    # Real-world failure mode: a grant guide mentions "local environmental
    # grant specialist" dozens of times. Each chunk emits a stakeholder
    # with no name/email — just a position_title. These should collapse
    # by role+institution+department rather than stacking up.
    s1 = _stake(position="local environmental grant specialist",
                inst="Wisconsin Department of Natural Resources",
                dept="Bureau of Community Financial Assistance")
    s2 = _stake(position="local environmental grant specialist",
                inst="Wisconsin Department of Natural Resources",
                dept="Bureau of Community Financial Assistance")
    s3 = _stake(position="local environmental grant specialist",
                inst="",
                dept="")
    # A different role at the same institution should stay separate
    s4 = _stake(position="local lake, stream or AIS biologist",
                inst="Wisconsin Department of Natural Resources")
    c1 = _chunk(stakeholders=[s1, s4])
    c2 = _chunk(stakeholders=[s2])
    c3 = _chunk(stakeholders=[s3])
    out = merge_chunks([c1, c2, c3])
    # s1+s2 merge via _merge_identity (identical fingerprint). s3 has
    # empty inst/dept so _merge_identity keeps it separate, but the
    # subsequent _finalize_stakeholders pass recognizes it as a subset
    # of s1 (same position_title, no conflicting fields) and folds it
    # in. s4 has a different position_title and stays distinct.
    assert len(out["stakeholders"]) == 2, (
        f"expected 2 unique roles, got {len(out['stakeholders'])}"
    )


def test_table_dedupe_tolerates_section_header_variance():
    # Real-world failure: the same table appears in two overlapping chunks
    # but the VLM assigns different preceding_section_header values to
    # each copy (e.g. "Funding" vs "Section 1, Table 1. Surface water…").
    # Same visual_page_number + same rows should still dedupe.
    full_rows = [{"Grant": "Surface Water Education", "Cap": "$5,000"},
                 {"Grant": "Lake Planning", "Cap": "$10,000"}]
    c1 = _chunk(tables=[_table(
        rows=full_rows, header="Funding", visual_page_number="6",
    )])
    c2 = _chunk(tables=[_table(
        rows=full_rows,
        header="Section 1, Table 1. Surface water grants and project types",
        visual_page_number="6",
    )])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1, (
        f"expected 1 deduped table, got {len(out['tables'])}"
    )


def test_narrative_dedupe_tolerates_section_header_variance():
    # Same narrative appears in two overlapping chunks, tagged with
    # different preceding_section_header each time. Should still dedupe.
    body = ("Most grants are required by state statute to be cost-shared, "
            "that is, grantees must contribute a percentage of the project’s "
            "total costs.")
    n1 = _narr(header="General Body Text", section="Funding",
               text=body)
    n2 = _narr(header="General Body Text", section="Public access",
               text=body)
    n1["visual_page_number"] = "7"
    n2["visual_page_number"] = "7"
    out = merge_chunks([
        _chunk(narrative_responses=[n1]),
        _chunk(narrative_responses=[n2]),
    ])
    assert len(out["narrative_responses"]) == 1, (
        f"expected 1 deduped narrative, got {len(out['narrative_responses'])}"
    )


def test_same_page_distinct_narratives_not_merged():
    # Guard against over-eager dedupe: two different narratives on the
    # same printed page must stay separate.
    n1 = _narr(header="General Body Text", section="Funding",
               text="Most grants are required by state statute")
    n2 = _narr(header="General Body Text", section="Public access",
               text="To ensure your application will be deemed eligible")
    n1["visual_page_number"] = "7"
    n2["visual_page_number"] = "7"
    out = merge_chunks([_chunk(narrative_responses=[n1, n2])])
    assert len(out["narrative_responses"]) == 2


def test_chunks_sidecar_from_full_records():
    # When full chunk records are passed, merged output carries per-chunk
    # summary + confidence in chunks[] and doc-level experiment settings.
    r1 = {
        "chunk_index": 0, "page_start": 0, "page_end": 9,
        "experiment": {
            "model": "Qwen-VL", "vlm_mode": "remote",
            "max_pages_per_chunk": 9, "chunk_overlap": 3,
            "elapsed_ms": 1234, "timestamp": "2025-04-18T10:00:00",
        },
        "extracted": _chunk(
            one_sentence_summary="Chunk 1 summary.",
            confidence_percentage=95,
            confidence_narrative="chunk 1 narrative",
        ),
    }
    r2 = {
        "chunk_index": 1, "page_start": 7, "page_end": 15,
        "experiment": {
            "model": "Qwen-VL", "vlm_mode": "remote",
            "max_pages_per_chunk": 9, "chunk_overlap": 3,
            "elapsed_ms": 2345, "timestamp": "2025-04-18T10:01:00",
        },
        "extracted": _chunk(
            one_sentence_summary="Chunk 2 summary.",
            confidence_percentage=85,
            confidence_narrative="chunk 2 narrative",
        ),
    }
    out = merge_chunks([r1, r2])
    # Top-level summary left blank for pass-2 VLM synthesis
    assert out["one_sentence_summary"] == ""
    # Doc-level experiment has the settings, not the per-chunk runtime fields
    assert out["experiment"]["model"] == "Qwen-VL"
    assert out["experiment"]["max_pages_per_chunk"] == 9
    assert "elapsed_ms" not in out["experiment"]
    assert "timestamp" not in out["experiment"]
    # Per-chunk sidecar preserves per-chunk data
    assert len(out["chunks"]) == 2
    assert out["chunks"][0]["page_start"] == 0
    assert out["chunks"][0]["extracted"]["one_sentence_summary"] == "Chunk 1 summary."
    assert out["chunks"][0]["extracted"]["confidence_percentage"] == 95
    assert out["chunks"][1]["experiment"]["elapsed_ms"] == 2345


def test_raw_dict_input_yields_empty_sidecar():
    # Backwards-compat path: raw extracted dicts produce a valid merge
    # with experiment={} and chunks=[].
    out = merge_chunks([_chunk(document_tags=["A"]), _chunk(document_tags=["B"])])
    assert out["experiment"] == {}
    assert out["chunks"] == []
    assert set(out["document_tags"]) == {"A", "B"}


def test_empty_tables_are_dropped_in_postprocess():
    # VLM sometimes tags a section header as a Standard_Table with no rows
    # ("Depreciation", "DONATED PROFESSIONAL LABOR"). These are pure noise
    # and should be removed in merge; not even a lint warning needs to fire.
    good = _table(rows=[{"a": 1}], header="Real Table")
    empty = _table(rows=[], header="Depreciation")
    out = merge_chunks([_chunk(tables=[good, empty])])
    assert len(out["tables"]) == 1
    assert out["tables"][0]["preceding_section_header"] == "Real Table"
    # No lint note for the dropped empty table
    assert not any("empty table_data" in n for n in out["potential_issues"])


def test_potential_issues_lint_flags_empty_narratives():
    c = _chunk(narrative_responses=[_narr(header="Aims", text="")])
    out = merge_chunks([c])
    assert any("empty verbatim_text" in n for n in out["potential_issues"])


def test_potential_issues_lint_clean_when_nothing_wrong():
    c = _chunk(
        tables=[_table(rows=[{"a": 1}], header="Budget")],
        narrative_responses=[_narr(header="Aims", text="Real content.")],
    )
    out = merge_chunks([c])
    assert out["potential_issues"] == []


def test_stakeholders_preserve_source_order():
    # Stakeholders no longer carry page metadata; preserve the order the
    # VLM emitted them in (which mirrors top-of-doc-to-bottom reading
    # order from the chunk loop).
    s1 = _stake(position="Reviewer", inst="Agency")
    s2 = _stake(position="Program Officer", inst="Agency")
    s3 = _stake(position="Director", inst="Agency")
    s4 = _stake(position="Signatory", inst="Agency")
    out = merge_chunks([
        _chunk(stakeholders=[s1, s2, s3, s4]),
        _chunk(stakeholders=[]),
    ])
    positions = [s["position_title"] for s in out["stakeholders"]]
    assert positions == ["Reviewer", "Program Officer", "Director", "Signatory"], \
        positions
    # Defensive: page fields should NOT appear on stakeholder output.
    for s in out["stakeholders"]:
        assert "visual_page_number" not in s
        assert "pdf_page_index" not in s


def test_empty_stakeholders_filtered_out():
    # An all-empty stakeholder is dropped rather than passed through.
    empty = _stake()  # every field ""
    named = _stake(last="Smith", first="Jane", inst="U")
    out = merge_chunks([_chunk(stakeholders=[empty, named]), _chunk(stakeholders=[])])
    assert len(out["stakeholders"]) == 1
    assert out["stakeholders"][0]["last_name"] == "Smith"


def test_normalize_visual_page_number():
    # Footer decoration stripped to the page identifier
    assert normalize_visual_page_number("50 | Page") == "50"
    assert normalize_visual_page_number("50|Page") == "50"
    assert normalize_visual_page_number("Page 12") == "12"
    assert normalize_visual_page_number("Page 12 of 142") == "12"
    assert normalize_visual_page_number("12 | 142") == "12"
    assert normalize_visual_page_number("12/142") == "12"
    # Clean identifiers passed through unchanged
    assert normalize_visual_page_number("50") == "50"
    assert normalize_visual_page_number("iii") == "iii"
    assert normalize_visual_page_number("A-5") == "A-5"
    # Null/empty preserved
    assert normalize_visual_page_number(None) is None
    assert normalize_visual_page_number("") == ""


def test_normalize_visual_page_number_smart_quote_and_exotic_separators():
    # The VLM is instructed to use smart quotes inside string values and
    # sometimes misapplies that to page-number footers — "1\u201c126"
    # coming from a printed "1 of 126" footer. Also covers en-dash,
    # em-dash, "of", and the general-purpose catch-all separator.
    assert normalize_visual_page_number("1\u201c126") == "1"
    assert normalize_visual_page_number("1\u201d126") == "1"
    assert normalize_visual_page_number("12\u2013142") == "12"
    assert normalize_visual_page_number("12\u2014142") == "12"
    assert normalize_visual_page_number("12 of 142") == "12"
    assert normalize_visual_page_number("12 -- 142") == "12"
    # Catch-all: digits, non-word run, digits
    assert normalize_visual_page_number("12...142") == "12"


def test_normalize_visual_page_number_sub_form_guard():
    # Chunk covers PDF indices [24, 33) → printed pages 25..33.
    rng = (24, 33)
    # In-range numeric passes through
    assert normalize_visual_page_number("25", rng) == "25"
    assert normalize_visual_page_number("33", rng) == "33"
    # Out-of-range bare numbers (sub-form misread) nulled out
    assert normalize_visual_page_number("1", rng) is None
    assert normalize_visual_page_number("8", rng) is None
    # "X/N" sub-pagination nulled when X is outside chunk range
    assert normalize_visual_page_number("1/8", rng) is None
    assert normalize_visual_page_number("Page 3 of 8", rng) is None
    # "Page 27 of 142" — real doc pagination, in-range → keep "27"
    assert normalize_visual_page_number("Page 27 of 142", rng) == "27"
    # Non-numeric (roman/appendix) bypasses the range check
    assert normalize_visual_page_number("iii", rng) == "iii"
    assert normalize_visual_page_number("A-5", rng) == "A-5"
    # No range supplied → legacy behavior (strip decoration only)
    assert normalize_visual_page_number("1/8") == "1"
    assert normalize_visual_page_number("1", None) == "1"


def test_merge_nulls_sub_form_pages_outside_chunk_range():
    # Two chunks, 20-page doc. Chunk 2 covers PDF 7..14 (printed 8..15)
    # but the VLM misreads an attached sub-form and emits "1" and "2"
    # as visual_page_numbers. Those must be nulled out so they don't
    # collide with real pages 1/2 from chunk 1.
    c1_tables = [
        _table(rows=[{"a": 1}], header="Real page 1", visual_page_number="1"),
        _table(rows=[{"b": 2}], header="Real page 2", visual_page_number="2"),
    ]
    c2_tables = [
        _table(rows=[{"x": 9}], header="Sub-form A", visual_page_number="1"),
        _table(rows=[{"y": 10}], header="Sub-form B", visual_page_number="2/8"),
    ]
    r1 = {"chunk_index": 0, "page_start": 0, "page_end": 9,
          "experiment": {}, "extracted": _chunk(tables=c1_tables)}
    r2 = {"chunk_index": 1, "page_start": 7, "page_end": 15,
          "experiment": {}, "extracted": _chunk(tables=c2_tables)}
    out = merge_chunks([r1, r2])
    pages = [t.get("visual_page_number") for t in out["tables"]]
    # Real pages 1 and 2 survive; sub-form misreads are nulled
    assert "1" in pages and "2" in pages
    # Exactly two sub-form tables with null page (not colliding with real 1/2)
    assert pages.count(None) == 2
    # And we kept all four tables (no collision collapses)
    assert len(out["tables"]) == 4


def test_cross_classification_dedup_same_page():
    # The VLM emits the same ranking-sheet table as Standard_Table in one
    # chunk and Key_Value_Form in the next. Same-page collapse should
    # merge them even though classifications differ.
    rows = [{"Applicant Name": "X", "Project Title": "Y", "PROJECT SCORE": "18"}]
    std = _table(rows=rows, header="APPENDIX A: APPLICATION RANKING SHEETS",
                 visual_page_number="80")
    std["table_classification"] = "Standard_Table"
    kv = _table(rows=rows, header="APPENDIX A: APPLICATION RANKING SHEETS",
                visual_page_number="80")
    kv["table_classification"] = "Key_Value_Form"
    kv["table_data"] = {"Applicant Name": "X", "Project Title": "Y",
                        "PROJECT SCORE": "18",
                        "1. Project Impact (33%)": "enhance knowledge",
                        "2. Project Design (22%)": "engaging in-person"}
    out = merge_chunks([_chunk(tables=[std, kv])])
    assert len(out["tables"]) == 1


def test_same_page_dedup_survives_rev_marker_drift():
    # Two chunks emit the same table but one header has "**REV**" prefix
    # while the other doesn't. Header normalization should treat them
    # as matching so ≥80% token overlap collapses them.
    rows = [{"Category": "Education", "Cap": "$5,000", "Share": "67%"},
            {"Category": "Planning", "Cap": "$10,000", "Share": "67%"},
            {"Category": "Management", "Cap": "$50,000", "Share": "75%"}]
    t1 = _table(rows=rows, header="**REV** APPENDIX A: RANKING SHEETS",
                visual_page_number="6")
    t2 = _table(rows=rows, header="APPENDIX A: RANKING SHEETS",
                visual_page_number="6")
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 1


def test_table_row_dedup_within_single_table():
    # Long table where the VLM repeated rows after losing its place.
    rows = [
        {"Rank": "1", "WBIC": "20", "Waterbody": "Lake Michigan"},
        {"Rank": "11", "WBIC": "1179900", "Waterbody": "Wisconsin River"},
        {"Rank": "1", "WBIC": "20", "Waterbody": "Lake Michigan"},  # dup
        {"Rank": "32", "WBIC": "88", "Waterbody": "Sturgeon Bay"},
        {"Rank": "11", "WBIC": "1179900", "Waterbody": "Wisconsin River"},  # dup
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="Top 300 AIS Prevention", visual_page_number="128")
    ])])
    kept_rows = out["tables"][0]["table_data"]
    assert len(kept_rows) == 3
    # Order preserved: first occurrence wins.
    assert [r["Rank"] for r in kept_rows] == ["1", "11", "32"]


def test_narrative_substring_collapse_same_page():
    # Two chunks capture the same section on page 18 — one prepends the
    # heading, the other doesn't. Fingerprint hashes the first 120 chars
    # so they survive primary dedup; substring collapse must drop the
    # shorter one.
    short = _narr(header="General Body Text", section="",
                  text="Wetland Incentives amounting to up to $10,000 each.")
    long_ = _narr(header="Wetland Incentives", section="SECTION 2",
                  text=("Section heading prepended. Wetland Incentives "
                        "amounting to up to $10,000 each."))
    short["visual_page_number"] = "18"
    long_["visual_page_number"] = "18"
    out = merge_chunks([_chunk(narrative_responses=[short, long_])])
    assert len(out["narrative_responses"]) == 1
    # The longer (containing) text is the one kept.
    assert "Section heading prepended" in out["narrative_responses"][0]["verbatim_text"]


def test_lint_flags_non_ascii_page_value():
    # If a non-ASCII char survives normalization, lint should call it out.
    t = _table(rows=[{"a": 1}], header="H", visual_page_number="weird\u00a7page")
    out = merge_chunks([_chunk(tables=[t])])
    assert any(
        "non-ASCII" in issue for issue in out["potential_issues"]
    ), out["potential_issues"]


def test_lint_flags_inconsistent_standard_table_keys():
    # VLM misreads column header → one row has different keys than the rest.
    rows = [
        {"A": "1", "B": "2"},
        {"A": "3", "B": "4"},
        {"X": "garbage"},  # inconsistent
    ]
    t = _table(rows=rows, header="H", visual_page_number="67")
    out = merge_chunks([_chunk(tables=[t])])
    assert any(
        "inconsistent column keys" in issue for issue in out["potential_issues"]
    ), out["potential_issues"]


def test_merge_normalizes_page_decoration_before_dedupe():
    # Two chunks see the same table at PDF page 51; chunk 1 captures the
    # printed page as "50" (clean), chunk 2 captures "50 | Page" (footer
    # text verbatim). They must dedupe to a single table.
    rows = [{"a": 1, "b": 2}]
    c1 = _chunk(tables=[_table(rows=rows, header="Project deliverables",
                               visual_page_number="50")])
    c2 = _chunk(tables=[_table(rows=rows, header="Project deliverables",
                               visual_page_number="50 | Page")])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1
    assert out["tables"][0]["visual_page_number"] == "50"


def test_extraction_prompt_recorded_in_experiment():
    prompt = "Extract tables and narratives from the given pages."
    r = {
        "chunk_index": 0, "page_start": 0, "page_end": 5,
        "experiment": {"model": "X"},
        "extracted": _chunk(tables=[_table(rows=[{"a": 1}])]),
    }
    out = merge_chunks([r], extraction_prompt=prompt)
    assert out["experiment"]["extraction_prompt"] == prompt
    # Multi-chunk path too
    out2 = merge_chunks([r, r], extraction_prompt=prompt)
    assert out2["experiment"]["extraction_prompt"] == prompt


def test_single_chunk_full_record_passthrough():
    # A doc with only one chunk: no dedupe runs, but chunks[] sidecar still
    # populated, experiment still copied.
    r = {
        "chunk_index": 0, "page_start": 0, "page_end": 5,
        "experiment": {"model": "X", "elapsed_ms": 999},
        "extracted": _chunk(
            tables=[_table(rows=[{"a": 1}])],
            one_sentence_summary="Whole-doc summary from chunk.",
        ),
    }
    out = merge_chunks([r])
    assert len(out["tables"]) == 1
    assert len(out["chunks"]) == 1
    assert out["experiment"] == {"model": "X"}  # elapsed_ms stripped


def test_cross_page_table_dedup_same_content():
    # A single logical table is labeled with different page numbers by
    # two overlapping chunks. Chunk A extracts 30 rows and labels the
    # copy p.128. Chunk B overlaps into the same region plus the
    # continuation and extracts 35 rows (superset), labeling the copy
    # p.129. Section headers also diverge between chunks ("APPENDIX G"
    # vs "Top 300 AIS") because the VLM latched onto different nearby
    # headings. Cross-page rule should collapse and keep the larger.
    rows_a = [
        {"Rank": str(i), "WBIC": str(100 + i), "Waterbody": f"Lake {i}"}
        for i in range(1, 31)
    ]
    rows_b = rows_a + [
        {"Rank": str(i), "WBIC": str(100 + i), "Waterbody": f"Lake {i}"}
        for i in range(31, 36)
    ]
    t1 = _table(rows=rows_a, header="APPENDIX G: AIS PREVENTION",
                visual_page_number="128")
    t2 = _table(rows=rows_b, header="Top 300 AIS Prevention",
                visual_page_number="129")
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 1, (
        f"expected cross-page collapse, got {len(out['tables'])}"
    )
    # Winner is the larger table.
    assert len(out["tables"][0]["table_data"]) == 35


def test_cross_page_collapse_respects_page_gap():
    # Identical tables legitimately recur far apart in the doc (e.g.,
    # same summary table reprinted as an appendix). Page gap larger than
    # the window means they should NOT collapse.
    rows = [
        {"Cat": f"Item {i}", "Val": str(i)} for i in range(1, 15)
    ]
    t_front = _table(rows=rows, header="Summary",
                     visual_page_number="5")
    t_appendix = _table(rows=rows, header="Summary",
                        visual_page_number="140")
    out = merge_chunks([_chunk(tables=[t_front, t_appendix])])
    assert len(out["tables"]) == 2, (
        f"expected no cross-page collapse across large gap, "
        f"got {len(out['tables'])}"
    )


def test_cross_page_collapse_respects_token_floor():
    # Two near-empty tables on adjacent pages with identical content
    # should NOT collapse via the cross-page rule — token floor protects
    # same-shape distinct tables from being collapsed just because they
    # look similar.
    rows = [{"Col": "A", "Val": "1"}]  # tiny: < 20 tokens
    t1 = _table(rows=rows, header="Y1 Budget", visual_page_number="10")
    t2 = _table(rows=rows, header="Y2 Budget", visual_page_number="11")
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 2, (
        f"token floor should protect small same-shape tables, "
        f"got {len(out['tables'])}"
    )


def test_narrative_dedupe_ignores_cite_markers():
    # Two chunks capture the same section on the same page. Cite markers
    # are numbered per-entry so they differ between the two copies —
    # without stripping, substring containment fails.
    body_a = (
        "Ordinance Development projects create local regulations to "
        "benefit surface waters. [cite: 56] They relate to topics like "
        "boating, AIS prevention, erosion control. [cite: 57]"
    )
    body_b = (
        "Ordinance Development projects create local regulations to "
        "benefit surface waters. [cite: 3] They relate to topics like "
        "boating, AIS prevention, erosion control. [cite: 4]"
    )
    n1 = _narr(header="General Body Text", section="Ordinance Development",
               text=body_a)
    n2 = _narr(header="Ordinance Development", section="SECTION 2",
               text=body_b)
    n1["visual_page_number"] = "20"
    n2["visual_page_number"] = "20"
    out = merge_chunks([
        _chunk(narrative_responses=[n1]),
        _chunk(narrative_responses=[n2]),
    ])
    assert len(out["narrative_responses"]) == 1, (
        f"cite-marker noise blocked collapse: "
        f"{len(out['narrative_responses'])} narratives kept"
    )


def test_narrative_fingerprint_collapses_across_cite_drift():
    # Same narrative, same page, same header — just different cite
    # numbers embedded in the first 120 chars. Fingerprint must strip
    # them so primary dedupe catches it (not just the substring pass).
    body_a = "Budget submission deadline is November 15. [cite: 1] All " \
             "projects must include a cover letter. [cite: 2]"
    body_b = "Budget submission deadline is November 15. [cite: 9] All " \
             "projects must include a cover letter. [cite: 10]"
    n1 = _narr(header="General Body Text", section="Deadlines", text=body_a)
    n2 = _narr(header="General Body Text", section="Deadlines", text=body_b)
    n1["visual_page_number"] = "7"
    n2["visual_page_number"] = "7"
    out = merge_chunks([
        _chunk(narrative_responses=[n1]),
        _chunk(narrative_responses=[n2]),
    ])
    assert len(out["narrative_responses"]) == 1


def test_malformed_cite_markers_stripped_from_output():
    # VLM tokenizer drift: "[cite: 世]" where a digit should be. Well-formed
    # "[cite: 3]" markers should be preserved.
    n = _narr(
        header="General Body Text",
        text="First statement. [cite: 1] Second statement. [cite: 世] "
             "Third statement. [cite: 3] Fourth with weird body. [cite: abc]",
    )
    n["visual_page_number"] = "5"
    out = merge_chunks([_chunk(narrative_responses=[n])])
    text = out["narrative_responses"][0]["verbatim_text"]
    assert "[cite: 1]" in text
    assert "[cite: 3]" in text
    assert "[cite: 世]" not in text
    assert "[cite: abc]" not in text


def test_malformed_cite_stripping_preserves_well_formed_tail():
    # Edge case: large cite numbers, multi-digit, mixed with malformed.
    n = _narr(
        header="Prose",
        text="A. [cite: 42] B. [cite: ] C. [cite: 100] D. [cite: 1a]",
    )
    n["visual_page_number"] = "1"
    out = merge_chunks([_chunk(narrative_responses=[n])])
    text = out["narrative_responses"][0]["verbatim_text"]
    assert "[cite: 42]" in text and "[cite: 100]" in text
    assert "[cite: ]" not in text  # empty body
    assert "[cite: 1a]" not in text  # mixed alpha


def test_exotic_unicode_stripped_from_narrative():
    # CJK char dropped into English narrative = VLM token drift. The
    # merger strips the glitched chars from values (keys are flagged
    # separately, see test_lint_flags_exotic_unicode_in_json_keys).
    n = _narr(
        header="General Body Text",
        text="Please refer to the code. Ch. NR 1.91, Wis.牌. Code.",
    )
    n["visual_page_number"] = "8"
    out = merge_chunks([_chunk(narrative_responses=[n])])
    assert "牌" not in out["narrative_responses"][0]["verbatim_text"]
    # The surrounding context survives — only the glitched char drops.
    assert "Wis.. Code." in out["narrative_responses"][0]["verbatim_text"]


def test_exotic_unicode_stripped_from_table_cells():
    # CJK char inside a table cell (form number glitch). Stripped.
    rows = [
        {"Form": "Grant Payment Request", "Number": "8700-001"},
        {"Form": "Mileage Log", "Number": "世700-012"},  # CJK drift
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="APPENDIX K: LIST OF FORMS",
               visual_page_number="141")
    ])])
    cells = out["tables"][0]["table_data"]
    for row in cells:
        for v in row.values():
            assert "世" not in v
    # The drift char in the second row's Number is gone.
    assert any(r.get("Number") == "700-012" for r in cells)


def test_lint_does_not_flag_smart_quotes_or_accents():
    # Smart quotes and Latin accented chars are legitimate — don't flag.
    n = _narr(
        header="General Body Text",
        text="Include a “quoted phrase” and an accented word: café. "
             "Also the department\u2019s policy.",
    )
    n["visual_page_number"] = "1"
    out = merge_chunks([_chunk(narrative_responses=[n])])
    assert not any(
        "exotic unicode" in issue for issue in out["potential_issues"]
    ), out["potential_issues"]


def test_lint_flags_exotic_unicode_in_json_keys():
    # VLM tokenizer drift on schema keys themselves — e.g., emits
    # "visual_page世_number" instead of "visual_page_number". Worse than
    # a value glitch because it's structural (downstream code looking
    # up the normal key name silently gets None).
    n = _narr(
        header="General Body Text",
        text="Normal narrative text without any exotic characters.",
    )
    n["visual_page世_number"] = "42"  # type: ignore
    out = merge_chunks([_chunk(narrative_responses=[n])])
    assert any(
        "dict key" in issue and "exotic unicode" in issue
        for issue in out["potential_issues"]
    ), out["potential_issues"]


def test_lint_flags_exotic_unicode_in_table_column_headers():
    # Column-header drift: VLM corrupts a Standard_Table column name.
    rows = [
        {"Form": "Mileage Log", "Nu世mber": "8700-012"},  # corrupted key
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="APPENDIX K", visual_page_number="141")
    ])])
    assert any(
        "dict key" in issue and "exotic unicode" in issue
        for issue in out["potential_issues"]
    ), out["potential_issues"]


def test_finalize_stakeholders_folds_subset_into_fuller():
    # Same org, same role; one entry fills department, the other leaves
    # it empty. No field conflicts => collapse into single survivor.
    fuller = _stake(
        role="Sponsor Contact", inst="Wisconsin DNR",
        dept="Bureau of Water Quality",
    )
    sparse = _stake(role="Sponsor Contact", inst="Wisconsin DNR")
    out = merge_chunks([_chunk(stakeholders=[fuller]), _chunk(stakeholders=[sparse])])
    stakeholders = out["stakeholders"]
    assert len(stakeholders) == 1, stakeholders
    assert stakeholders[0]["department"] == "Bureau of Water Quality"


def test_finalize_stakeholders_keeps_entries_with_conflicting_fields():
    # Same institution, same role, but DIFFERENT department strings
    # (neither empty). Real conflict => both survive.
    a = _stake(
        role="Sponsor Contact", inst="Wisconsin DNR",
        dept="Bureau of Water Quality",
    )
    b = _stake(
        role="Sponsor Contact", inst="Wisconsin DNR",
        dept="Bureau of Community Financial Assistance",
    )
    out = merge_chunks([_chunk(stakeholders=[a, b])])
    assert len(out["stakeholders"]) == 2, out["stakeholders"]


def test_finalize_stakeholders_keeps_distinct_emails_apart():
    # Two grant admin contacts at the same institution with different
    # emails are distinct people, even if every other field matches.
    cbcw = _stake(
        role="Grants Administrative Contact",
        inst="Wisconsin DNR",
        email="DNRCBCWGrants@wisconsin.gov",
    )
    swims = _stake(
        role="Grants Administrative Contact",
        inst="Wisconsin DNR",
        email="DNRSWIMS@wisconsin.gov",
    )
    out = merge_chunks([_chunk(stakeholders=[cbcw, swims])])
    assert len(out["stakeholders"]) == 2


def test_self_keyed_standard_table_reclassified_to_literal_grid():
    # VLM emitted a tabular section as Standard_Table with rows that have
    # key == value for every cell — no real schema. Reclassifier should
    # convert to Literal_Grid and preserve the cell layout as a 2D array.
    rows = [
        {"Planning Needs Assessment": "Planning Needs Assessment",
         "Data Gap Analysis": "Data Gap Analysis"},
        {"Write Plan": "Write Plan",
         "Plan & Implementation": "Plan & Implementation"},
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="APPENDIX B: MANAGEMENT PLANNING",
               visual_page_number="106")
    ])])
    t = out["tables"][0]
    assert t["table_classification"] == "Literal_Grid", t["table_classification"]
    assert t["table_data"] == [
        ["Planning Needs Assessment", "Data Gap Analysis"],
        ["Write Plan", "Plan & Implementation"],
    ], t["table_data"]


def test_self_keyed_reclassifier_leaves_real_standard_table_alone():
    # A Standard_Table with real column headers (key != value) should
    # NOT be reclassified — only fully self-keyed rows trigger conversion.
    rows = [
        {"Date": "May 1", "Action": "Confirm eligibility"},
        {"Date": "Nov 15", "Action": "Submit application"},
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="Grant cycle timeline", visual_page_number="7")
    ])])
    t = out["tables"][0]
    assert t["table_classification"] == "Standard_Table"
    assert isinstance(t["table_data"][0], dict)


def test_self_keyed_reclassifier_skips_mixed_rows():
    # Even one cell with a real header (key != value) is enough to
    # leave the table alone — better an awkward Standard_Table than to
    # discard a real schema.
    rows = [
        {"Header A": "Header A", "Header B": "Header B"},
        {"Header A": "real value", "Header B": "real value 2"},
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="Mixed table", visual_page_number="42")
    ])])
    t = out["tables"][0]
    assert t["table_classification"] == "Standard_Table"


def _table_crel(crel, **kw):
    kw.setdefault("rows", [{"x": 1}])
    t = _table(**kw)
    t["chunk_relative_page_index"] = crel
    return t


def test_pdf_page_index_derived_from_chunk_relative():
    # Chunk covers PDF pages 11-20 (page_start=10, page_end=20 half-open).
    # A table with chunk_relative_page_index=5 came from the 5th image
    # of the chunk, which is PDF page 10 + 5 = 15. Page fields live on
    # tables only.
    t = _table_crel(5, header="Section", visual_page_number="15")
    r = {
        "chunk_index": 1, "page_start": 10, "page_end": 20,
        "experiment": {},
        "extracted": _chunk(tables=[t]),
    }
    out = merge_chunks([r])
    item = out["tables"][0]
    assert item["pdf_page_index"] == 15
    # chunk_relative_page_index is consumed during derivation; not in output.
    assert "chunk_relative_page_index" not in item


def test_pdf_page_index_out_of_range_nulled():
    # Chunk covers PDF pages 11-20 (10 images total). chunk_relative=99
    # is outside the chunk's image range — VLM mis-counted. Should be
    # nulled but the rest of the table is kept.
    t = _table_crel(99, header="Section", visual_page_number="15")
    r = {
        "chunk_index": 1, "page_start": 10, "page_end": 20,
        "experiment": {},
        "extracted": _chunk(tables=[t]),
    }
    out = merge_chunks([r])
    item = out["tables"][0]
    assert item["pdf_page_index"] is None
    assert item["preceding_section_header"] == "Section"


def test_pdf_page_index_string_coerced_to_int():
    # VLM occasionally emits "5" instead of 5. Coerce to int when in
    # range; null when not parseable.
    t_ok = _table_crel("5", header="A")
    t_bad = _table_crel("not-a-number", header="B")
    r = {
        "chunk_index": 0, "page_start": 10, "page_end": 20,
        "experiment": {},
        "extracted": _chunk(tables=[t_ok, t_bad]),
    }
    out = merge_chunks([r])
    items = out["tables"]
    ok = [t for t in items if t.get("preceding_section_header") == "A"][0]
    bad = [t for t in items if t.get("preceding_section_header") == "B"][0]
    assert ok["pdf_page_index"] == 15  # 10 + 5
    assert bad["pdf_page_index"] is None


def test_pdf_page_index_legacy_field_is_ignored():
    # Migration: if VLM still emits the legacy pdf_page_index field
    # directly (mirror of visual_page_number — wrong), the deriver
    # ignores it and uses chunk_relative_page_index instead.
    t = _table(header="A", rows=[{"x": 1}])
    t["pdf_page_index"] = 999  # legacy/wrong
    t["chunk_relative_page_index"] = 3
    r = {
        "chunk_index": 0, "page_start": 10, "page_end": 20,
        "experiment": {},
        "extracted": _chunk(tables=[t]),
    }
    out = merge_chunks([r])
    assert out["tables"][0]["pdf_page_index"] == 13


def test_pdf_page_index_left_alone_without_chunk_range():
    # Raw-dict input (no chunk_page_range available) means the merger
    # can't derive pdf_page_index; the field passes through unchanged
    # whatever the caller put on the table.
    t = _table(header="A", rows=[{"x": 1}])
    t["pdf_page_index"] = 999
    out = merge_chunks([_chunk(tables=[t])])
    assert out["tables"][0]["pdf_page_index"] == 999


def _table_with_pdf(rows, pdf_page, header="", visual_page=None):
    t = _table(rows=rows, header=header,
               visual_page_number=visual_page or str(pdf_page))
    t["pdf_page_index"] = pdf_page
    return t


def test_supertable_collapse_merges_consecutive_pages():
    # Real-world pattern: a "Program-approved protocols" supertable spans
    # pages 45-47 with one subtable per page, all sharing identical
    # column headers but distinct preceding_section_header values.
    cols = ["Category", "Protocol", "Link"]
    t1 = _table_with_pdf(
        [{"Category": "Decon", "Protocol": "Boat Decon", "Link": "url1"}],
        pdf_page=45, header="Program-approved protocols",
    )
    t2 = _table_with_pdf(
        [{"Category": "Citizen", "Protocol": "Secchi", "Link": "url2"}],
        pdf_page=46, header="CITIZEN MONITORING",
    )
    t3 = _table_with_pdf(
        [{"Category": "AIS", "Protocol": "Early Det", "Link": "url3"}],
        pdf_page=47, header="AIS MONITORING",
    )
    out = merge_chunks([_chunk(tables=[t1, t2, t3])])
    assert len(out["tables"]) == 1, [t.get("preceding_section_header") for t in out["tables"]]
    merged = out["tables"][0]
    # First table's section header is the supertable heading; survives.
    assert merged["preceding_section_header"] == "Program-approved protocols"
    # All three subtables' rows are present, in page order.
    assert len(merged["table_data"]) == 3
    assert [r["Protocol"] for r in merged["table_data"]] == [
        "Boat Decon", "Secchi", "Early Det",
    ]


def test_supertable_collapse_skips_when_columns_differ():
    # Adjacent Standard_Tables on consecutive pages but with different
    # column shapes — these are unrelated tables; do not merge.
    t1 = _table_with_pdf(
        [{"A": "1", "B": "2"}], pdf_page=10, header="Table A",
    )
    t2 = _table_with_pdf(
        [{"X": "9", "Y": "8"}], pdf_page=11, header="Table B",
    )
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 2


def test_supertable_collapse_respects_page_gap():
    # Same column shape but the gap between subtables exceeds the
    # max_page_gap threshold — likely two unrelated tables, not a
    # supertable. Default threshold is 2; pages 10 and 14 differ by 4.
    cols = ["Date", "Action"]
    t1 = _table_with_pdf(
        [{"Date": "May 1", "Action": "X"}], pdf_page=10, header="Schedule A",
    )
    t2 = _table_with_pdf(
        [{"Date": "Nov 15", "Action": "Y"}], pdf_page=14, header="Schedule B",
    )
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 2


def test_supertable_collapse_chains_three_in_a_row():
    # Five matching subtables on pages 80, 81, 82, 83, 84 should all
    # collapse into a single entry.
    rows = [
        ({"Col": "v1"}, 80, "Supertable header"),
        ({"Col": "v2"}, 81, "Sub A"),
        ({"Col": "v3"}, 82, "Sub B"),
        ({"Col": "v4"}, 83, "Sub C"),
        ({"Col": "v5"}, 84, "Sub D"),
    ]
    tables = [_table_with_pdf([r], pdf_page=p, header=h) for r, p, h in rows]
    out = merge_chunks([_chunk(tables=tables)])
    assert len(out["tables"]) == 1
    assert len(out["tables"][0]["table_data"]) == 5
    assert out["tables"][0]["preceding_section_header"] == "Supertable header"


def test_supertable_collapse_skips_continuation_flagged():
    # A table marked continues_to_next_chunk is the cross-chunk stitch's
    # job; the supertable collapser should skip it to avoid double-stitch.
    t1 = _table_with_pdf(
        [{"Col": "a"}], pdf_page=10, header="Continuation candidate",
    )
    t1["continues_to_next_chunk"] = True
    t2 = _table_with_pdf(
        [{"Col": "b"}], pdf_page=11, header="Other",
    )
    out = merge_chunks([_chunk(tables=[t1, t2])])
    # Both survive — the continuation flag is stripped by post-processing
    # but the supertable check ran before strip, so they stayed separate.
    assert len(out["tables"]) == 2


def test_supertable_collapse_skips_when_pdf_page_index_missing():
    # Without pdf_page_index, the merger has no way to confirm adjacency.
    # Don't merge — better to keep two tables than risk a wrong merge.
    cols = ["Col"]
    t1 = _table(rows=[{"Col": "a"}], header="A", visual_page_number="10")
    # Note: no pdf_page_index set
    t2 = _table(rows=[{"Col": "b"}], header="B", visual_page_number="11")
    out = merge_chunks([_chunk(tables=[t1, t2])])
    assert len(out["tables"]) == 2


def test_array_valued_standard_table_reclassified_to_literal_grid():
    # APPENDIX K LIST OF FORMS pattern: each row's value is a list of
    # forms under a section header. The VLM emitted as Standard_Table
    # but the cells are arrays — convert to Literal_Grid where the
    # section header opens a group and each form gets its own row.
    rows = [
        {"Application forms": ["Form 8700-284", "Form 8700-035"]},
        {"Reimbursement forms": ["Form 8700-001", "Form 8700-349A"]},
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="APPENDIX K", visual_page_number="141")
    ])])
    t = out["tables"][0]
    assert t["table_classification"] == "Literal_Grid"
    # Section header followed by its forms, then next section.
    assert t["table_data"][0] == ["Application forms"]
    assert t["table_data"][1] == ["Form 8700-284"]
    assert t["table_data"][2] == ["Form 8700-035"]
    assert t["table_data"][3] == ["Reimbursement forms"]


def test_array_valued_reclassifier_skips_mixed_rows():
    # Even one row with a non-array value blocks reclassification.
    rows = [
        {"key": ["a", "b"]},
        {"key": "scalar"},
    ]
    out = merge_chunks([_chunk(tables=[
        _table(rows=rows, header="Mixed", visual_page_number="1")
    ])])
    assert out["tables"][0]["table_classification"] == "Standard_Table"


def test_narrative_fragment_absorbed_by_longer():
    # Chunk-overlap leftover: a shorter narrative whose text is a
    # substring of a longer narrative should be dropped.
    full = _narr(header="Section A", text="The full body of section A spans multiple sentences.")
    fragment = _narr(header="Section A", text="The full body of section A")
    out = merge_chunks([_chunk(narrative_responses=[full, fragment])])
    assert len(out["narrative_responses"]) == 1
    assert "multiple sentences" in out["narrative_responses"][0]["verbatim_text"]


def test_narrative_kept_when_text_unique():
    # If a narrative's text isn't a substring of another, keep both.
    a = _narr(header="A", text="Completely different content about topic Z.")
    b = _narr(header="B", text="Standalone text from an unnumbered figure caption.")
    out = merge_chunks([_chunk(narrative_responses=[a, b])])
    assert len(out["narrative_responses"]) == 2


def test_fragment_not_preferred_over_complete():
    # Even if the fragment has more rows by accident, full copy wins
    c1 = _chunk(tables=[_table(
        rows=[{"r": 1}, {"r": 2}],
        header="Budget",
        cfp=False, ctn=False,
    )])
    c2 = _chunk(tables=[_table(
        rows=[{"r": 1}, {"r": 2}, {"r": "bogus"}],
        header="Budget",
        cfp=True, ctn=False,  # fragment — continues from previous
    )])
    out = merge_chunks([c1, c2])
    assert len(out["tables"]) == 1
    # Prefer the non-fragment copy
    rows = out["tables"][0]["table_data"]
    assert "bogus" not in [r.get("r") for r in rows]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_single_chunk_passthrough,
    test_identity_dedupe_stakeholders,
    test_identity_dedupe_addresses,
    test_span_dedupe_collapses_full_copies_in_overlap,
    test_section_header_disambiguates_same_shape_tables,
    test_continuation_stitch_across_chunks,
    test_three_chunk_fragment_chain,
    test_narrative_stitch,
    test_doc_level_aggregation,
    test_doc_details_null_coalesce,
    test_merge_chunks_json_handles_parse_errors,
    test_visual_page_number_survives_dedupe_and_stitch,
    test_stakeholder_role_dedupe_unnamed,
    test_table_dedupe_tolerates_section_header_variance,
    test_narrative_dedupe_tolerates_section_header_variance,
    test_same_page_distinct_narratives_not_merged,
    test_chunks_sidecar_from_full_records,
    test_raw_dict_input_yields_empty_sidecar,
    test_potential_issues_lint_flags_empty_narratives,
    test_empty_tables_are_dropped_in_postprocess,
    test_potential_issues_lint_clean_when_nothing_wrong,
    test_stakeholders_preserve_source_order,
    test_empty_stakeholders_filtered_out,
    test_normalize_visual_page_number,
    test_normalize_visual_page_number_smart_quote_and_exotic_separators,
    test_normalize_visual_page_number_sub_form_guard,
    test_merge_nulls_sub_form_pages_outside_chunk_range,
    test_cross_classification_dedup_same_page,
    test_same_page_dedup_survives_rev_marker_drift,
    test_table_row_dedup_within_single_table,
    test_narrative_substring_collapse_same_page,
    test_lint_flags_non_ascii_page_value,
    test_lint_flags_inconsistent_standard_table_keys,
    test_merge_normalizes_page_decoration_before_dedupe,
    test_extraction_prompt_recorded_in_experiment,
    test_single_chunk_full_record_passthrough,
    test_cross_page_table_dedup_same_content,
    test_cross_page_collapse_respects_page_gap,
    test_cross_page_collapse_respects_token_floor,
    test_narrative_dedupe_ignores_cite_markers,
    test_narrative_fingerprint_collapses_across_cite_drift,
    test_malformed_cite_markers_stripped_from_output,
    test_malformed_cite_stripping_preserves_well_formed_tail,
    test_exotic_unicode_stripped_from_narrative,
    test_exotic_unicode_stripped_from_table_cells,
    test_lint_does_not_flag_smart_quotes_or_accents,
    test_lint_flags_exotic_unicode_in_json_keys,
    test_lint_flags_exotic_unicode_in_table_column_headers,
    test_finalize_stakeholders_folds_subset_into_fuller,
    test_finalize_stakeholders_keeps_entries_with_conflicting_fields,
    test_finalize_stakeholders_keeps_distinct_emails_apart,
    test_self_keyed_standard_table_reclassified_to_literal_grid,
    test_self_keyed_reclassifier_leaves_real_standard_table_alone,
    test_self_keyed_reclassifier_skips_mixed_rows,
    test_array_valued_standard_table_reclassified_to_literal_grid,
    test_array_valued_reclassifier_skips_mixed_rows,
    test_narrative_fragment_absorbed_by_longer,
    test_narrative_kept_when_text_unique,
    test_pdf_page_index_derived_from_chunk_relative,
    test_pdf_page_index_legacy_field_is_ignored,
    test_pdf_page_index_out_of_range_nulled,
    test_pdf_page_index_string_coerced_to_int,
    test_pdf_page_index_left_alone_without_chunk_range,
    test_supertable_collapse_merges_consecutive_pages,
    test_supertable_collapse_skips_when_columns_differ,
    test_supertable_collapse_respects_page_gap,
    test_supertable_collapse_chains_three_in_a_row,
    test_supertable_collapse_skips_continuation_flagged,
    test_supertable_collapse_skips_when_pdf_page_index_missing,
    test_fragment_not_preferred_over_complete,
]


def main() -> int:
    passed = 0
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
        except Exception:
            print(f"ERROR {t.__name__}:")
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
