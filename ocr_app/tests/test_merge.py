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

from scripts.merge import merge_chunks, merge_chunks_json  # noqa: E402


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


def _stake(email="", first="", last="", full="", inst="", role="Unknown"):
    return {
        "context_snippet": "",
        "stakeholder_role": role,
        "full_name": full,
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": "",
        "institution": inst,
        "department": "",
        "position_title": "",
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
    # Longest summary wins
    assert "longer descriptive" in out["one_sentence_summary"]


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
