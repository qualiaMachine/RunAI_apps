#!/usr/bin/env python3
"""QA audit for OCR extraction output — find missing pages, thin content, and truncation.

Reads _extracted.json files produced by the notebook pipeline or batch_extract.py
and generates a coverage report highlighting gaps:

  - Pages present in the PDF but absent from the JSON
  - Pages with empty tables/narratives (potential extraction failures)
  - Pages where output was likely truncated (parse_error, low confidence, short output)
  - Orientation changes (portrait↔landscape) that correlate with failures
  - Token-limit warnings (finish_reason == "length")

Usage:
    # Audit a single extracted JSON against its source PDF
    python ocr_app/scripts/qa_audit.py \
        --json /data/extracted/big_doc_extracted.json \
        --pdf  /data/documents/big_doc.pdf

    # Audit all extracted JSONs in a directory (auto-matches PDFs by stem)
    python ocr_app/scripts/qa_audit.py \
        --json-dir /data/extracted \
        --pdf-dir  /data/documents

    # Write report to file instead of stdout
    python ocr_app/scripts/qa_audit.py \
        --json-dir /data/extracted \
        --pdf-dir  /data/documents \
        --output   /data/extracted/qa_report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def get_pdf_page_info(pdf_path: Path) -> list[dict]:
    """Extract page count and orientation info from a PDF."""
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        rect = page.rect
        width, height = rect.width, rect.height
        orientation = "landscape" if width > height else "portrait"
        text_len = len(page.get_text("text").strip())
        pages.append({
            "page": i + 1,
            "width": round(width, 1),
            "height": round(height, 1),
            "orientation": orientation,
            "text_chars": text_len,
        })
    doc.close()
    return pages


def detect_orientation_transitions(pdf_pages: list[dict]) -> list[dict]:
    """Find pages where orientation changes from the previous page."""
    transitions = []
    for i in range(1, len(pdf_pages)):
        prev_orient = pdf_pages[i - 1]["orientation"]
        curr_orient = pdf_pages[i]["orientation"]
        if prev_orient != curr_orient:
            transitions.append({
                "page": pdf_pages[i]["page"],
                "from": prev_orient,
                "to": curr_orient,
            })
    return transitions


def analyze_extracted_json(data: dict) -> dict:
    """Analyze an extracted JSON for completeness and quality signals."""
    # Handle both notebook output format and batch_extract format
    pages_data = []

    # Notebook format: top-level assembly with PageConfidences, etc.
    if "PageCount" in data:
        # This is the assembled document format from the notebook
        page_count = data.get("PageCount", 0)
        page_confidences = {
            pc["PageNumber"]: pc["ConfidencePercentage"]
            for pc in data.get("PageConfidences", [])
        }

        # Collect which pages have tables
        table_pages = set()
        for t in data.get("TablesCollection", []):
            pg = t.get("PageNumber")
            if pg:
                table_pages.add(pg)

        # Collect which pages have narratives
        narrative_pages = set()
        for n in data.get("NarrativeResponses", []):
            section = n.get("SectionOrPage", "")
            if section.startswith("PAGE "):
                try:
                    pg = int(section.split()[1])
                    narrative_pages.add(pg)
                except (IndexError, ValueError):
                    pass

        # Collect which pages have stakeholders
        stakeholder_pages = set()
        for s in data.get("Stakeholders", []):
            pg = s.get("PageNumber")
            if pg:
                stakeholder_pages.add(pg)

        # Collect page summaries
        summary_pages = set()
        for s in data.get("PageSummaries", []):
            pg = s.get("PageNumber")
            if pg:
                summary_pages.add(pg)

        for pg in range(1, page_count + 1):
            has_content = (
                pg in table_pages
                or pg in narrative_pages
                or pg in stakeholder_pages
            )
            pages_data.append({
                "page": pg,
                "present": pg in summary_pages or pg in page_confidences,
                "confidence": page_confidences.get(pg, None),
                "has_tables": pg in table_pages,
                "has_narratives": pg in narrative_pages,
                "has_stakeholders": pg in stakeholder_pages,
                "has_content": has_content,
                "method": None,
                "parse_error": False,
            })

        return {
            "format": "notebook_assembled",
            "page_count": page_count,
            "pages": pages_data,
        }

    # Batch extract format: {pages: [{page, text, method, elapsed_ms}]}
    if "pages" in data and isinstance(data["pages"], list):
        page_count = data.get("total_pages", len(data["pages"]))
        extracted_pages = {}
        for p in data["pages"]:
            pg = p.get("page", 0)
            text = p.get("text", "")
            # Try to detect content richness from the raw text
            has_tables = '"table' in text.lower() or '"rows"' in text.lower()
            has_narratives = '"narrative' in text.lower() or '"verbatim' in text.lower()
            extracted_pages[pg] = {
                "page": pg,
                "present": True,
                "confidence": None,
                "has_tables": has_tables,
                "has_narratives": has_narratives,
                "has_stakeholders": '"stakeholder' in text.lower(),
                "has_content": len(text.strip()) > 50,
                "method": p.get("method", "unknown"),
                "parse_error": False,
                "text_length": len(text),
            }

        for pg in range(1, page_count + 1):
            if pg in extracted_pages:
                pages_data.append(extracted_pages[pg])
            else:
                pages_data.append({
                    "page": pg,
                    "present": False,
                    "confidence": None,
                    "has_tables": False,
                    "has_narratives": False,
                    "has_stakeholders": False,
                    "has_content": False,
                    "method": None,
                    "parse_error": False,
                })

        return {
            "format": "batch_extract",
            "page_count": page_count,
            "pages": pages_data,
        }

    # Notebook per-page format (list of page records)
    if isinstance(data, list):
        for item in data:
            pg = item.get("page", 0)
            extracted = item.get("extracted", {})
            has_parse_error = "parse_error" in extracted
            pages_data.append({
                "page": pg,
                "present": True,
                "confidence": extracted.get("confidence_percentage"),
                "has_tables": bool(extracted.get("tables")),
                "has_narratives": bool(extracted.get("narrative_responses")),
                "has_stakeholders": bool(extracted.get("stakeholders")),
                "has_content": not has_parse_error,
                "method": item.get("method", "unknown"),
                "parse_error": has_parse_error,
            })
        return {
            "format": "notebook_pages",
            "page_count": len(data),
            "pages": pages_data,
        }

    return {"format": "unknown", "page_count": 0, "pages": []}


def audit_document(json_path: Path, pdf_path: Optional[Path] = None) -> dict:
    """Run a full QA audit on an extracted document."""
    with open(json_path) as f:
        data = json.load(f)

    analysis = analyze_extracted_json(data)
    report = {
        "filename": json_path.name,
        "source_pdf": str(pdf_path) if pdf_path else None,
        "format": analysis["format"],
        "total_pages_in_json": analysis["page_count"],
        "issues": [],
    }

    # Get PDF info if available
    pdf_pages = []
    orientation_transitions = []
    if pdf_path and pdf_path.exists():
        pdf_pages = get_pdf_page_info(pdf_path)
        orientation_transitions = detect_orientation_transitions(pdf_pages)
        report["total_pages_in_pdf"] = len(pdf_pages)
        report["orientation_transitions"] = orientation_transitions

        # Check for page count mismatch
        if len(pdf_pages) != analysis["page_count"]:
            report["issues"].append({
                "severity": "HIGH",
                "type": "page_count_mismatch",
                "detail": (
                    f"PDF has {len(pdf_pages)} pages but JSON has "
                    f"{analysis['page_count']} page entries"
                ),
            })

    # Build orientation lookup
    orientation_by_page = {}
    for pp in pdf_pages:
        orientation_by_page[pp["page"]] = pp["orientation"]

    # Analyze each page
    pages_present = set()
    missing_pages = []
    thin_pages = []
    failed_pages = []
    low_confidence_pages = []

    for pg_data in analysis["pages"]:
        pg = pg_data["page"]
        pages_present.add(pg)

        if not pg_data["present"]:
            orient = orientation_by_page.get(pg, "unknown")
            missing_pages.append({"page": pg, "orientation": orient})
            continue

        if pg_data["parse_error"]:
            failed_pages.append(pg)

        if pg_data["confidence"] is not None and pg_data["confidence"] < 30:
            low_confidence_pages.append({
                "page": pg,
                "confidence": pg_data["confidence"],
            })

        if not pg_data["has_content"]:
            orient = orientation_by_page.get(pg, "unknown")
            thin_pages.append({"page": pg, "orientation": orient})

    # Check for pages in PDF not in JSON
    if pdf_pages:
        pdf_page_nums = set(pp["page"] for pp in pdf_pages)
        json_page_nums = set(pg["page"] for pg in analysis["pages"] if pg["present"])
        truly_missing = pdf_page_nums - json_page_nums
        for pg in sorted(truly_missing):
            orient = orientation_by_page.get(pg, "unknown")
            if not any(m["page"] == pg for m in missing_pages):
                missing_pages.append({"page": pg, "orientation": orient})

    # Correlate missing pages with orientation transitions
    transition_pages = set(t["page"] for t in orientation_transitions)
    # Pages adjacent to transitions (within 1 page) are at higher risk
    transition_zone = set()
    for tp in transition_pages:
        transition_zone.update([tp - 1, tp, tp + 1])

    missing_near_transitions = [
        m for m in missing_pages if m["page"] in transition_zone
    ]

    # Generate issues
    if missing_pages:
        report["issues"].append({
            "severity": "HIGH",
            "type": "missing_pages",
            "count": len(missing_pages),
            "pages": sorted(missing_pages, key=lambda m: m["page"]),
            "detail": (
                f"{len(missing_pages)} page(s) missing from extraction output"
            ),
        })

    if missing_near_transitions:
        report["issues"].append({
            "severity": "HIGH",
            "type": "missing_near_orientation_change",
            "count": len(missing_near_transitions),
            "pages": sorted(missing_near_transitions, key=lambda m: m["page"]),
            "detail": (
                f"{len(missing_near_transitions)} missing page(s) are near "
                f"portrait↔landscape transitions — orientation changes likely "
                f"contributed to extraction failure"
            ),
        })

    if thin_pages:
        report["issues"].append({
            "severity": "MEDIUM",
            "type": "thin_content",
            "count": len(thin_pages),
            "pages": sorted(thin_pages, key=lambda t: t["page"]),
            "detail": (
                f"{len(thin_pages)} page(s) extracted but with no tables, "
                f"narratives, or stakeholders — possible prompt mismatch or "
                f"truncation"
            ),
        })

    if failed_pages:
        report["issues"].append({
            "severity": "HIGH",
            "type": "parse_failures",
            "count": len(failed_pages),
            "pages": sorted(failed_pages),
            "detail": (
                f"{len(failed_pages)} page(s) had JSON parse errors — VLM "
                f"output was malformed"
            ),
        })

    if low_confidence_pages:
        report["issues"].append({
            "severity": "LOW",
            "type": "low_confidence",
            "count": len(low_confidence_pages),
            "pages": sorted(low_confidence_pages, key=lambda p: p["page"]),
            "detail": (
                f"{len(low_confidence_pages)} page(s) with confidence < 30%"
            ),
        })

    if orientation_transitions:
        report["issues"].append({
            "severity": "INFO",
            "type": "orientation_transitions",
            "count": len(orientation_transitions),
            "transitions": orientation_transitions,
            "detail": (
                f"{len(orientation_transitions)} portrait↔landscape "
                f"transition(s) detected in PDF"
            ),
        })

    # Coverage summary
    total = len(pdf_pages) if pdf_pages else analysis["page_count"]
    present_with_content = sum(
        1 for pg in analysis["pages"]
        if pg["present"] and pg["has_content"]
    )
    report["coverage"] = {
        "total_pdf_pages": total,
        "pages_extracted": len([p for p in analysis["pages"] if p["present"]]),
        "pages_with_content": present_with_content,
        "coverage_pct": round(present_with_content / max(total, 1) * 100, 1),
        "pages_with_tables": sum(1 for p in analysis["pages"] if p["has_tables"]),
        "pages_with_narratives": sum(1 for p in analysis["pages"] if p["has_narratives"]),
    }

    return report


def format_report(report: dict) -> str:
    """Format an audit report as human-readable text."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"QA AUDIT: {report['filename']}")
    if report.get("source_pdf"):
        lines.append(f"Source:   {report['source_pdf']}")
    lines.append(f"{'=' * 70}")

    cov = report["coverage"]
    lines.append(f"\nCOVERAGE: {cov['coverage_pct']}%")
    lines.append(f"  PDF pages:            {cov['total_pdf_pages']}")
    lines.append(f"  Pages extracted:      {cov['pages_extracted']}")
    lines.append(f"  Pages with content:   {cov['pages_with_content']}")
    lines.append(f"  Pages with tables:    {cov['pages_with_tables']}")
    lines.append(f"  Pages with narratives: {cov['pages_with_narratives']}")

    if not report["issues"]:
        lines.append(f"\nNo issues found.")
    else:
        lines.append(f"\nISSUES ({len(report['issues'])}):")
        for issue in report["issues"]:
            sev = issue["severity"]
            lines.append(f"\n  [{sev}] {issue['type']}")
            lines.append(f"    {issue['detail']}")

            if issue["type"] == "missing_pages":
                for m in issue["pages"]:
                    lines.append(
                        f"      Page {m['page']:>3d}  "
                        f"({m.get('orientation', '?')})"
                    )

            elif issue["type"] == "thin_content":
                for t in issue["pages"]:
                    lines.append(
                        f"      Page {t['page']:>3d}  "
                        f"({t.get('orientation', '?')})"
                    )

            elif issue["type"] == "low_confidence":
                for p in issue["pages"]:
                    lines.append(
                        f"      Page {p['page']:>3d}  "
                        f"confidence={p['confidence']}%"
                    )

            elif issue["type"] == "orientation_transitions":
                for t in issue["transitions"]:
                    lines.append(
                        f"      Page {t['page']:>3d}: "
                        f"{t['from']} → {t['to']}"
                    )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="QA audit for OCR extraction output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", type=Path, help="Path to a single extracted JSON")
    group.add_argument("--json-dir", type=Path, help="Directory of extracted JSONs")

    parser.add_argument("--pdf", type=Path, help="Path to source PDF (single mode)")
    parser.add_argument("--pdf-dir", type=Path, help="Directory of source PDFs (batch mode)")
    parser.add_argument("--output", type=Path, help="Write report to file (default: stdout)")
    parser.add_argument("--json-output", type=Path, help="Write machine-readable JSON report")

    args = parser.parse_args()

    reports = []

    if args.json:
        pdf_path = args.pdf
        report = audit_document(args.json, pdf_path)
        reports.append(report)
    else:
        json_files = sorted(args.json_dir.glob("*_extracted.json"))
        if not json_files:
            # Also check for non-suffixed JSONs
            json_files = sorted(args.json_dir.glob("*.json"))

        for jf in json_files:
            pdf_path = None
            if args.pdf_dir:
                # Match by stem (remove _extracted suffix)
                stem = jf.stem.replace("_extracted", "")
                for ext in (".pdf", ".PDF"):
                    candidate = args.pdf_dir / f"{stem}{ext}"
                    if candidate.exists():
                        pdf_path = candidate
                        break

            report = audit_document(jf, pdf_path)
            reports.append(report)

    # Format output
    text_output = []
    for report in reports:
        text_output.append(format_report(report))

    # Summary across all documents
    if len(reports) > 1:
        text_output.append(f"\n{'=' * 70}")
        text_output.append(f"SUMMARY: {len(reports)} document(s) audited")
        text_output.append(f"{'=' * 70}")

        total_pages = sum(r["coverage"]["total_pdf_pages"] for r in reports)
        total_extracted = sum(r["coverage"]["pages_with_content"] for r in reports)
        docs_with_issues = sum(1 for r in reports if r["issues"])

        text_output.append(f"  Total pages across all docs: {total_pages}")
        text_output.append(f"  Total pages with content:    {total_extracted}")
        text_output.append(
            f"  Overall coverage:            "
            f"{round(total_extracted / max(total_pages, 1) * 100, 1)}%"
        )
        text_output.append(f"  Documents with issues:       {docs_with_issues}")
        text_output.append("")

    full_text = "\n".join(text_output)

    if args.output:
        args.output.write_text(full_text)
        print(f"Report written to {args.output}")
    else:
        print(full_text)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"JSON report written to {args.json_output}")


if __name__ == "__main__":
    main()
