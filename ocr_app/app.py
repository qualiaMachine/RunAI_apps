#!/usr/bin/env python3
"""Streamlit UI for document extraction.

Upload PDFs, TIFFs, or images to extract structured data using a VLM.
All pages are rendered as images and sent to the model for structured
JSON extraction matching the grant admin schema.

Launch:
    streamlit run ocr_app/app.py

Environment variables:
    OCR_SERVICE_URL  - URL of the extraction server (default: http://localhost:8090)
"""

import io
import json
import os
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://localhost:8090")

# Default extraction prompt — same as notebook batch processing
DEFAULT_PROMPT = """Role & Objective: You are an expert data extraction assistant specializing in university grant administration documents. Data quality is paramount: do not abbreviate, shorten, or use external assumptions to fill in missing values.

Task: Extract all data from this single document page into the JSON structure below. Do not analyze data.

{
  "confidence_percentage": <float 0-100>,
  "confidence_narrative": "<brief note on extraction quality and comprehensiveness>",
  "has_annotation": <boolean>,
  "has_watermark": <boolean>,
  "signature_lines": {
    "has_signature_line": <boolean>,
    "has_valid_signature": <boolean>
  },
  "document_tags": ["<high-level grant admin tags, e.g. IRB, IACUC, Biosafety>"],
  "one_sentence_summary": "<one sentence summary>",
  "document_details": {
    "application_id": "", "application_type": "", "title": "",
    "requested_amount": null, "completed_date": "", "sub_document_type": ""
  },
  "stakeholders": [
    {
      "context_snippet": "<3-5 words near the stakeholder info>",
      "stakeholder_role": "<Principal Investigator | Co-Investigator | Collaborator | Key Personnel | Grants Administrative Contact | Sponsor Contact | Authorized Organizational Representative | Unknown>",
      "full_name": "", "first_name": "", "last_name": "",
      "email": "", "phone": "", "institution": "", "department": "",
      "position_title": "", "highest_education": "",
      "raw_stakeholder_text": "<verbatim text block containing stakeholder info>"
    }
  ],
  "addresses": [
    {
      "context_snippet": "<3-5 words near the address>",
      "addressee": "", "care_of": null,
      "address_line1": "", "address_line2": "",
      "city": "", "state_province": "", "postal_code": "", "country": "",
      "stakeholder_type": "<Funding Agency | Grantee Institution | Subrecipient | Principal Investigator | Grants Administrative Contact | Unknown>",
      "raw_address_text": "<verbatim text block containing the full address>"
    }
  ],
  "tables": [
    {
      "table_classification": "<Literal_Grid | Key_Value_Form | Standard_Table>",
      "table_data": "<see classification rules below>"
    }
  ],
  "narrative_responses": [
    {
      "prompt_or_header": "<exact question, section header, or 'General Body Text'>",
      "verbatim_text": "<complete, unsummarized text with [cite: N] markers>"
    }
  ],
  "other_metadata": {}
}

PROCESSING RULES:
- STAKEHOLDER EXTRACTION (CRITICAL): Every grant document has AT LEAST two stakeholders: a funding agency/sponsor AND a recipient institution. ALWAYS extract both. The funding/granting agency (e.g. NSF, NIH, NHPRC, DOE) must be extracted as a stakeholder with role "Sponsor Contact" even if only mentioned in a header, award number prefix, or letterhead. The recipient institution must be extracted as role "Authorized Organizational Representative" or "Unknown". Also extract all individuals mentioned as investigators, collaborators, points of contact, or key personnel.
- NARRATIVE EXTRACTION (CRITICAL FOR RAG): Extract ALL body text, paragraphs, memos, and application answers VERBATIM to ensure 100% document coverage. If text is part of a Q&A form, include the question in prompt_or_header. For unstructured letter/memo body, use "General Body Text". Do NOT summarize, truncate, or condense.
- CITATIONS: Add [cite: N] numbered tags after each distinct statement in narrative text, incrementing N from 1.
- TABLE CLASSIFICATION:
  * Literal_Grid: irregular tables without clear headers — table_data is a 2D array of strings (list of rows).
  * Key_Value_Form: label-value pairs (e.g. form cover sheets) — table_data is a single JSON object {label: value}.
  * Standard_Table: clear column headers — table_data is an array of objects with column headers as keys.
- SIGNATURES: Do NOT read handwriting. Only note if a signature LINE exists and if a signature is DETECTED.
- STAKEHOLDER ROLES: Use ONLY the allowed stakeholder_role values listed above. If context does not make the role explicitly clear, use "Unknown". Capture raw_stakeholder_text verbatim.
- HYPERLINKS: If hyperlinks are provided in the context above, include them in the relevant narrative text or other_metadata. Preserve the exact URL.
- ADDRESSES: Use ONLY the allowed stakeholder_type values listed above. If unclear, use "Unknown". Place "Care Of"/"Attention" lines ONLY in care_of. Capture raw_address_text verbatim.
- Preserve ALL dollar amounts, dates, reference numbers exactly as they appear.
- Missing fields: use null or "" as appropriate. Escape all strings.
- Output ONLY valid JSON. No markdown fences, no introductory text."""

st.set_page_config(
    page_title="Document Extraction",
    page_icon="\U0001F4C4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def _check_server() -> dict | None:
    try:
        resp = httpx.get(f"{OCR_SERVICE_URL}/health", timeout=5.0)
        return resp.json()
    except Exception:
        return None


def _extract_pdf(file_bytes: bytes, filename: str,
                 prompt: str, max_tokens: int, pages: str | None) -> dict:
    files = {"file": (filename, file_bytes)}
    data = {"format": "json", "max_tokens": str(max_tokens), "prompt": prompt}
    if pages:
        data["pages"] = pages
    resp = httpx.post(
        f"{OCR_SERVICE_URL}/extract/pdf", files=files, data=data, timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_image(file_bytes: bytes, filename: str,
                   prompt: str, max_tokens: int) -> dict:
    files = {"file": (filename, file_bytes)}
    data = {"format": "json", "max_tokens": str(max_tokens), "prompt": prompt}
    resp = httpx.post(
        f"{OCR_SERVICE_URL}/extract/image", files=files, data=data, timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def _format_elapsed(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F4C4 Document Extraction")
    st.caption("Grants, Archives & Institutional Records")

    status = _check_server()
    if status and status.get("status") == "ok":
        llm = status.get("llm_model", "unknown")
        st.success(f"Model: {llm}")
    else:
        st.error(f"Server unreachable at {OCR_SERVICE_URL}")

    st.divider()

    with st.expander("Extraction prompt", expanded=False):
        extraction_prompt = st.text_area(
            "Edit prompt (applied to every page)",
            value=DEFAULT_PROMPT,
            height=400,
        )

    with st.expander("Advanced options"):
        max_tokens = st.slider(
            "Max output tokens", 256, 16000, 16000, 256,
        )
        pdf_pages = st.text_input(
            "PDF pages (optional)",
            placeholder="e.g., 1-5 or 1,3,5",
        )

    st.divider()
    st.caption(f"Server: `{OCR_SERVICE_URL}`")
    st.caption("All pages processed as images via VLM")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Document Extraction")
st.markdown(
    "Upload grant award notices, budgets, terms & conditions, archival "
    "documents, or other institutional records. All pages are rendered "
    "as images and sent to the VLM for structured JSON extraction."
)

uploaded_files = st.file_uploader(
    "Upload PDFs, TIFFs, or images",
    type=["pdf", "tiff", "tif", "png", "jpg", "jpeg", "webp", "bmp", "gif"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more files to get started.")
    st.stop()

run = st.button("Extract", type="primary", use_container_width=True)

if not run:
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            if f.type == "application/pdf":
                st.markdown(f"**{f.name}** (PDF, {len(f.getvalue()) / 1024:.0f} KB)")
            else:
                img = Image.open(io.BytesIO(f.getvalue()))
                st.image(img, caption=f.name, use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Process files
# ---------------------------------------------------------------------------
prompt = extraction_prompt.strip()

for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.getvalue()
    is_pdf = uploaded_file.type == "application/pdf"

    st.divider()
    st.subheader(uploaded_file.name)

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            if is_pdf:
                result = _extract_pdf(
                    file_bytes, uploaded_file.name,
                    prompt, max_tokens, pdf_pages or None,
                )

                # Summary metrics
                total_ms = result.get("total_elapsed_ms", 0)
                total = result.get("total_pages", 0)

                col_stats = st.columns(3)
                col_stats[0].metric("Pages", total)
                col_stats[1].metric("Time", _format_elapsed(total_ms))
                col_stats[2].metric("Avg/page", _format_elapsed(total_ms / max(total, 1)))

                # Per-page results
                for page_result in result.get("pages", []):
                    page_num = page_result["page"]
                    elapsed = page_result["elapsed_ms"]
                    label = f"Page {page_num} ({_format_elapsed(elapsed)})"

                    with st.expander(label, expanded=(page_num == 1)):
                        st.code(page_result["text"], language="json")

                # Combine all pages for download
                all_text = "\n\n".join(
                    p["text"] for p in result.get("pages", [])
                )

            else:
                # Image / TIFF
                col_img, col_result = st.columns([1, 1])
                with col_img:
                    img = Image.open(io.BytesIO(file_bytes))
                    st.image(img, caption=f"{img.width}x{img.height}", use_container_width=True)

                result = _extract_image(
                    file_bytes, uploaded_file.name,
                    prompt, max_tokens,
                )

                with col_result:
                    elapsed = result.get("elapsed_ms", 0)
                    st.caption(f"VLM extraction ({_format_elapsed(elapsed)})")
                    st.code(result["text"], language="json")

                all_text = result["text"]

            # Download buttons
            st.download_button(
                f"Download JSON — {uploaded_file.name}",
                data=all_text,
                file_name=f"{Path(uploaded_file.name).stem}_extracted.json",
                mime="application/json",
            )

            # Also offer JSONL format
            st.download_button(
                f"Download JSONL — {uploaded_file.name}",
                data=all_text.replace("\n\n", "\n"),
                file_name=f"{Path(uploaded_file.name).stem}_extracted.jsonl",
                mime="application/json",
                key=f"jsonl_{uploaded_file.name}",
            )

        except httpx.HTTPStatusError as e:
            st.error(f"Server error: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError:
            st.error(f"Cannot connect to server at {OCR_SERVICE_URL}")
        except Exception as e:
            st.error(f"Error: {e}")
