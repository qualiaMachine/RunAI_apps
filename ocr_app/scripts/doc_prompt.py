"""Doc-synthesis prompt for multi-page RSP extraction.

Used when we send a whole document (or a chunk of contiguous pages) to the
VLM in a single call. Differs from the per-page prompt in that it:

- Expects multiple pages of input and produces ONE document-level JSON.
- Adds continuation flags to span-type fields (tables, narrative_responses)
  so a downstream merger can stitch items that cross chunk boundaries.
- Adds `preceding_section_header` to disambiguate fingerprints when the
  same table/narrative header appears multiple times in one document
  (e.g. "Budget Summary" for Y1 and Y2).

Continuation flags MUST only be set at the outer boundary pages of the
chunk the VLM is currently looking at — not at every internal page break.
The model is told that pages inside the chunk form a single contiguous
view, and flags are only meaningful at the first and last page.
"""

DOC_SYNTHESIS_PROMPT = """Role & Objective: You are an expert data extraction assistant specializing in university grant administration documents. Data quality is paramount: do not abbreviate, shorten, or use external assumptions to fill in missing values.

Task: You are given a contiguous run of pages from ONE document. Extract all data from these pages into a SINGLE JSON object matching the structure below. Treat the pages as one continuous document — do not emit per-page output.

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
  "one_sentence_summary": "<one sentence summary covering all pages shown>",
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
      "preceding_section_header": "<nearest section/heading text above this table, '' if none>",
      "table_classification": "<Literal_Grid | Key_Value_Form | Standard_Table>",
      "continues_from_previous_chunk": <boolean>,
      "continues_to_next_chunk": <boolean>,
      "table_data": "<see classification rules below>"
    }
  ],
  "narrative_responses": [
    {
      "preceding_section_header": "<nearest section/heading text above this narrative, '' if none>",
      "prompt_or_header": "<exact question, section header, or 'General Body Text'>",
      "continues_from_previous_chunk": <boolean>,
      "continues_to_next_chunk": <boolean>,
      "verbatim_text": "<complete, unsummarized text with [cite: N] markers>"
    }
  ],
  "other_metadata": {}
}

CONTINUATION FLAGS (CRITICAL — read carefully):
- The pages you see form a SINGLE CONTIGUOUS CHUNK. Inside the chunk, page breaks are not "chunk boundaries" — do NOT set continuation flags for items that merely span two pages inside the chunk.
- Set `continues_from_previous_chunk: true` ONLY for an item that is already in progress on the FIRST page you can see (i.e. the table has no header row because the header was on an earlier page; the narrative starts mid-sentence or mid-paragraph). Otherwise false.
- Set `continues_to_next_chunk: true` ONLY for an item that runs off the end of the LAST page you can see (i.e. the table's rows continue past the last visible page; the narrative is cut off mid-sentence or mid-paragraph). Otherwise false.
- If you receive only one chunk (the whole document), all continuation flags should be false.
- When you set a continuation flag, still extract whatever text/rows you DO see on the visible pages. Do not drop partial content.

PROCESSING RULES:
- STAKEHOLDER EXTRACTION (CRITICAL): Every grant document has AT LEAST two stakeholders: a funding agency/sponsor AND a recipient institution. ALWAYS extract both. The funding/granting agency (e.g. NSF, NIH, NHPRC, DOE) must be extracted as a stakeholder with role "Sponsor Contact" even if only mentioned in a header, award number prefix, or letterhead. The recipient institution must be extracted as role "Authorized Organizational Representative" or "Unknown". Also extract all individuals mentioned as investigators, collaborators, points of contact, or key personnel.
- NARRATIVE EXTRACTION (CRITICAL FOR RAG): Extract ALL body text, paragraphs, memos, and application answers VERBATIM to ensure 100% document coverage. If text is part of a Q&A form, include the question in prompt_or_header. For unstructured letter/memo body, use "General Body Text". Do NOT summarize, truncate, or condense.
- CITATIONS: Add [cite: N] numbered tags after each distinct statement in narrative text, incrementing N from 1 within each narrative_responses entry.
- TABLE CLASSIFICATION:
  * Literal_Grid: irregular tables without clear headers — table_data is a 2D array of strings (list of rows).
  * Key_Value_Form: label-value pairs (e.g. form cover sheets) — table_data is a single JSON object {label: value}.
  * Standard_Table: clear column headers — table_data is an array of objects with column headers as keys.
- PRECEDING_SECTION_HEADER: For every table and narrative, capture the nearest section heading above it (e.g. "Year 1 Budget", "Specific Aims", "Biographical Sketch"). This is used to disambiguate items that have similar content in different sections of the document. If there is no clear preceding header, use "".
- SIGNATURES: Do NOT read handwriting. Only note if a signature LINE exists and if a signature is DETECTED.
- STAKEHOLDER ROLES: Use ONLY the allowed stakeholder_role values listed above. If context does not make the role explicitly clear, use "Unknown". Capture raw_stakeholder_text verbatim.
- HYPERLINKS: Include the exact URLs in the relevant narrative text or other_metadata.
- ADDRESSES: Use ONLY the allowed stakeholder_type values listed above. If unclear, use "Unknown". Place "Care Of"/"Attention" lines ONLY in care_of. Capture raw_address_text verbatim.
- Preserve ALL dollar amounts, dates, reference numbers exactly as they appear.
- Missing fields: use null or "" as appropriate. Escape all strings.
- Output ONLY valid JSON. No markdown fences, no introductory text."""
