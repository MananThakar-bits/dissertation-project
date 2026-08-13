# Final Dissertation Report

**Temporal Prediction of In-Trial Adverse Events in Oncology Clinical Trials Using Machine Learning**
Thakar Manan Pradipbhai — 2024DA04324 — BITS ZG628T

## Deliverables

| File | Purpose |
|---|---|
| `2024DA04324_Final_Dissertation_Report.pdf` | **The submission file.** 85 pages, 4.80 MB |
| `2024DA04324_Final_Dissertation_Report.docx` | Editable source, for adding signatures or supervisor edits |

## Before submitting

Everything is filled in except the handwritten parts. What remains:

- **Signatures and their dates** — Certificate (supervisor and student), Abstract Sheet
  (student and supervisor), and the Checklist declaration.

Sign the printed pages and scan only those, or apply digital signatures to the PDF. The
guidelines allow signature pages to be scanned images; the rest of the report must stay as
selectable text, so do not scan the whole document.

If you edit the `.docx` instead, press `Ctrl+A` then `F9` before exporting so the table of
contents and page references stay correct, then re-export via File → Save as → PDF or by
re-running `build/export_pdf.py`.

## Project particulars recorded in the report

| Field | Value |
|---|---|
| Organisation | Jade Global Software Pvt. Ltd., Pune |
| Supervisor | Mr. Kishor Gund, Principal Consultant |
| Additional Examiner | Mr. Alu Alex John, Assistant Principal Consultant |
| Faculty Mentor | Vinaya Sathyanarayana, BITS Pilani WILP Division |
| Date of start | 25 April 2026 |
| Date of submission | 2 August 2026 |
| Duration | 14 weeks |

These are defined once as constants at the top of `build/build_report.py`; change them
there and rebuild rather than editing the `.docx` by hand.

## Conformance with the submission guidelines

| Requirement | Status |
|---|---|
| PDF format, 10 MB or less | 4.80 MB |
| Text is selectable, not scanned images | 129,890 characters extractable |
| Page size 9 × 11 in, 1 in margins | Yes |
| Double-spaced body text | Yes (tables, captions and the abstract sheet are single-spaced for legibility) |
| Roman numerals before the Introduction, Arabic from Chapter 1 | Yes — Chapter 1 begins on page 1 |
| Figure numbers and captions **below** figures | 35 figures |
| Table numbers and captions **above** tables | 29 tables |
| All references cited in the body | 19 of 19 verified |
| Checklist as the final page | Yes |

## Regenerating the report

The report is built from the analysis outputs, so the numbers cannot drift from the
notebooks. Requires `python-docx`, `pillow`, `pywin32` and Microsoft Word.

```bash
python build/make_schematics.py
```

```bash
python build/prep_figs.py
```

```bash
python build/build_report.py
```

```bash
python build/export_pdf.py
```

`make_schematics.py` draws the two methodology diagrams, `prep_figs.py` downsizes the
35 selected figures from `outputs/figures/` to keep the PDF under the size limit,
`build_report.py` assembles the `.docx`, and `export_pdf.py` drives Word to populate the
table of contents and page references before exporting the PDF.

Report content lives in `build/content_body*.py` (chapters 1–9) and
`build/content_appendix.py` (appendices, glossary, checklist).
