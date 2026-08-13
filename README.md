# Final Dissertation Report

**Temporal Prediction of In-Trial Adverse Events in Oncology Clinical Trials Using Machine Learning**

## Regenerating

Every figure and table is read from `outputs/`, so the report cannot drift from the notebooks.
Requires `python-docx`, `pillow`, `pywin32` and Microsoft Word.

```bash
python build/make_schematics.py && python build/prep_figs.py && python build/build_report.py && python build/export_pdf.py
```

`make_schematics.py` draws the two methodology diagrams; `prep_figs.py` downsizes the selected
figures from `outputs/figures/` to keep the PDF within the size limit; `build_report.py`
assembles the `.docx`; `export_pdf.py` drives Word to populate the table of contents and page
references before exporting.

Report content lives in `build/content_body*.py` (chapters 1–9) and `build/content_appendix.py`
(appendices, glossary, checklist).
