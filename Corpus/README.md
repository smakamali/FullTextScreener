# Survey corpus

`survey_corpus.csv` lists every paper extracted for the robust query optimization survey (as of 2026-11-26): 179 papers, of which 172 are in the meta-analysis (`status = Included`) and 7 were excluded after extraction (`status = Excluded`).

| Column | Content |
|---|---|
| `paper_id` | Id used in the extraction results |
| `pdf_filename` | File name the pipeline expects in `input_pdf_folder` |
| `title`, `year` | From the extraction metadata |
| `authors` | From the extraction metadata, or from the survey's bibliography where that was missing. Formats differ between the two sources. Empty for two excluded papers not in the bibliography. |
| `status` | `Included` or `Excluded` |
| `bib_key`, `doi` | The paper's entry in the survey's `references.bib` (DOI where the entry has one) |

The PDFs are not part of this repository. `Config/metadata.csv` is a separate 30-paper list used for the earlier pipeline evaluations.

**Papers with replaced PDFs:**
- **121 (STHoles):** the extraction PDF is the Microsoft Research technical report MSR-TR-2001-36.
- **55 (*Least expected cost query optimization: what can we expect?*):** the extraction PDF is an OCR'd copy, because the original's text layer is unreadable.
