# Survey corpus

`survey_corpus.csv` lists every paper extracted for the robust query optimization survey (as of 2026-11-26): 179 papers, of which 168 are in the meta-analysis (`status = Included`) and 11 were excluded after extraction (`status = Excluded`): 7 cited only for a general technique or excluded by the authors, and 4 secondary studies (surveys, tutorials, a historical overview; excluded 2026-11-26).

| Column | Content |
|---|---|
| `paper_id` | Id used in the extraction results |
| `pdf_filename` | File name the pipeline expects in `input_pdf_folder` |
| `title`, `year` | From the extraction metadata |
| `authors` | From the extraction metadata, or from the survey's bibliography where that was missing. Formats differ between the two sources. Empty for two excluded papers not in the bibliography. |
| `status` | `Included` or `Excluded` |
| `bib_key`, `doi` | The paper's entry in the survey's `references.bib` (DOI where the entry has one) |

The PDFs are not part of this repository. `Config/metadata.csv` is a separate 30-paper list used for the earlier pipeline evaluations.

**PDFs to replace for the re-extraction:** the original PDFs of two papers have unreadable text layers, so their current extractions are unreliable.
- **121 (STHoles):** use the Microsoft Research technical report MSR-TR-2001-36.
- **55 (*Least expected cost query optimization: what can we expect?*):** use an OCR'd copy of the original.
