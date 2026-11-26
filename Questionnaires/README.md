# Questionnaires

The 25-question questionnaire the pipeline answers for each paper. To use one, point `questions_file_path` in `Config/ConfigLitScr.cfg` to it.

| File | Status |
|---|---|
| `questions_v2.1.json` | **Current (2026-11-26).** Tightened wording: definitions, exclusions, "No" instead of "Unsure" when a paper does not discuss the topic, and "Not Applicable" for non-ML papers. Answer labels are unchanged except that Q14 gains "Not Applicable". |
| `questions_v2.0.json` | First tightened draft. It was tested, then superseded by v2.1. |
| `questions_v1.json` | The questionnaire behind all survey extractions before 2026-11-26. |

`CHANGES.md` gives the reason for each change and the test results. Its relative paths (`../validation/…`, `backlog_run/…`) refer to the survey's `DataExtraction` folder, which is not part of this repository.
