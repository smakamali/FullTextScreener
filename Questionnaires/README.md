# Questionnaires

The 25-question questionnaire the pipeline answers for each paper. To use one, point `questions_file_path` in `Config/ConfigLitScr.cfg` to it.

| File | Status |
|---|---|
| `questions_v3.1.json` | **Current (2026-11-26).** v3.0 with Q13 (methods a paper only evaluates are not proposed by it) and Q9 (stricter canonical benchmark names) tightened after its test. |
| `questions_v3.0.json` | Tested 2026-11-26 on 7 papers; superseded by v3.1. Every categorical label defined; Q7, Q18, Q19 and Q25 changed from free text to fixed options; Q13 means "proposes an approach that uses machine learning" (evaluation studies are not ML papers); Q9 uses canonical benchmark names; Q2 records the scope of robustness. Not comparable with v2.1 on those questions. |
| `questions_v2.1.json` | Superseded by v3.0 the same day. Tightened wording: definitions, exclusions, "No" instead of "Unsure" when a paper does not discuss the topic, and "Not Applicable" for non-ML papers. Answer labels are unchanged except that Q14 gains "Not Applicable". |
| `questions_v2.0.json` | First tightened draft. It was tested, then superseded by v2.1. |
| `questions_v1.json` | The questionnaire behind all survey extractions before 2026-11-26. |

`CHANGES.md` gives the reason for each change and the test results. Its relative paths (`../validation/…`, `backlog_run/…`) refer to the survey's `DataExtraction` folder, which is not part of this repository.
