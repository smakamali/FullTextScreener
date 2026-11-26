# Questionnaire v2: what changed and why (2026-11-26)

v1 is `DataExtraction/backlog_run/questions.json`, the canonical set used for every extraction so far. v2 is `questions.json` in this folder.

## Design constraints
- **Labels unchanged:** every answer option keeps its exact label, so the codebooks, `meta_analysis_v2_src.py` and the merge script work unchanged. The one exception is Q14, which gains the option "Not Applicable", as Q15–Q17 already have it.
- **Question IDs and count unchanged** (25).
- **No pipeline code changed:** the system prompt still says "return Unsure" when unsupported. v2 overrides this per question where "No" is the right answer for absent evidence.
- **Source of each change:** each one targets an error pattern found in the 525-answer check (`../validation/accuracy_check_2026-11-26.md`).

## Changes
| Q | v1 accuracy (used answers) | Problem in v1 | Change in v2 |
|---|---|---|---|
| Q1 | 48% | Similar evidence labelled inconsistently; average metrics treated as robustness measures | Defines each option; average-performance metrics alone do not count |
| Q3 | 81% | Unclear whether metric or loss formulas count as formal content | Counts theorems, lemmas, proofs, bounds; formulas that only define a model or metric do not count |
| Q4 | 95% | Demo paper with no results answered Yes | Requires reported results; demos without results are No |
| Q7 | 67% | Evaluation studies and demos named the evaluated technique; incomplete names | Asks for the paper's own technique and how it works; Not provided for evaluation studies, surveys, demos; no related work |
| Q8 | 71% | Method-internal risk scores, mechanisms and logs listed as metrics; tail metrics missed | Lists only measured quantities reported in results; tail and worst-case examples given; exclusions spelled out |
| Q10 | 71% | TPC-H counted as real; a query workload named "synthetic" counted as synthetic data | Classifies datasets, not workloads; examples of real and synthetic; generated queries over real data count as Real |
| Q11 | 57% | Hardware, hyperparameters and splits listed | Restricted to data, query and plan characteristics; exclusions spelled out |
| Q12 | 67% | Ordinary performance comparisons and demos counted as robustness experiments | Requires adverse or shifted conditions, or worst-case/tail/regression results; Not Applicable without experiments |
| Q14–Q17 | 91–100% | Non-ML papers often got Other or Unsure | Explicit Not Applicable for non-ML papers; Q15 defines Regression; Q16 counts pre-trained LLMs; Q17 defines transfer learning, including domain adaptation |
| Q21 | 45% | Unsure where the paper is clear | Defines OOD concerns with examples; No if not discussed; Unsure only if ambiguous |
| Q22 | 73% | Missed an unseen-database test; Unsure instead of No | Defines OOD experiments with examples; No if none; Unsure only if ambiguous |
| Q23 | 64% | Other for graph attention, self-attention and LLM encoders | Maps GAT to GNN and self-attention/LLM to Transformer; allows several |
| Q24 | 64% | Unsure, or No despite stated robustness motivation | Yes if a stated design reason is robustness (with examples), otherwise No |
| Q25 | 27% | Listed main-approach components, related work, ablations, future work | Separate additional techniques only, with examples and exclusions |

Unchanged: Q2, Q5, Q6, Q9, Q13, Q18, Q19, Q20.

## Test of v2.0 (2026-11-26)
- **Papers:** the six weakest from the v1 check: 165, 166, 171, 173, 182, 183. Run in `../test_questionnaire_v2/`.
- **Scope of the check:** the 75 answers to the changed questions that the meta-analysis uses (ML-only questions only for the three ML papers). Same judge and C/P/I criteria as the v1 check.
- **Verdicts:** `../validation/v2_verdicts_2026-11-26.csv`.

| | Correct | Partial | Wrong |
|---|---|---|---|
| v1 | 39 (52%) | 26 (35%) | 10 (13%) |
| v2.0 | 57 (76%) | 15 (20%) | 3 (4%) |

- **Fixed:** most Unsure/No errors on Q21, Q22 and Q24; Q23 GAT; Q25 over-listing; Q7 on evaluation studies; Q12 on 182.
- **Regressions (4):**
  - **Prompt examples copied as answers:** a 99th-percentile q-error (165 Q8), and joins, selectivity and templates (173 Q11). "Scale factor" was also added for 182 Q11.
  - **171 Q10:** "Unsure" for simulated data.
  - **171 Q3:** "No", although the decision-theoretic derivation is borderline formal analysis.
- **Still weak:** Q1 (182 and 183 still "Quantitative" for averages and speedups), Q7 on 166 (names its evaluation framework), Q21 on 173 (a related-work mention counted).

## v2.1 (applied after the test; the v2.0 file is kept as `questions_v2.0.json`)
| Q | Added |
|---|---|
| Q1 | Only averages: "Mentioned but not measured" if robustness is claimed, otherwise "Not provided" |
| Q7 | An evaluation framework or methodology is not a robustness technique |
| Q8, Q11 | The examples are illustrations, not a checklist; list only what appears in, or is varied by, the experiments |
| Q10 | Simulated data counts as Synthetic |
| Q21 | Mentions only in related work do not count |

## Test of v2.1 (2026-11-26)
- **Same 6 papers and 75 answers;** run in `../test_questionnaire_v21/`. Verdicts: `../validation/v21_verdicts_2026-11-26.csv`.

| | Correct | Partial | Wrong |
|---|---|---|---|
| v1 | 39 (52%) | 26 (35%) | 10 (13%) |
| v2.0 | 57 (76%) | 15 (20%) | 3 (4%) |
| v2.1 | 54 (72%) | 18 (24%) | 3 (4%) |

- **Worked:** Q8 (165 no longer invents a 99th-percentile q-error); Q10 (171 simulated data is Synthetic).
- **Did not work:** Q11 (173 and 182 still list characteristics that aren't varied); Q7 (166 still names its evaluation framework); Q21 (173 still counts related work).
- **Backfired:** Q1. 165 dropped from Quantitative to "Mentioned but not measured" although it defines risk metrics; 182 and 183 are unchanged.
- **Run-to-run variation:** on the 90 categorical answers whose wording is identical in v2.0 and v2.1, 8 (9%) changed between the two runs, at temperature 0 (for example 165 Q13 and Q17). Differences of a few answers between v2.0 and v2.1 are within this noise, so v2.0 and v2.1 perform about the same; both are clearly better than v1.
- **Format:** 3 of 150 answers came back as unparsed JSON or a nested object (v2.0: 1). `merge_results.py` repairs JSON strings.

## Final: v2.1 with the v2.0 Q1 (adopted 2026-11-26)
- **`questions.json` in this folder** is v2.1 with the Q1 addition reverted to the v2.0 wording. It differs from v2.0 only in Q7, Q8, Q10, Q11 and Q21. The v2.0 file is kept as `questions_v2.0.json`.
- **Canonical:** `survey-conventions/scripts/common.py` (`QUESTIONS`) now points here, so `prepare_run.py` copies it into new runs and `merge_results.py` requires it.
- **v1 is kept:** `backlog_run/questions.json` stays as the record of the questionnaire behind all extractions before 2026-11-26. The records in `consolidated_output.json` keep v1 until the full re-extraction replaces them.
- **Not tested as a whole:** the final file was not run again. Its Q1 is the tested v2.0 wording, and its other questions are the tested v2.1 wording.

## Effect on reporting if adopted
- **Q4:** demos without results move from experimental to non-experimental.
- **Q3:** papers whose only formal content is a definition of a metric or model become No.
- **Q8:** the robustness-oriented-metric share should drop slightly, because method-internal risk scores are no longer listed.
- **Methodology and Appendix:** the text says the question wording was refined after the evaluation, which stays true.
