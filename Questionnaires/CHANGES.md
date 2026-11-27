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

## Q2 fix (2026-11-26, after the evidence-first test)
- **Problem:** with the Evidence → Reasoning → ShortAnswer order (FullTextScreener 0.9.1), Q2 named a component on 182 and 165 although the model's own reasoning said the paper does not define robustness.
- **Change:** Q2 now asks first whether the paper itself defines robustness, and says what counts: a statement of what robustness means, formally or in words. Using the word, claiming a method is robust, or making a component more robust does not count. Without a definition the answer is "No Definitions Provided". "Unsure" is only for a defined robustness whose part of the system is ambiguous. Labels unchanged. The previous file is kept as `questions_v2.1_before_Q2fix.json`.
- **Test:** Q2 only, on 182, 165 and 183 (`../q2fix_test/`, questionnaire with Q2 alone). Verdicts: `../validation/q2fix_verdicts_2026-11-26.csv`.

| Paper | Evidence-first answer | After the fix |
|---|---|---|
| 182 | join ordering (I) | plan optimization (I) |
| 165 | cost model (I) | No Definitions Provided (C) |
| 183 | DBMS (end-to-end) (C) | DBMS (end-to-end) (C) |

- **Result:** 2 of 3 correct (was 1 of 3). 182 still contradicts its own reasoning ("does not provide a formal or standalone definition", then a component).

## Q2 reframed as scope (2026-11-26, user decision)
- **Decision:** Q2 records which part of the system the paper's robustness applies to, whether the paper states a definition (new or existing) or only implies one. Novelty does not matter. This matches how the earlier accuracy checks graded Q2. The strict "defines robustness" version above is kept as `questions_v2.1_Q2strict.json`.
- **Wording:** scope of the robustness the paper pursues or evaluates; the most specific part (with join-ordering and cost-model examples); plan optimization only for plan choice as a whole; "No Definitions Provided" only if robustness is neither defined nor pursued. Labels unchanged.
- **Under this reading** the evidence-first answers "join ordering" (182) and "cost model" (165) are correct, so the Q2 "regression" in the evidence-first test was the strict grading, not the model.
- **Test:** Q2 only, 182, 165, 183 (`../q2scope_test/`). Verdicts: `../validation/q2scope_verdicts_2026-11-26.csv`.

| Paper | Expected | Answer | Verdict |
|---|---|---|---|
| 182 | join ordering | plan optimization | P |
| 165 | cost model | cost model | C |
| 183 | DBMS (end-to-end) | plan optimization | P |

- **Result:** reasoning and answer now agree on all three, but "plan optimization" acts as a catch-all for 182 and 183, despite the "most specific part" instruction.

## Q2 with label meanings (2026-11-26, adopted)
- **Change:** Q2 (scope wording above) now gives a one-line meaning for every label and says to choose the most specific label that fits. "plan optimization" is limited to choosing among complete plans when robustness is not tied to a narrower part; "DBMS (end-to-end)" covers run-time execution (adaptive execution, fallbacks) and the system as a whole. Labels unchanged. The previous version is kept as `questions_v2.1_Q2scope_nodefs.json`.
- **Test:** Q2 only, 182, 165, 183 (`../q2labels_test/`). Verdicts: `../validation/q2labels_verdicts_2026-11-26.csv`.

| Paper | Expected | Answer | Verdict |
|---|---|---|---|
| 182 | join ordering | join ordering | C |
| 165 | cost model | cost model | C |
| 183 | DBMS (end-to-end) | DBMS (end-to-end) | C |

- **Result:** 3 of 3 correct (scope wording without meanings: 1 C, 2 P). Each reasoning cites the matching label meaning.
- **Caveat:** three papers, one run each; about 9% of categorical answers change between identical runs.

## Q13 and the ML-only questions (2026-11-26, user decision; not yet tested)
- **Decision:** a paper is ML-based only if the approach it itself proposes uses machine learning, as the paper's Methodology already defines it. Evaluation and benchmark studies of learned methods (e.g., 165, 166) stay in the meta-analysis but outside the ML subset.
- **Q13:** "Does the paper propose an approach that uses machine learning?"; Yes includes approaches relying on a pre-trained model such as an LLM; No if the proposed approach does not use ML or the paper proposes no approach of its own.
- **Q14–Q25:** every "Not Applicable" clause now reads "if the paper does not propose an approach that uses machine learning"; Q15 gains that clause (it had none). Labels unchanged. The previous file is kept as `questions_v2.1_before_Q13fix.json`.

## Q6 and Q20 with label meanings (2026-11-26, user-approved wording; not yet tested)
- **Q6:** each label defined. Primary = robustness is the main stated goal or a headline contribution (title, abstract, contribution list); Secondary = main goal is something else but robustness benefits are claimed, discussed or shown; Not addressed = not discussed; Unsure only if ambiguous.
- **Q20:** Yes = the method estimates the uncertainty of its own predictions (intervals, variance, confidence, Bayesian or ensemble disagreement) or acts on it (fallback, filtering or reweighting low-confidence predictions, risk-based plan choice); No = only notes that estimates can be wrong, or reports errors without estimating or using uncertainty; Unsure only if ambiguous; Not Applicable if no proposed ML approach. Before: 36 corpus papers answered Unsure.
- **Q5:** kept unchanged (user decision): it answers a different question from Q3 and Q4, even though the meta-analysis does not use it.
- Labels unchanged. The previous file is kept as `questions_v2.1_before_Q6Q20fix.json`.

## Label meanings for the remaining categorical questions (2026-11-26, user-approved; not yet tested)
- **Scope:** every categorical question now defines each of its labels. Open-ended and list questions (Q7, Q8, Q9, Q11, Q18, Q19, Q25) are unchanged. Labels unchanged.
- **Rewritten:** Q3, Q4, Q5, Q14, Q15, Q16, Q17, Q23 (full text in `proposed_label_definitions_2026-11-26.md`).
- **Sentences added:** Q10 (Not provided, Unsure), Q12 (No, Unsure), Q13 (Unsure), Q24 (Unsure).
- **Calls approved with the wording:** Q5 gives both labels separated by a semicolon when a paper evaluates robustness both ways; Q14 counts a prompted LLM as Other; Q14 and Q15 list each type or task when an approach combines several; Q17 counts pre-trained LLMs and embeddings as transfer learning, but not retraining on the same database; Q23 puts random forests and single decision trees under BDT.
- **Analysis code:** `normalize()` in `meta_analysis_v2_src.py` now also splits on ';', so a multi-label answer maps to its first label as comma-separated ones already did.
- The previous file is kept as `questions_v2.1_before_labeldefs.json`.

## Q9 with canonical names (2026-11-26, user-approved; not yet tested)
- **Problem:** the same benchmark appeared under several spellings ("JOB", "JOB benchmark", "Join Order Benchmark", "IMDB", "IMDb dataset"), and 'Not provided' and 'Not Applicable' were used interchangeably (40 and 41 corpus answers).
- **Change:** asks for benchmarks, workloads or datasets; gives canonical names (JOB, CEB, STATS-CEB, Stack, DSB, TPC-H, TPC-DS, SSB, JCC-H, IMDb, DMV, Census, Forest, Power, Kddcup98, DBLP); all Join Order Benchmark variants (JOB-light, JOB-light-ranges, JOB-M, ...) are written 'JOB' (user decision); a workload is named rather than the dataset it runs on, so 'JOB' does not also add IMDb to "other real datasets" in the codebook; self-generated data is 'synthetic'; Not provided = experiments without named data, Not Applicable = no experiments.
- The Q9 codebook rules already match every canonical name, so the analysis code is unchanged. The previous file is kept as `questions_v2.1_before_Q9fix.json`.

## Q7 as strategy options; Q25 as ML-practice options (2026-11-26, user-approved; not yet tested)
- **Why:** in free text, Q7 could not be used in the meta-analysis, and the taxonomy class already comes from where each paper is cited. Q7 now records *how* robustness is achieved, a dimension the component taxonomy does not capture. Q25 (27% correct in the v1 check) mostly failed on the "additional, separate" boundary.
- **Q7 (all papers):** one or more of 11 strategy options plus Other, main first, including techniques added on top of the main method: Run-time adaptation, Worst-case guarantees, Risk-aware plan selection, Uncertainty estimation, Robust operators, More accurate estimates, Feedback-driven correction, Generalization and adaptation, Regression safeguards, Improved plan search, Robustness quantification only, Other; then ' — ' and the method name; 'Not provided' unchanged. The options were derived from the v1 Q7 answers of all 168 analysis papers. With several options, the analysis's `normalize()` takes the first (main) one.
- **Q25 (ML papers):** no longer "additional techniques"; lists which ML robustness practices the proposed approach uses, from fixed options: Retraining or incremental adaptation, Transfer or pre-training, Data augmentation, Ensembles, Uncertainty or OOD detection, Fallback or guard, Regularization, Other; Not provided; Not Applicable. Every option matches a category of the existing Q25 codebook, so the analysis code is unchanged.
- **Dropped:** the planned injection of Q7's answer into Q25 (no longer needed, so no pipeline change).
- **Not yet in the analysis:** Q7 is not reported; adding it (counts, by era) is a later step.
- The previous file is kept as `questions_v2.1_before_Q7Q25fix.json`.

## Q18 and Q19 as options (2026-11-26, user-approved; not yet tested)
- **Source:** the v1 Q18 and Q19 answers of the 88 ML papers.
- **Q18 (training data; several allowed):** Synthetic query generation, Benchmark or production workload, Execution feedback, Optimizer-generated, Data sampling, Synthetic datasets, Data augmentation, Other; Not provided (also for vague descriptions); Not Applicable.
- **Q19 (input encoding; several allowed):** Feature vectors, Sets, Trees, Graphs, Sequences or text, Data tuples, Other; Not provided; Not Applicable. "Data tuples" covers data-driven estimators, which v1 often answered Not Applicable.
- **Analysis code (`meta_analysis_v2_src.py`, Q18 rules):** new categories 'optimizer-generated' and 'synthetic datasets'; 'synthetic query generation' no longer matches "synthetic dataset(s)"; 'data sampling' matches its label; "simulation learning" moved from execution feedback to optimizer-generated. Every Q18 option now maps to exactly one category. Backup: `preserved/2026-11-26/meta_analysis_v2_src_before_Q18rules.py`. This can shift the current (v1) Q18 shares slightly; the notebook was not re-run. Q19 is still not reported.
- The previous file is kept as `questions_v2.1_before_Q18Q19fix.json`.

## v3.0 (2026-11-26)
- **`questions.json` is now v3.0**, also saved as `questions_v3.0.json`. It contains every change above since "Final: v2.1 with the v2.0 Q1": Q2 as scope with label meanings; Q6, Q20 and all other categorical questions with a meaning for every label; Q13 and the ML-only clauses ("proposes an approach that uses machine learning"); Q9 with canonical benchmark names; Q7 as robustness-strategy options; Q25 as ML-practice options; Q18 and Q19 as options.
- **Why a major version:** Q7, Q18, Q19 and Q25 changed from free text to fixed options, and Q13 changed meaning (evaluation studies are no longer ML papers). v1 and v2.1 answers are not directly comparable on those questions.
- **Test:** the 7-paper test (`../v3test_7papers/`: 182, 38, 94, 114, 183, 131, 165) uses this file.

## v3.1 (2026-11-26): Q13 and Q9 tightened after the v3.0 test
- **v3.0 test** (`../v3test_7papers/`; verdicts `../validation/v30_7papers_verdicts_2026-11-26.csv`): all 175 answers parsed. Q7, Q14–Q24 and Q25 (graded against the v3.0 wording: the whole approach's ML practices) mostly correct; Q18 and Q19 give clean options; the two non-ML papers get Not Applicable on all ML-only questions.
- **Q13:** 165 (an evaluation study of learned cost models) still answered Yes, calling the evaluated models "proposed", although its own Q7 says Not provided; this wrongly filled its 12 ML-only answers. Added: "Methods the paper only evaluates or compares are not proposed by it." Not retested (user decision): the few evaluation studies are curated after the full rerun (see the merge-extraction skill).
- **Q9:** canonical names were ignored on 2 of 7 papers (165 "JOB-light, IMDB"; 114 "Join Order Benchmark"). The JOB rule now names the forbidden spellings, says not to list IMDb when JOB is listed, and asks for a final check that replaces non-canonical names.
- Saved as `questions_v3.1.json`; `questions.json` is v3.1.
- **Q9 retest (v3.1, Q9 only, 165 and 114; `../q9test_v31/`):** both now canonical: 165 "JOB, Baseball, TPC-H" (was "JOB-light, IMDB, Baseball, TPC-H"); 114 "..., JOB" (was "Join Order Benchmark").
