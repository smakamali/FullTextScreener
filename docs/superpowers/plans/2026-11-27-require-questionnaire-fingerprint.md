# Require a Questionnaire Fingerprint for Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With `skip_existing = True`, never reuse a saved result that does not record which questionnaire produced it.

**Architecture:** `checkCompleteResult` in `FullTextScreener.py` currently accepts a complete result without `questionnaire_sha256` ("written before the fingerprint was added"). Since questionnaires v2.1, v3.0 and v3.1 all use Q1–Q25, such a result can carry answers to a different questionnaire. After this change a missing fingerprint makes the result non-reusable, so the paper is redone. This is option 1 from review comment 3 on PR #5, chosen by the user on 2026-11-27.

**Tech Stack:** Python 3, `unittest` (`tests/test_offline.py`).

**Spec:** no separate document. The decision is recorded above and in Global Constraints. The review comment is on smakamali/FullTextScreener#5, at `Config/ConfigLitScrTemp.cfg:105`.

## Global Constraints

- **Reason text:** a result without a fingerprint is rejected with exactly `"it predates questionnaire fingerprints"`. The run already prints `The saved result for <pdf> is not reusable (<reason>); processing the paper again.`
- **A fingerprint that is present but empty** (`""` or `None`) counts as missing.
- **Order of checks:** "no answers", then a missing fingerprint, then a different questionnaire, then missing or failed answers. A result with no answers keeps its existing reason.
- **Unchanged:** behaviour for results that carry a fingerprint, and the result file format.
- **Where the work goes:** branch `fix/require-questionnaire-fingerprint` from up-to-date `main`, independent of PR #5. One commit, then a PR into `main`, not merged. Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Never commit** `Config/ConfigLitScr.cfg`.

## Review Focus

1. **A `.json` result written by 0.9.0 or earlier sits in `output/individual` during a resumed run.** Expected: the paper is redone, and the log names the reason. The test is in Task 1 (`test_resume_redoes_unfingerprinted_result`).
2. **A fingerprint field present but empty** (`"questionnaire_sha256": ""`). Expected: treated as missing. The test is in Task 1.
3. **`Convertor/merge_outputs.py` merging old results.** Expected: unaffected, because it doesn't call `checkCompleteResult`. Task 1, Step 4 checks this with a grep.

---

### Task 1: Reject results without a fingerprint

**Files:**
- Modify: `FullTextScreener.py`, `checkCompleteResult` (≈1511–1530): the docstring and one check.
- Modify: `readme.md`, the "Resuming" bullet (≈256) and the `skip_existing` comment in the config example (≈118).
- Test: `tests/test_offline.py`, `ResumeTests` (≈354–450), using its `complete(**extra)` helper and `writeIndividual`.

**Interfaces:**
- Consumes: none.
- Produces: `checkCompleteResult(data, questions, qHash) -> (bool, str)`, with the same signature. It now returns `(False, "it predates questionnaire fingerprints")` when `data.get("questionnaire_sha256")` is falsy.

- [ ] **Step 1: Write the failing tests.**
  - **Edit `test_check_complete_result`:** the line commented `# written before the fingerprint` must now assert `assertFalse(ok)`, and its reason must equal `"it predates questionnaire fingerprints"`.
  - **Add `test_empty_fingerprint_is_missing`:** `fts.checkCompleteResult(self.complete(questionnaire_sha256=""), self.QUESTIONS, self.hash)` returns `(False, "it predates questionnaire fingerprints")`.
  - **Add `test_resume_redoes_unfingerprinted_result`:**
    - set up `writeIndividual("a.pdf.json", self.complete())` with no fingerprint and `skip_existing = True`, following the existing test at ≈446 that writes `questionnaire_sha256="old"`;
    - run `literature_screening` through the class harness;
    - assert that the fake `process_single_paper` was called for `a.pdf`, and that stdout, captured with `contextlib.redirect_stdout(io.StringIO())`, contains `it predates questionnaire fingerprints`.
- [ ] **Step 2: Run the tests.** Run `C:/Users/smaka/anaconda3/envs/fulltextscreener/python.exe -m unittest tests.test_offline.ResumeTests -v`. Expected: the three tests above fail; the rest pass.
- [ ] **Step 3: Implement.** In `checkCompleteResult`, after the "no answers" check, add the missing-fingerprint check, so the existing different-questionnaire check only ever compares two real fingerprints. Rewrite the docstring sentence "Results written before the fingerprint was added are accepted if they are otherwise complete." to say that they are redone.
- [ ] **Step 4: Run all tests and check the other caller.** Run `python -m unittest discover -s tests`. Expected: OK, with 138 tests (136 + 2 new). Then run `grep -n checkCompleteResult Convertor/*.py`. Expected: no matches.
- [ ] **Step 5: Update the README.**
  - **Resuming bullet:** "Complete results made with the same questionnaire are reused; results without a questionnaire fingerprint (written before 0.9.1), partial, unreadable or outdated results are redone."
  - **Config-example comment:** keep "same questionnaire".
- [ ] **Step 6: Commit and open the PR.**
  - Run `git add FullTextScreener.py tests/test_offline.py readme.md`, then `git commit -m "fix: do not reuse results without a questionnaire fingerprint"`, with the trailer.
  - Push, and open a PR into `main`. Its description covers the bug (the review comment on PR #5), the new behaviour and its cost (pre-0.9.1 results are redone), and the test count.
