# GPU Embeddings and Parallel Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the full 168-paper re-extraction from about 50 hours to about 10–15 hours, first by embedding on the GPU, then by answering several papers at once.

**Architecture:** (1) The embedding model runs on the GPU in 8-bit. The precision it actually loaded with becomes part of the persistent-index paper key, so indexes built at one precision are never searched with query embeddings of another. (2) `literature_screening` gets a `workers` setting. With `workers > 1`, a thread pool processes papers concurrently in one process. All workers share one LLM, one embedding model (calls serialized by a lock) and one asyncio event loop. The loop runs in its own thread, and each worker submits its agent runs to it. With `workers = 1`, the run behaves exactly as today.

**Tech Stack:** Python 3, llama-index 0.14 (workflow `ReActAgent`), HuggingFace `gte-Qwen2-7B-instruct` with bitsandbytes 8-bit, Neo4j vector store, OpenRouter `mistralai/mistral-large-2512`, `concurrent.futures.ThreadPoolExecutor`, `unittest`.

**Spec:** no separate design document. The decisions were made with the user on 2026-11-26 and are recorded under Global Constraints. Measurements behind them, from the 7-paper test run `DataExtraction/v3test_7papers`:
- first-time indexing on the CPU takes 7–13 minutes per paper;
- answering the 25 questions takes 6–10 minutes per paper;
- 429 retries cost 4.5 minutes in total over 1.5 hours;
- the GPU was 8% busy.

## Global Constraints

- **Precision:** embeddings use `embed_device = cuda` and `embed_quantization = 8bit` in the live config's `[openrouter]` section. The template `Config/ConfigLitScrTemp.cfg` keeps the CPU default (the keys stay commented out) and gains a comment pointing to the new options.
- **Existing indexes stay valid:** a paper key computed for CPU fp32 must equal today's key, so existing indexes are still reused. Any other precision changes the key.
- **Default is unchanged:** `workers = 1` is the default, and with it the run behaves exactly as today, including the existing tests.
- **Worker limits:**
  - `workers` must be an integer from 1 to 4.
  - `workers > 1` requires `persistent_index = True`. With `persistent_index = False`, every paper wipes the shared database, so the run must refuse to start with a `PipelineConfigError`.
- **Sharing:** all workers share one LLM, one embedding model and one event loop. No second copy of the 7B embedding model may be loaded, because two copies do not fit on the 16 GB GPU.
- **Errors and stopping:**
  - Run-wide errors (`NonRetryableApiError`, `PipelineConfigError`) and Ctrl-C stop the whole run.
  - Papers in progress save `<pdf>.partial.json` at their next question boundary.
  - The combined output file is written however the run ends.
  - An unexpected error in one paper is recorded for that paper, and the other workers continue.
- **Result format:** result files, their keys and their format are unchanged. `merge_results.py` and the watcher must keep working.
- **Version:** the app version becomes 0.11.0 (`appVersion` in `Config/ConfigLitScrTemp.cfg`).
- **Where the work goes:** branch `feat/gpu-parallel` off `main`, one commit per task, and a PR into `main` at the end. Never commit `Config/ConfigLitScr.cfg`.

## Review Focus

1. **A CPU-built index searched with GPU query embeddings.** Old indexes have no precision in their key. Expected: a GPU run gets a new key and re-indexes. The test is in Task 1.
2. **Loading on the GPU fails silently** (bitsandbytes missing, out of memory), so the model falls back to the CPU. Expected: the key records `cpu-fp32`, not the requested `cuda-8bit`, and the run says so. The test is in Task 1.
3. **Ctrl-C with 3 workers mid-paper.** Expected:
   - no hang;
   - each running paper leaves a `.partial.json`;
   - the combined output is written;
   - the event-loop thread exits.

   The test is in Task 3.
4. **A 402 (credit exhausted) in one worker while the others are mid-paper.** Expected: the others stop at their next question with partial files, and the run raises `NonRetryableApiError`. The test is in Task 3.
5. **Two workers indexing at the same moment** while a third embeds queries. Expected: embedding calls are serialized and no exception occurs. The test is in Task 2, plus the live check in Task 4.

---

### Task 1: Embedding precision in the paper key, GPU config

**Files:**
- Modify: `FullTextScreener.py`:
  - `buildHuggingFaceEmbedding` (≈877–950);
  - `computePaperKey` (≈1314–1336);
  - its call in `process_single_paper` (≈1449).
- Modify: `Config/ConfigLitScrTemp.cfg` (comment and `appVersion = 0.11.0`), `readme.md` (one line under configuration).
- Test: `tests/test_offline.py`, class `PaperKeyTests`.

**Interfaces:**
- Produces:
  - `embeddingPrecision(embedModel) -> str`: returns `'cuda-8bit'`, `'cuda-4bit'`, `'cuda-fp16'`, `'cuda-fp32'` or `'cpu-fp32'`. It reads the attribute `_fts_precision`, which `buildHuggingFaceEmbedding` sets on the model it returns, set with `object.__setattr__` because the model is a pydantic object. If the attribute is missing, it returns `'cpu-fp32'`.
  - `computePaperKey(pdfPath, config, embedPrecision='cpu-fp32') -> str`.

- [ ] **Step 1: Write the failing tests** in `PaperKeyTests`:
  - `test_cpu_fp32_key_unchanged`: `computePaperKey(p, cfg) == computePaperKey(p, cfg, 'cpu-fp32')`, and both equal a key computed with today's code. Hard-code today's digest for the existing fixture PDF and config, captured by running the current function before editing it.
  - `test_precision_changes_key`: `computePaperKey(p, cfg, 'cuda-8bit') != computePaperKey(p, cfg, 'cpu-fp32')`.
  - `test_precision_attribute`: `embeddingPrecision(types.SimpleNamespace()) == 'cpu-fp32'`. For an object with `_fts_precision = 'cuda-8bit'` the result is `'cuda-8bit'`.
  - `test_fallback_records_cpu`: patch `torch.cuda.is_available` to return `False`, and `HuggingFaceEmbedding` to a stub class accepting `**kw`. Then `buildHuggingFaceEmbedding()` with `embedDevice='cuda'`, `embedQuantization='8bit'` returns a model whose `embeddingPrecision` is `'cpu-fp32'`.
- [ ] **Step 2: Run the tests.** Run `python -m unittest tests.test_offline.PaperKeyTests -v`. Expected: the 4 new tests fail (no `embeddingPrecision`, wrong `computePaperKey` signature).
- [ ] **Step 3: Implement.**
  - In `buildHuggingFaceEmbedding`, set `_fts_precision` on each return path, from what actually loaded.
  - In `computePaperKey`, add `"embed_precision"` to the `settings` dict **only when it is not `'cpu-fp32'`**, which keeps old keys valid.
  - In `process_single_paper`, pass `embeddingPrecision(screeningAgent.embedModel)`.
  - Print the precision with the "Indexed …" / "Reusing the index …" lines.
- [ ] **Step 4: Run all offline tests.** Run `python -m unittest discover -s tests`. Expected: all pass (92 + 4).
- [ ] **Step 5: Check GPU loading on this machine.** Set `embed_device = cuda` and `embed_quantization = 8bit` in the live `Config/ConfigLitScr.cfg` (not committed). Then run `python -c "import FullTextScreener as f; m=f.loadModels('./Config/ConfigLitScr.cfg'); print(f.embeddingPrecision(m[1]))"`. Expected: it prints `cuda-8bit`, and `nvidia-smi` shows about 8–9 GB used.
- [ ] **Step 6: Commit.** Run `git add FullTextScreener.py tests/test_offline.py Config/ConfigLitScrTemp.cfg readme.md`, then `git commit -m "feat: embedding precision in the paper key; GPU embedding documented"`.

### Task 2: Shared event-loop thread and embedding lock

**Files:**
- Modify: `FullTextScreener.py`:
  - new helpers after `freeModels` (≈697);
  - `ChatbotAgents.chatbot` (≈1018–1049).
- Test: `tests/test_offline.py`, class `EventLoopAndConfigTests`.

**Interfaces:**
- Produces:
  - `class LoopThread`: `__init__()` starts a daemon thread running a new asyncio loop. It also has `loop` (the `asyncio.AbstractEventLoop`) and `close(timeout=10)`, which stops the loop, joins the thread and closes the loop.
  - `runOnLoop(loop, coroFactory) -> Any`:
    - If `loop.is_running()` (another thread owns it), it submits with `asyncio.run_coroutine_threadsafe(coroFactory(), loop)` and returns `future.result()`. On `KeyboardInterrupt` or any `BaseException` it calls `future.cancel()` and re-raises.
    - Otherwise it uses `loop.run_until_complete(coroFactory())`, today's behaviour.
  - `serializeEmbedding(embedModel, lock) -> None`: wraps the instance's `_get_query_embedding`, `_get_text_embedding`, `_get_text_embeddings`, `_aget_query_embedding` and `_aget_text_embedding`, where present, so each call holds `lock` (a `threading.Lock`). It uses `object.__setattr__`. It is idempotent: a second call does not double-wrap (mark it with `_fts_serialized = True`).
- Consumes: none.

- [ ] **Step 1: Write the failing tests** in `EventLoopAndConfigTests`:
  - `test_run_on_running_loop_from_threads`: start a `LoopThread`. Four threads each call `runOnLoop(lt.loop, lambda: asyncio.sleep(0.05, result=i))`. All four results come back, and the total time is under 0.15 s, which shows they ran concurrently. `lt.close()` leaves `lt.loop.is_closed()` True.
  - `test_run_on_idle_loop_unchanged`: with a new idle loop, `runOnLoop(loop, lambda: asyncio.sleep(0, result=7)) == 7`.
  - `test_cancel_on_interrupt`: with a `LoopThread`, a coroutine that sleeps 5 s, and `future.result` patched to raise `KeyboardInterrupt`, the call raises `KeyboardInterrupt` and the coroutine is cancelled (check a flag set in its `except asyncio.CancelledError`).
  - `test_serialize_embedding`: a stub embed object whose `_get_text_embeddings` records max concurrency across 3 threads (sleep 0.02 inside). After `serializeEmbedding`, the max concurrency is 1. Calling it twice still gives a single wrapper.
  - `test_chatbot_uses_run_on_loop`: a `ChatbotAgents` built with `object.__new__` and patched `selectedAgent.run`, with `_loop` a running `LoopThread.loop`. Then `chatbot("q", None)` returns the stub's response, and memory rollback on exception still works (reuse the existing memory-rollback test's fixtures).
- [ ] **Step 2: Run the tests.** Run `python -m unittest tests.test_offline.EventLoopAndConfigTests -v`. Expected: the new tests fail.
- [ ] **Step 3: Implement** `LoopThread`, `runOnLoop` and `serializeEmbedding`. Change `chatbot()` to call `runOnLoop(self._loop, _run)` in place of `self._loop.run_until_complete(_run())`, keeping the memory snapshot and rollback.
- [ ] **Step 4: Run all offline tests.** Run `python -m unittest discover -s tests`. Expected: all pass.
- [ ] **Step 5: Commit.** Run `git commit -am "feat: event loop thread and serialized embedding for concurrent papers"`.

### Task 3: `workers` setting and parallel `literature_screening`

**Files:**
- Modify: `FullTextScreener.py`:
  - `_answerAllQuestions` (≈1529);
  - `process_single_paper` (signature, ≈1389);
  - `literature_screening` (≈1577–1675).
- Modify: `Config/ConfigLitScrTemp.cfg` (`[general]`, commented `workers = 1` with an explanation), `readme.md`, `FullTextScreener.md`.
- Test: `tests/test_offline.py`, class `ResumeTests`, reusing its `literature_screening` harness with patched `process_single_paper` / `loadModels` / `freeModels`, plus a new class `ParallelRunTests`.

**Interfaces:**
- Consumes: `LoopThread`, `runOnLoop`, `serializeEmbedding` (Task 2).
- Produces:
  - `class RunStopped(Exception)`: raised by `_answerAllQuestions` when the stop event is set.
  - `_answerAllQuestions(screeningAgent, questions, paper_answers, enableLogging, paper_id="", stopEvent=None)`: before each question, it raises `RunStopped` if `stopEvent` is set.
  - `process_single_paper(row, questions, configPath, input_pdf_folder, models=None, reindex=False, loop=None, stopEvent=None)`: it passes `stopEvent` on. `RunStopped` saves `.partial.json` like any interruption (the existing `BaseException` path) and is re-raised.
  - `readWorkers(config) -> int`: reads `[general] workers` (default 1). Values outside 1–4, or non-integers, raise `PipelineConfigError`, as does `workers > 1` together with `persistent_index = False`.
  - `literature_screening(...)`: same signature. With `workers == 1` it runs today's loop unchanged. With `workers > 1`:
    - It loads the models once and calls `serializeEmbedding(models[1], threading.Lock())`.
    - It creates one `LoopThread` and a `threading.Event` `stopEvent`.
    - It submits each paper not skipped by `skip_existing` to a `ThreadPoolExecutor(max_workers=workers)`, passing `loop=loopThread.loop, stopEvent=stopEvent`.
    - It collects results in the main thread with `as_completed`.
    - A paper's ordinary exception is recorded as today, and the run continues.
    - On `NonRetryableApiError`, `PipelineConfigError` or `KeyboardInterrupt` (including Ctrl-C while waiting), it sets `stopEvent`, cancels the futures not yet started, waits for the running ones (they stop at their next question), and then re-raises the first run-wide error, or `KeyboardInterrupt`. `RunStopped` from the waiting papers is not reported as an error.
    - The `finally` block writes the combined output, closes the `LoopThread` and frees the models.
    - It prints `Processing completed for paper k/N.` per paper, as today.

- [ ] **Step 1: Write the failing tests** in `ParallelRunTests` (config fixture with `workers = 3`, `persistent_index = True`):
  - `test_all_papers_processed_once`: 7 rows, fake `process_single_paper` sleeping 0.05 s. Every pdf_filename is called exactly once, the combined output holds 7 entries, and the total time is under 0.25 s.
  - `test_skip_existing_respected`: 2 of 5 rows have complete results on disk, so the fake is called for exactly 3.
  - `test_one_paper_error_does_not_stop_run`: the fake raises `ValueError` for one paper. The other papers complete, and that paper's entry has `answers.error` starting with `ValueError`.
  - `test_non_retryable_stops_all`:
    - The fake for paper 1 raises `NonRetryableApiError(402, "x")` after 0.05 s.
    - The fakes for the others loop on `stopEvent.wait(0.01)` and raise `RunStopped` once it is set.
    - Expected: `literature_screening` raises `NonRetryableApiError`, papers never started are never called, and the combined output file exists.
  - `test_keyboard_interrupt_stops_all`: like the last test, but `concurrent.futures.as_completed` is patched to raise `KeyboardInterrupt` on its first `next()`. Expected: `KeyboardInterrupt` is raised, every running fake has seen `stopEvent` set, and `LoopThread.close` was called.
  - `test_workers_requires_persistent`: `workers = 2` with `persistent_index = False`, so `readWorkers` raises `PipelineConfigError`.
  - `test_workers_bounds`: `workers = 0`, `5` or `two` raise `PipelineConfigError`; a missing key returns 1.
  - In `AnswerLoopTests`, `test_stop_event_raises_run_stopped`: with the event set before question 2, `RunStopped` is raised and `paper_answers` holds exactly Q1.
- [ ] **Step 2: Run the tests.** Run `python -m unittest tests.test_offline.ParallelRunTests tests.test_offline.AnswerLoopTests -v`. Expected: the new tests fail.
- [ ] **Step 3: Implement** `RunStopped`, the `stopEvent` parameter, `readWorkers` and the parallel branch of `literature_screening`. Keep the `workers == 1` path textually as it is today.
- [ ] **Step 4: Run all offline tests.** Run `python -m unittest discover -s tests`. Expected: all pass, including every existing `ResumeTests` case (sequential path unchanged).
- [ ] **Step 5: Commit.** Run `git commit -am "feat: workers setting: process several papers concurrently"`.

### Task 4: Watcher for several running papers, then live verification

**Files:**
- Modify: `../.claude/skills/run-extraction/scripts/watch_progress.py`, in the survey project, not the repo:
  - `state['current']` becomes the set of running papers, meaning papers whose "Processing paper" line has appeared and that have no result file yet;
  - the "Now:" block shows one bar per running paper;
  - the ETA stays based on clock time, which already accounts for parallelism.
- Modify: `../.claude/skills/run-extraction/SKILL.md` (document `workers`, GPU settings, recommended `workers = 3`).

- [ ] **Step 1: Check the watcher with a test log.**
  - Write a small fake log with two interleaved papers, A and B, in a temp run folder.
  - Run `python -X utf8 watch_progress.py <tmp_run> --interval 1` for 3 s.
  - Expected: both appear as RUNNING with their own question counts, and per-paper indexing times are correct.
- [ ] **Step 2: Run the live GPU check, with `workers = 1`.** Reuse `DataExtraction/v3test_7papers`. Move its `output/individual` aside to `output/individual_cpu_run`, set `skip_existing = True`, then run 3 papers that need indexing: 38, 94 and 114 (trim the metadata copy to them). Expected:
  - indexing takes about 1–2 minutes per paper (CPU: 7–9);
  - the log shows `cuda-8bit` and new paper keys;
  - all 75 answers parse.

  Compare their categorical answers with `individual_cpu_run`, and report the differences to the user. The agreement should be within the known 9% run-to-run variation.
- [ ] **Step 3: Run the live parallel check, with `workers = 3`.** Use the same 7 papers in a fresh output folder. Expected:
  - all 7 papers have complete results;
  - no `context leak` lines in the log;
  - 429s are counted, and the wall time is reported.

  Then calculate the hours projected for 168 papers, the dollar cost (unchanged per paper; the user records it), and recommend a `workers` value.
- [ ] **Step 4: Open the PR.** Push `feat/gpu-parallel` and open a PR into `main`, with a description covering the paper-key change (CPU keys unchanged), the `workers` setting and the live numbers.
