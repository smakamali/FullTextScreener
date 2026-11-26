# Persistent Per-Paper Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every indexed paper in Neo4j across runs, restrict each search to the paper being screened (verified by a guard), skip indexing for papers already indexed, and load the models once per run.

**Architecture:** Every chunk gets a `paper_key` (fingerprint of PDF bytes + indexing settings). A small registry class talks to Neo4j (completion markers, chunk counts, deleting half-built papers). The retriever filters on `paper_key`; a node postprocessor checks every retrieved chunk and records a violation that the answer loop turns into a failed question. `literature_screening` loads the LLM and embedding model once and passes them to each paper's `ChatbotAgents`.

**Tech Stack:** Python 3, llama-index 0.14.26 (`MetadataFilters`, `BaseNodePostprocessor`), `llama-index-vector-stores-neo4jvector` (filtered queries need Neo4j ≥ 5.18; the container runs 5.26), neo4j driver, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-26-persistent-index-design.md`

## Global Constraints
- All code changes in `FullTextScreener.py` (the codebase is one large module; follow it). Tests in `tests/test_offline.py`; they must stay offline (no API, Neo4j, GPU or models).
- Run tests from the repo root: `C:/Users/smaka/anaconda3/envs/fulltextscreener/python.exe -m unittest discover -s tests -v`. All 40 existing tests must keep passing.
- Paper key: `sha256` over the PDF bytes plus the settings, first **16 hex characters**.
- Settings in the key: `nodeParserType`, `chunk_size`, `chunk_overlap`, `bufferSize`, `breakpointPercentileThreshold` (from `[nodeparser]`), the embedding model name (`[<service>] embed_model`), and `READER_VERSION` (module constant, start at `"1"`).
- The file name is **not** part of the key.
- `paper_key` must not change what is embedded or sent to the LLM: add it to each document's `excluded_embed_metadata_keys` and `excluded_llm_metadata_keys`.
- Chunk node label is the store's default `Chunk`; completion marker label `IndexedPaper` with properties `paper_key`, `file_name`, `n_chunks`, `indexed_at` (ISO timestamp).
- New config key `[general] persistent_index`, default `True`; `False` keeps today's wipe-per-paper behaviour exactly.
- New CLI flag `--reindex`: re-index each paper of the run even if its marker is valid.
- No change to questions, prompts, answer format, output file layout, or the chatbot/Gradio path (`clearFiles`, `uploadFiles` stay as they are).
- Never print `Config/ConfigLitScr.cfg` unmasked and never commit it. Work on branch `feat/persistent-index`; commits are local; push and PR only when the user asks.
- Bump `appVersion` to `0.10.0` in `Config/ConfigLitScrTemp.cfg`.

## Review Focus
1. **Agent swallows the guard's exception:** the workflow `ReActAgent` turns a tool exception into an observation and the model may still answer. Expected: the question is recorded with an `error` key, not answered. (Test in Task 3: `test_violation_recorded_as_failed_question`.)
2. **Crash mid-indexing:** chunks written, no marker. Expected: next run deletes that key's chunks and re-indexes. (Task 2: `test_unmarked_chunks_are_not_indexed`; Task 5: `test_half_built_paper_is_rebuilt`.)
3. **Same PDF under two names in one run** (e.g. `V2TEST_165.pdf` and its real name): expected one index, reused, and answers for both. (Task 1: `test_key_ignores_file_name`; Task 5: `test_second_name_reuses_index`.)
4. **Replaced PDF with the same name** (OCR'd copy): expected a new key and fresh indexing; the old chunks are never retrieved. (Task 1: `test_key_changes_with_bytes`; Task 3 guard tests.)
5. **`persistent_index = False`:** expected identical behaviour to 0.9.1 (wipe, index, no filter required). (Task 5: `test_non_persistent_wipes_each_paper`.)

---

### Task 1: Paper key and document tagging

**Files:**
- Modify: `FullTextScreener.py` (new helpers near `questionnaireHash`, ~line 1104)
- Test: `tests/test_offline.py` (new class `PaperKeyTests`)

**Interfaces:**
- Produces: `READER_VERSION = "1"`; `computePaperKey(pdfPath: str, config: Config) -> str` (16 hex chars); `tagDocuments(documents: list, paperKey: str) -> list` (sets `metadata['paper_key']` and appends `'paper_key'` to both exclusion lists on each document, returns the same list).

- [ ] **Step 1: Write the failing tests**

```python
class PaperKeyTests(unittest.TestCase):
    # build a minimal fake config object with .nodeparser, .llm_model['service'], .dict[service]['embed_model']
    def test_key_is_16_hex(self): ...            # re.fullmatch(r"[0-9a-f]{16}", key)
    def test_key_ignores_file_name(self): ...    # same bytes in a.pdf and b.pdf -> equal keys
    def test_key_changes_with_bytes(self): ...   # one byte different -> different key
    def test_key_changes_with_settings(self): ...# chunk_size 1000 vs 800, embed_model A vs B, READER_VERSION patched -> each differs
    def test_tag_documents_excluded_from_embedding_and_llm(self):
        from llama_index.core import Document
        d = Document(text="t", metadata={"file_name": "a.pdf"})
        fts.tagDocuments([d], "abc")
        self.assertEqual(d.metadata["paper_key"], "abc")
        self.assertNotIn("abc", d.get_content(metadata_mode="embed"))
        self.assertNotIn("abc", d.get_content(metadata_mode="llm"))
        self.assertIn("a.pdf", d.get_content(metadata_mode="embed"))   # existing metadata unchanged
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: computePaperKey`).
- [ ] **Step 3: Implement** both helpers. Hash the bytes, then a JSON dump (`sort_keys=True`) of the settings dict.
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat: paper key and document tagging`.

### Task 2: Neo4j index registry

**Files:**
- Modify: `FullTextScreener.py` (new class `PaperIndexRegistry` after `Storage`)
- Test: `tests/test_offline.py` (class `RegistryTests`, with a fake driver whose `session().run(query, **params)` records calls and returns canned records)

**Interfaces:**
- Consumes: `Storage.url`, `.username`, `.password`.
- Produces: `PaperIndexRegistry(storage: Storage)` with
  - `countChunks(paperKey) -> int` (`MATCH (n:Chunk {paper_key:$k}) RETURN count(n)`)
  - `isIndexed(paperKey) -> bool`: marker exists **and** `countChunks == n_chunks` **and** `n_chunks > 0`
  - `deletePaper(paperKey) -> None`: deletes that key's `Chunk` nodes and its marker, nothing else
  - `markIndexed(paperKey, fileName) -> int`: counts chunks, `MERGE`s the marker with `n_chunks`, `file_name`, `indexed_at`; raises `ValueError` if the count is 0; returns the count.
  - Constructor accepts an optional `driver` argument (for tests); otherwise opens `GraphDatabase.driver`.

- [ ] **Step 1: Write the failing tests:** `test_marked_with_matching_count_is_indexed`, `test_unmarked_chunks_are_not_indexed`, `test_count_mismatch_is_not_indexed`, `test_delete_only_touches_that_key` (assert every query issued by `deletePaper` has `paper_key` bound to the given key and contains no unconditional `MATCH (n) DETACH DELETE n`), `test_mark_with_zero_chunks_raises`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `PaperIndexRegistry`.**
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat: Neo4j registry of indexed papers`.

### Task 3: Filtered retrieval and leak guard

**Files:**
- Modify: `FullTextScreener.py` — `QueryEngine._buildVectorQueryEngine`, `ChatbotAgents.updateQueryEngine`/`_buildAll`, `_answerAllQuestions`; new `ContextLeakError`, `PaperKeyGuard`
- Test: `tests/test_offline.py` (class `LeakGuardTests`; extend `FakeAgent` with a `leakGuard`)

**Interfaces:**
- Produces:
  - `class ContextLeakError(Exception)`
  - `class PaperKeyGuard(BaseNodePostprocessor)`: field `paperKey: str`; `_postprocess_nodes(nodes, query_bundle)` raises `ContextLeakError` and stores the message in `self.violation` (private attr) if `nodes` is empty or any `node.node.metadata.get('paper_key') != paperKey`; otherwise returns nodes unchanged. `consumeViolation() -> str | None` returns and clears it.
  - `QueryEngine._buildVectorQueryEngine(forceReindex=False, paperKey=None)`: when `paperKey` is set, the `VectorIndexRetriever` gets `filters=MetadataFilters(filters=[MetadataFilter(key="paper_key", value=paperKey, operator=FilterOperator.EQ)])` and the query engine gets `node_postprocessors=[guard]`; stores the guard as `self.leakGuard` (else `None`).
  - `ChatbotAgents.setPaper(paperKey: str) -> None`: rebuilds query engine, tool and agent for that key. `ChatbotAgents.leakGuard` property → `self.queryEngine.leakGuard`.
  - `_answerAllQuestions`: after each successful `safeApiCall`, if `getattr(screeningAgent, 'leakGuard', None)` has a violation, record `_errorAnswer(q_text, f"context leak: {msg}", raw)`, count it as failed, no re-ask.

- [ ] **Step 1: Write the failing tests:** `test_guard_passes_own_chunks`, `test_guard_raises_on_foreign_chunk`, `test_guard_raises_on_missing_key`, `test_guard_raises_on_empty_result`, `test_retriever_gets_paper_filter` (build `QueryEngine` with a mocked `Index`; assert the retriever's `_filters` hold one EQ filter on `paper_key`), `test_violation_recorded_as_failed_question` (FakeAgent whose guard has a violation set during `chatbot`; the answer has an `error` starting `context leak` and `failed == 1`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat: per-paper retrieval filter and leak guard`.

### Task 4: Load models once per run

**Files:**
- Modify: `FullTextScreener.py` — `ChatbotAgents.__init__`, `ChatbotAgents.cleanup`, new `loadModels`, `literature_screening`, `process_single_paper` signature
- Test: `tests/test_offline.py` (class `SharedModelTests`)

**Interfaces:**
- Produces:
  - `loadModels(configPath: str) -> tuple` returning `(llm, embedModel)` using today's service dispatch.
  - `ChatbotAgents(configPath, models=None)`: with `models`, uses them and sets `self.ownsModels = False` (no loader called); without, loads as today and `ownsModels = True`.
  - `cleanup()`: frees LLM/embedding only when `ownsModels`; still closes the loop, drops memory, agent and query engine; calls `clearFiles()` only when the paper is not persistent (Task 5 sets `self.persistent`; default `False` keeps today's behaviour).
  - `process_single_paper(row, questions, configPath, input_pdf_folder, models=None, reindex=False)`.
  - `literature_screening(..., reindex=False)` calls `loadModels` once before the loop and passes `models`; frees them in its `finally`.

- [ ] **Step 1: Write the failing tests:** `test_shared_models_not_reloaded` (patch the three `get*LlmAndEmbedding` methods and `updateQueryEngine`/`selectDefaultAgent`; constructing with `models` calls none of the loaders), `test_cleanup_keeps_shared_models` (after `cleanup()`, the passed model objects were not deleted or moved; check a sentinel attribute is intact).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** `Settings.llm` / `Settings.embed_model` still set in `__init__`.
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat: load the LLM and embedding model once per run`.

### Task 5: Persistent flow in `process_single_paper`, config and CLI

**Files:**
- Modify: `FullTextScreener.py` — `process_single_paper`, `Index.append` (new `paperKey` parameter: tag documents with `tagDocuments` before `_buildVector`), `parseArgs` (`--reindex`), `__main__` (pass `args.reindex`), `literature_screening` docstring
- Modify: `Config/ConfigLitScrTemp.cfg` (`persistent_index = True` with a comment; `appVersion = 0.10.0`)
- Test: `tests/test_offline.py` (class `PersistentFlowTests`, mocking `ChatbotAgents`, `PaperIndexRegistry`, `countPdfWords`, `_answerAllQuestions`)

**Interfaces:**
- Consumes: Tasks 1–4.
- Flow when `persistent_index` is true:
  1. PDF exists and has text (unchanged checks).
  2. `key = computePaperKey(pdf_path, config)`.
  3. If `reindex` or not `registry.isIndexed(key)`: `registry.deletePaper(key)`, upload the PDF, `appendIndex` with that key, `registry.markIndexed(key, pdf_filename)`; print `Indexed <file> (<n> chunks, key <key>)`. Else print `Reusing the index of <file> (key <key>)`.
  4. `screeningAgent.setPaper(key)`; answer questions; save as today. Store `paper_key` in the result next to `questionnaire_sha256`.
- When false: exactly today's code path (`clearFiles`, upload, `appendIndex`, no filter).

- [ ] **Step 1: Write the failing tests:** `test_indexed_paper_skips_indexing` (no upload/`appendIndex`, `setPaper(key)` called), `test_new_paper_is_indexed_and_marked`, `test_half_built_paper_is_rebuilt` (`isIndexed` False → `deletePaper` before indexing), `test_reindex_flag_forces_indexing`, `test_second_name_reuses_index` (two rows, same bytes, different names: indexing once), `test_non_persistent_wipes_each_paper` (`clearFiles` called, `setPaper` not called, registry unused), `test_result_records_paper_key`, `test_reindex_cli_flag` (`parseArgs(['--reindex']).reindex is True`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat: reuse indexed papers across runs (persistent_index)`.

### Task 6: Live check and docs

**Files:**
- Modify: `readme.md`, `FullTextScreener.md` (a short "Persistent index" section: what is stored, the key, `--reindex`, `persistent_index = False`, and that changing chunking or embedding settings re-indexes automatically)
- Run folder: `../DataExtraction/q2fix_test/` (the interrupted Q2-only test on papers 182, 165, 183; live config already points there)

- [ ] **Step 1: Wipe once:** the database still holds the half-indexed paper 183 from the interrupted run without a `paper_key`. Clear Neo4j once (`Storage.clear()` plus `Index.clear()`) before the first persistent run.
- [ ] **Step 2: First run** (background, command from the run-extraction skill). Expected in `run.log`: `Indexed V2TEST_182.pdf …`, `Indexed V2TEST_165.pdf …`, `Indexed V2TEST_183.pdf …`; three result files with `paper_key`; no `context leak` errors.
- [ ] **Step 3: Isolation check:** a one-off script (scratchpad) that, for each of the three keys, runs the filtered retriever for the Q2 text and prints the set of `paper_key` and `file_name` values returned. Expected: exactly one key per query, equal to the queried key. Also confirm in Neo4j: three `IndexedPaper` markers, every `Chunk` has a `paper_key`.
- [ ] **Step 4: Second run** with `skip_existing = False`. Expected: three `Reusing the index of …` lines, no `Generating embeddings` progress bars, answers for all three, and setup time per paper under 1 minute.
- [ ] **Step 5: Update the docs**, run all tests, expect PASS.
- [ ] **Step 6: Commit** `docs: persistent index`.
