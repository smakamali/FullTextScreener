# Persistent per-paper index (design, 2026-11-26)

## Intent
Rerunning the questionnaire on papers that are already indexed (for example a one-question retest, or a resumed run) should not wipe Neo4j and re-index every PDF. Today indexing and setup take 5–9 of the 18–23 minutes per paper. The hard requirement: **no passage from one paper may ever reach the model while it answers questions about another paper.**

## How it works today
For each paper, `process_single_paper` builds a new `ChatbotAgents` (which also reloads the 7B embedding model), wipes the whole database (`clearFiles`: `MATCH (n) DETACH DELETE n` and drops all indexes), indexes the one PDF, and asks the questions. The retriever has no filter: wiping the database is the only isolation.

## Goals
- Index each paper once and keep it in Neo4j across runs.
- Restrict every search to the current paper, and verify it.
- Re-index a paper automatically when its PDF or the indexing settings change.
- Load the LLM and embedding model once per run, not once per paper.

## Non-goals
- No change to the questions, prompts, answer format, agent memory (still a new memory per paper) or output files.
- No pruning tool for stale entries (old fingerprints stay in the database; they are never retrieved). Can be added later.
- The chatbot/Gradio path keeps its current behaviour.

## Anti-goals
- Any silent cross-paper retrieval. A failure of isolation must stop the question with an error, never produce an answer.
- Reusing a half-built index after a crash.

## Approaches considered
1. **One shared index, filtered by paper key (chosen).** Every chunk carries a `paper_key`; each search filters on it; a guard checks every returned chunk. Simple to reason about and to test. Filtered queries in `Neo4jVectorStore` use exact cosine similarity over the paper's chunks instead of the approximate vector index, so retrieval may differ slightly from earlier runs (at least as accurate).
2. **A separate Neo4j vector index per paper.** Strongest isolation without a filter, keeps approximate search, but ~180 indexes and fragile index management in the library.
3. **Per-paper index files on disk, no Neo4j.** Physical isolation, but moves away from the existing Neo4j setup; more change than needed.

The user chose 1.

## Design
- **Paper key:** `sha256` of the PDF bytes plus the indexing settings (chunking method, chunk size, chunk overlap, embedding model, reader version), shortened to 16 hex characters. A renamed copy of the same PDF gets the same key; a replaced PDF (e.g. an OCR'd copy) or changed settings get a new one.
- **Stored on every chunk:** `paper_key` and `file_name` (already stored), plus the LLM-extracted citation as today.
- **Completion marker:** after a paper's chunks are all written, a node `(:IndexedPaper {paper_key, file_name, n_chunks, indexed_at})` is created. A paper counts as indexed only if the marker exists and the chunk count matches. Chunks with a key but no marker (crash mid-indexing) are deleted and the paper is re-indexed.
- **Skip indexing:** if the marker is valid, the paper is not read, chunked, embedded, or sent to the LLM for its citation.
- **Filtered retrieval:** the `VectorIndexRetriever` gets `MetadataFilters([paper_key == current key])`.
- **Guard (second line of defence):** a node postprocessor on the query engine checks every retrieved chunk's `paper_key`. Any mismatch, or a chunk without a key, raises `ContextLeakError`; the question is recorded with an `error` key and the paper saved as partial, like other failed questions. Zero retrieved chunks also counts as an error (it would mean the filter or index is broken).
- **No more per-paper wipe** in `process_single_paper`. Setting `[general] persistent_index = True` (default `True`); `False` restores today's wipe-per-paper behaviour.
- **Force re-index:** command-line flag `--reindex` re-indexes the run's papers even when their marker is valid.
- **Models loaded once:** the LLM and embedding model are created once per run and passed to each paper's agent; the agent, its tool and its memory are still new for each paper.
- **Tests (offline):** paper key changes with bytes and settings and not with the file name; the guard raises on a foreign or missing key and on an empty result; the marker check rejects partial indexes. **Live check:** index two papers, ask the same question for each, confirm every retrieved chunk belongs to the asked paper, then rerun and confirm indexing is skipped and answers still arrive.

## Left to the builder
- Exact names of helper functions and where the guard lives in the code.
- How the Cypher for the marker and chunk counts is written.
- The `reader version` constant (bump it whenever the PDF reader or chunking code changes).
