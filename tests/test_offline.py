"""
Offline tests for FullTextScreener.py: no API calls, no Neo4j, no models are loaded.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import FullTextScreener as fts  # noqa: E402


class ParseAnswerTests(unittest.TestCase):
    def assertAnswer(self, parsed, short, evidence=None):
        self.assertIsInstance(parsed, dict)
        self.assertEqual(list(parsed), ["Evidence", "Reasoning", "ShortAnswer"])
        self.assertEqual(parsed["ShortAnswer"], short)
        if evidence is not None:
            self.assertEqual(parsed["Evidence"], evidence)

    def test_plain_json(self):
        text = json.dumps({"Evidence": "e", "Reasoning": "r", "ShortAnswer": "Yes"})
        self.assertAnswer(fts.parseAnswer(text), "Yes", "e")

    def test_escaped_newline_inside_string(self):
        # The old parser turned "\\n" into a real newline and then failed on it
        text = json.dumps({"ShortAnswer": "Yes", "Evidence": "line one\nline two"})
        self.assertAnswer(fts.parseAnswer(text), "Yes", "line one\nline two")

    def test_latex_backslash(self):
        text = json.dumps({"ShortAnswer": "No", "Evidence": "uses \\nabla f"})
        self.assertAnswer(fts.parseAnswer(text), "No", "uses \\nabla f")

    def test_raw_newline_inside_string(self):
        text = '{"Evidence": "first\nsecond", "Reasoning": "r", "ShortAnswer": "Yes"}'
        self.assertAnswer(fts.parseAnswer(text), "Yes", "first\nsecond")

    def test_text_before_and_after(self):
        text = 'Here is the answer: {"ShortAnswer": "Unsure", "Reasoning": "r", "Evidence": "e"} Hope this helps.'
        self.assertAnswer(fts.parseAnswer(text), "Unsure")

    def test_code_fence(self):
        text = '```json\n{"Evidence": "e", "Reasoning": "r", "ShortAnswer": ["TPC-H", "JOB"]}\n```'
        self.assertAnswer(fts.parseAnswer(text), ["TPC-H", "JOB"])

    def test_alias_keys(self):
        text = json.dumps({"answer": "Yes", "justification": "because", "evidence": "quote"})
        parsed = fts.parseAnswer(text)
        self.assertAnswer(parsed, "Yes", "quote")
        self.assertEqual(parsed["Reasoning"], "because")

    def test_nested_answer(self):
        text = json.dumps({"Q7": {"ShortAnswer": "AQE", "Reasoning": "r", "Evidence": "e"}})
        self.assertAnswer(fts.parseAnswer(text), "AQE")

    def test_tool_json_wrapped_in_agent_json(self):
        inner = {"answer": "Yes", "justification": "j", "evidence": "e"}
        text = json.dumps({"QuestionText": "q", "ShortAnswer": inner})
        self.assertAnswer(fts.parseAnswer(text), "Yes", "e")

    def test_structured_short_answer_becomes_string(self):
        text = json.dumps({"Evidence": "e", "Reasoning": "r",
                           "ShortAnswer": {"technique": "AQE", "description": "re-optimizes"}})
        self.assertAnswer(fts.parseAnswer(text), "technique: AQE; description: re-optimizes")

    def test_double_encoded_json_string(self):
        text = json.dumps(json.dumps({"ShortAnswer": "Yes", "Reasoning": "r", "Evidence": "e"}))
        self.assertAnswer(fts.parseAnswer(text), "Yes")

    def test_missing_keys_filled(self):
        parsed = fts.parseAnswer('{"ShortAnswer": "Yes"}')
        self.assertAnswer(parsed, "Yes", "")
        self.assertEqual(parsed["Reasoning"], "")

    def test_unparseable_returns_none(self):
        self.assertIsNone(fts.parseAnswer("I could not find this in the paper."))
        self.assertIsNone(fts.parseAnswer('{"note": "no answer key"}'))
        self.assertIsNone(fts.parseAnswer(None))

    def test_clean_and_convert_json_keeps_original_on_failure(self):
        self.assertEqual(fts.cleanAndConvertJson("not json"), "not json")
        self.assertEqual(fts.cleanAndConvertJson('{"ShortAnswer": "Yes"}')["ShortAnswer"], "Yes")


def _httpError(status):
    import httpx
    import openai
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    if status == 429:
        return openai.RateLimitError("rate limited", response=response, body=None)
    return openai.APIStatusError(f"status {status}", response=response, body=None)


class ErrorHandlingTests(unittest.TestCase):
    def test_http_status_found(self):
        self.assertEqual(fts.getHttpStatus(_httpError(429)), 429)
        self.assertEqual(fts.getHttpStatus(_httpError(402)), 402)
        wrapped = RuntimeError("workflow failed")
        wrapped.__cause__ = _httpError(401)
        self.assertEqual(fts.getHttpStatus(wrapped), 401)
        self.assertIsNone(fts.getHttpStatus(ValueError("no status here")))

    def test_step_limit_detected(self):
        self.assertTrue(fts.isAgentStepLimit(RuntimeError("Max iterations of 10 reached! Either something went wrong")))
        self.assertFalse(fts.isAgentStepLimit(RuntimeError("connection reset")))

    @mock.patch.object(fts.time, "sleep")
    def test_non_retryable_stops_at_once(self, sleep):
        call = mock.Mock(side_effect=_httpError(402))
        with self.assertRaises(fts.NonRetryableApiError):
            fts.safeApiCall(call, enableLogging=False)
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()

    @mock.patch.object(fts.time, "sleep")
    def test_step_limit_not_retried(self, sleep):
        call = mock.Mock(side_effect=RuntimeError("Max iterations of 10 reached!"))
        with self.assertRaises(fts.AgentStepLimitError):
            fts.safeApiCall(call, enableLogging=False)
        self.assertEqual(call.call_count, 1)

    @mock.patch.object(fts.time, "sleep")
    def test_transient_then_success(self, sleep):
        call = mock.Mock(side_effect=[_httpError(429), _httpError(503), "ok"])
        self.assertEqual(fts.safeApiCall(call, enableLogging=False), "ok")
        self.assertEqual(call.call_count, 3)

    @mock.patch.object(fts.time, "sleep")
    def test_transient_exhausted_reports_last_error(self, sleep):
        call = mock.Mock(side_effect=_httpError(429))
        with self.assertRaises(Exception) as ctx:
            fts.safeApiCall(call, enableLogging=False, maxAttempts=3)
        self.assertIn("after 3 attempts", str(ctx.exception))


class ContextWindowTests(unittest.TestCase):
    def setUp(self):
        fts._openRouterContextCache.clear()

    @mock.patch.object(fts.time, "sleep")
    def test_failed_lookup_raises_and_is_not_cached(self, sleep):
        with mock.patch("requests.get", side_effect=ConnectionError("down")):
            with self.assertRaises(fts.PipelineConfigError):
                fts.getOpenRouterContextWindow("mistralai/mistral-large-2512")
        self.assertNotIn("mistralai/mistral-large-2512", fts._openRouterContextCache)

    def test_unknown_model_raises(self):
        response = mock.Mock(**{"json.return_value": {"data": [{"id": "other/model", "context_length": 8000}]}})
        with mock.patch("requests.get", return_value=response):
            with self.assertRaises(fts.PipelineConfigError):
                fts.getOpenRouterContextWindow("mistral/mistral-large-latest")

    @mock.patch.object(fts.time, "sleep")
    def test_success_after_retry_is_cached(self, sleep):
        response = mock.Mock(**{"json.return_value": {"data": [{"id": "m/x", "context_length": 262144}]}})
        with mock.patch("requests.get", side_effect=[ConnectionError("blip"), response]) as get:
            self.assertEqual(fts.getOpenRouterContextWindow("m/x"), 262144)
            self.assertEqual(fts.getOpenRouterContextWindow("m/x"), 262144)
        self.assertEqual(get.call_count, 2)


class MemoryLimitTests(unittest.TestCase):
    def agent(self, contextWindow, limit=100000, maxNew=1024):
        agent = fts.ChatbotAgents.__new__(fts.ChatbotAgents)
        agent.memory_token_limit = limit
        agent.max_new_tokens = maxNew
        agent.llm = types.SimpleNamespace(metadata=types.SimpleNamespace(context_window=contextWindow))
        return agent

    def test_large_window_keeps_configured_limit(self):
        self.assertEqual(self.agent(262144).effectiveMemoryLimit(), 100000)

    def test_small_window_caps_limit(self):
        self.assertEqual(self.agent(32768).effectiveMemoryLimit(), 32768 - 1024 - fts.ChatbotAgents.PROMPT_MARGIN_TOKENS)

    def test_tiny_window_has_floor(self):
        self.assertEqual(self.agent(8000).effectiveMemoryLimit(), fts.ChatbotAgents.MIN_MEMORY_TOKENS)


class FakeAgent:
    """Stands in for ChatbotAgents: returns scripted replies (or raises scripted errors) in order."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.queries = []

    def chatbot(self, queryStr, history):
        self.queries.append(queryStr)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


GOOD = json.dumps({"Evidence": "e", "Reasoning": "r", "ShortAnswer": "Yes"})


@mock.patch.object(fts.time, "sleep")
class AnswerLoopTests(unittest.TestCase):
    def test_reask_after_unparseable_reply(self, sleep):
        agent = FakeAgent(["no json here", GOOD])
        answers = {}
        failed = fts._answerAllQuestions(agent, {"Q1": "Question one?"}, answers, enableLogging=False)
        self.assertEqual(failed, 0)
        self.assertEqual(answers["Q1"]["ShortAnswer"], "Yes")
        self.assertEqual(answers["Q1"]["QuestionText"], "Question one?")
        self.assertIn("could not be read", agent.queries[1])

    def test_failed_question_recorded_and_run_continues(self, sleep):
        agent = FakeAgent(["bad", "still bad", GOOD])
        answers = {}
        failed = fts._answerAllQuestions(agent, {"Q1": "one?", "Q2": "two?"}, answers, enableLogging=False)
        self.assertEqual(failed, 1)
        self.assertIn("error", answers["Q1"])
        self.assertEqual(answers["Q1"]["raw"], "still bad")
        self.assertEqual(answers["Q2"]["ShortAnswer"], "Yes")

    def test_step_limit_asked_once_more(self, sleep):
        agent = FakeAgent([RuntimeError("Max iterations of 10 reached!"), GOOD])
        answers = {}
        self.assertEqual(fts._answerAllQuestions(agent, {"Q1": "one?"}, answers, enableLogging=False), 0)
        self.assertEqual(len(agent.queries), 2)

    def test_non_retryable_propagates(self, sleep):
        agent = FakeAgent([_httpError(402)])
        with self.assertRaises(fts.NonRetryableApiError):
            fts._answerAllQuestions(agent, {"Q1": "one?"}, {}, enableLogging=False)


class ResumeTests(unittest.TestCase):
    QUESTIONS = {"Q1": "one?", "Q2": "two?"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.individual = os.path.join(root, "output", "individual")
        os.makedirs(self.individual)
        self.output = os.path.join(root, "output", "output.json")
        self.metadata = os.path.join(root, "metadata.csv")
        with open(self.metadata, "w", encoding="utf-8") as f:
            f.write("paper_id,pdf_filename,title,authors,year\n1,a.pdf,A,X,2020\n2,b.pdf,B,Y,2021\n")
        self.config = os.path.join(root, "test.cfg")
        with open(self.config, "w", encoding="utf-8") as f:
            f.write(f"[general]\nskip_existing = True\n[dir-structure]\noutputDir = {os.path.join(root, 'output')}\n")
        self.hash = fts.questionnaireHash(self.QUESTIONS)

    def tearDown(self):
        self.tmp.cleanup()

    def complete(self, **extra):
        answers = {q: {"QuestionText": t, "Evidence": "e", "Reasoning": "r", "ShortAnswer": "Yes"}
                   for q, t in self.QUESTIONS.items()}
        return {"metadata": {}, "answers": answers, **extra}

    def writeIndividual(self, name, content):
        with open(os.path.join(self.individual, name), "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def fakeProcess(self, row, questions, configPath, folder, models=None, reindex=False):
        self.processed.append(row["pdf_filename"])
        self.modelsSeen.append(models)
        return {row["pdf_filename"]: self.complete(questionnaire_sha256=self.hash)}

    def run_screening(self, side_effect=None):
        self.processed, self.modelsSeen = [], []
        self.models = ("llm", "embed")
        with mock.patch.object(fts, "process_single_paper", side_effect=side_effect or self.fakeProcess),              mock.patch.object(fts, "loadModels", return_value=self.models) as self.loadModels,              mock.patch.object(fts, "freeModels") as self.freeModels:
            fts.literature_screening(self.metadata, self.QUESTIONS, self.config, "pdfs", self.output)

    def test_models_loaded_once_and_freed(self):
        self.run_screening()
        self.assertEqual(self.processed, ["a.pdf", "b.pdf"])
        self.loadModels.assert_called_once_with(self.config)
        self.assertEqual(self.modelsSeen, [self.models, self.models])
        self.freeModels.assert_called_once_with(*self.models)

    def test_no_models_loaded_when_all_skipped(self):
        for name in ("a.pdf", "b.pdf"):
            self.writeIndividual(f"{name}.json", self.complete(questionnaire_sha256=self.hash))
        self.run_screening()
        self.assertEqual(self.processed, [])
        self.loadModels.assert_not_called()
        self.freeModels.assert_not_called()

    def test_check_complete_result(self):
        ok, _ = fts.checkCompleteResult(self.complete(questionnaire_sha256=self.hash), self.QUESTIONS, self.hash)
        self.assertTrue(ok)
        ok, _ = fts.checkCompleteResult(self.complete(), self.QUESTIONS, self.hash)  # written before the fingerprint
        self.assertTrue(ok)
        ok, why = fts.checkCompleteResult(self.complete(questionnaire_sha256="other"), self.QUESTIONS, self.hash)
        self.assertFalse(ok)
        self.assertIn("different questionnaire", why)
        broken = self.complete()
        broken["answers"]["Q2"] = "raw text"
        self.assertFalse(fts.checkCompleteResult(broken, self.QUESTIONS, self.hash)[0])
        missing = self.complete()
        del missing["answers"]["Q1"]
        self.assertFalse(fts.checkCompleteResult(missing, self.QUESTIONS, self.hash)[0])

    def test_reuses_complete_and_redoes_truncated(self):
        self.writeIndividual("a.pdf.json", self.complete(questionnaire_sha256=self.hash))
        self.writeIndividual("b.pdf.json", '{"metadata": {"truncated')
        self.run_screening()
        self.assertEqual(self.processed, ["b.pdf"])
        with open(self.output, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["a.pdf", "b.pdf"])

    def test_partial_only_is_redone(self):
        self.writeIndividual("a.pdf.partial.json", self.complete())
        self.run_screening()
        self.assertEqual(self.processed, ["a.pdf", "b.pdf"])

    def test_outdated_questionnaire_is_redone(self):
        self.writeIndividual("a.pdf.json", self.complete(questionnaire_sha256="old"))
        self.run_screening()
        self.assertIn("a.pdf", self.processed)

    def test_combined_output_written_when_run_stops(self):
        def stopOnSecond(row, *args, **kwargs):
            if row["pdf_filename"] == "b.pdf":
                raise fts.NonRetryableApiError(402, "out of credit")
            return self.fakeProcess(row, *args, **kwargs)
        with self.assertRaises(fts.NonRetryableApiError):
            self.run_screening(side_effect=stopOnSecond)
        with open(self.output, encoding="utf-8") as f:
            self.assertEqual(list(json.load(f)), ["a.pdf"])

    def test_unexpected_paper_error_does_not_stop_run(self):
        def failFirst(row, *args, **kwargs):
            if row["pdf_filename"] == "a.pdf":
                raise RuntimeError("unexpected")
            return self.fakeProcess(row, *args, **kwargs)
        self.run_screening(side_effect=failFirst)
        with open(self.output, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("error", data["a.pdf"]["answers"])
        self.assertNotIn("error", data["b.pdf"]["answers"])


class ConfigResolutionTests(unittest.TestCase):
    def test_explicit_missing_config_raises(self):
        with self.assertRaises(fts.PipelineConfigError):
            fts.resolveConfigPath("does/not/exist.cfg")

    def test_legacy_config_name_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "Config"))
            open(os.path.join(tmp, "Config", "ConfigLitScrInUse.cfg"), "w").close()
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("FTS_CONFIG", None)
                    self.assertEqual(fts.resolveConfigPath(None), fts.LEGACY_CONFIG_PATH)
            finally:
                os.chdir(cwd)

    def test_questions_fallback(self):
        config = types.SimpleNamespace(dir_structure={})
        path = fts.resolveQuestionsPath(config)
        self.assertTrue(path.endswith(os.path.join("Questionnaires", "questions_v1.json")))
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 25)


class AtomicWriteTests(unittest.TestCase):
    def test_no_temp_file_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            fts.writeJsonAtomic(path, {"a": 1})
            self.assertEqual(os.listdir(tmp), ["x.json"])
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"a": 1})


def fakeIndexConfig(chunk_size=1000, embed_model="Alibaba-NLP/gte-Qwen2-7B-instruct"):
    return types.SimpleNamespace(
        nodeparser={"nodeparsertype": "static", "chunk_size": chunk_size, "chunk_overlap": 50,
                    "buffersize": 1, "breakpointpercentilethreshold": 95},
        llm_model={"service": "openrouter"},
        dict={"openrouter": {"embed_model": embed_model}},
    )


class PaperKeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def pdf(self, name, data=b"%PDF-1.4 paper bytes"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_key_is_16_hex(self):
        key = fts.computePaperKey(self.pdf("a.pdf"), fakeIndexConfig())
        self.assertRegex(key, r"^[0-9a-f]{16}$")

    def test_key_ignores_file_name(self):
        cfg = fakeIndexConfig()
        self.assertEqual(fts.computePaperKey(self.pdf("a.pdf"), cfg), fts.computePaperKey(self.pdf("b.pdf"), cfg))

    def test_key_changes_with_bytes(self):
        cfg = fakeIndexConfig()
        self.assertNotEqual(fts.computePaperKey(self.pdf("a.pdf", b"one"), cfg),
                            fts.computePaperKey(self.pdf("b.pdf", b"onf"), cfg))

    def test_key_changes_with_settings(self):
        path = self.pdf("a.pdf")
        base = fts.computePaperKey(path, fakeIndexConfig())
        self.assertNotEqual(base, fts.computePaperKey(path, fakeIndexConfig(chunk_size=800)))
        self.assertNotEqual(base, fts.computePaperKey(path, fakeIndexConfig(embed_model="other/model")))
        with mock.patch.object(fts, "READER_VERSION", "2"):
            self.assertNotEqual(base, fts.computePaperKey(path, fakeIndexConfig()))

    def test_tag_documents_excluded_from_embedding_and_llm(self):
        from llama_index.core import Document
        d = Document(text="t", metadata={"file_name": "a.pdf"})
        fts.tagDocuments([d], "abc123")
        self.assertEqual(d.metadata["paper_key"], "abc123")
        self.assertNotIn("abc123", d.get_content(metadata_mode="embed"))
        self.assertNotIn("abc123", d.get_content(metadata_mode="llm"))
        self.assertIn("a.pdf", d.get_content(metadata_mode="embed"))


class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeNeo4j:
    """A tiny in-memory stand-in for the Neo4j driver: chunk counts and IndexedPaper markers per paper key."""
    def __init__(self, chunks=None, markers=None):
        self.chunks = dict(chunks or {})
        self.markers = dict(markers or {})
        self.queries = []

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def run(self, query, **params):
        self.queries.append((query, params))
        key = params.get("k")
        if query.startswith("CREATE INDEX"):
            return []
        if "DELETE" in query:
            if "PaperChunk" in query or ":Chunk" in query:
                self.chunks.pop(key, None)
            if ":IndexedPaper" in query:
                self.markers.pop(key, None)
            return []
        if "MERGE" in query:
            self.markers[key] = params["n"]
            return []
        if "count(" in query:
            return [FakeRecord(n=self.chunks.get(key, 0))]
        if ":IndexedPaper" in query:
            return [FakeRecord(n_chunks=self.markers[key])] if key in self.markers else []
        raise AssertionError(f"unexpected query {query}")


class RegistryTests(unittest.TestCase):
    def registry(self, **kw):
        self.db = FakeNeo4j(**kw)
        return fts.PaperIndexRegistry(storage=None, driver=self.db, nodeLabel="PaperChunk3584")

    def test_marked_with_matching_count_is_indexed(self):
        self.assertTrue(self.registry(chunks={"k1": 12}, markers={"k1": 12}).isIndexed("k1"))

    def test_unmarked_chunks_are_not_indexed(self):
        self.assertFalse(self.registry(chunks={"k1": 12}).isIndexed("k1"))

    def test_count_mismatch_is_not_indexed(self):
        self.assertFalse(self.registry(chunks={"k1": 7}, markers={"k1": 12}).isIndexed("k1"))
        self.assertFalse(self.registry(chunks={}, markers={"k1": 0}).isIndexed("k1"))

    def test_delete_only_touches_that_key(self):
        reg = self.registry(chunks={"k1": 3, "k2": 4}, markers={"k1": 3, "k2": 4})
        reg.deletePaper("k1")
        self.assertEqual(self.db.chunks, {"k2": 4})
        self.assertEqual(self.db.markers, {"k2": 4})
        for query, params in self.db.queries:
            if query.startswith("CREATE INDEX"):
                continue
            self.assertEqual(params.get("k"), "k1")
            self.assertIn("$k", query)

    def test_mark_with_zero_chunks_raises(self):
        reg = self.registry()
        with self.assertRaises(ValueError):
            reg.markIndexed("k1", "a.pdf")
        self.assertEqual(self.db.markers, {})

    def test_mark_records_count(self):
        reg = self.registry(chunks={"k1": 5})
        self.assertEqual(reg.markIndexed("k1", "a.pdf"), 5)
        self.assertTrue(reg.isIndexed("k1"))


def chunkNode(text, paperKey):
    from llama_index.core.schema import TextNode, NodeWithScore
    meta = {"file_name": f"{paperKey}.pdf"}
    if paperKey is not None:
        meta["paper_key"] = paperKey
    return NodeWithScore(node=TextNode(text=text, metadata=meta), score=1.0)


class LeakGuardTests(unittest.TestCase):
    def test_guard_passes_own_chunks(self):
        guard = fts.PaperKeyGuard(paperKey="k1")
        nodes = [chunkNode("a", "k1"), chunkNode("b", "k1")]
        self.assertEqual(guard.postprocess_nodes(nodes), nodes)
        self.assertIsNone(guard.consumeViolation())

    def test_guard_raises_on_foreign_chunk(self):
        guard = fts.PaperKeyGuard(paperKey="k1")
        with self.assertRaises(fts.ContextLeakError):
            guard.postprocess_nodes([chunkNode("a", "k1"), chunkNode("b", "k2")])
        self.assertIn("k2", guard.consumeViolation())
        self.assertIsNone(guard.consumeViolation())   # consumed

    def test_guard_raises_on_missing_key(self):
        guard = fts.PaperKeyGuard(paperKey="k1")
        with self.assertRaises(fts.ContextLeakError):
            guard.postprocess_nodes([chunkNode("a", None)])
        self.assertIsNotNone(guard.consumeViolation())

    def test_guard_raises_on_empty_result(self):
        guard = fts.PaperKeyGuard(paperKey="k1")
        with self.assertRaises(fts.ContextLeakError):
            guard.postprocess_nodes([])
        self.assertIsNotNone(guard.consumeViolation())

    def buildEngine(self, paperKey):
        from llama_index.core import VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.llms import MockLLM
        nodes = [chunkNode(f"paper one passage {i}", "k1").node for i in range(4)]
        nodes += [chunkNode(f"paper two passage {i}", "k2").node for i in range(4)]
        index = VectorStoreIndex(nodes=nodes, embed_model=MockEmbedding(embed_dim=8))
        engine = object.__new__(fts.QueryEngine)
        engine.index = types.SimpleNamespace(vectorIndex=index)
        engine.vectorTopK = 6
        with mock.patch.object(fts.Settings, "_llm", MockLLM()):
            engine._buildVectorQueryEngine(paperKey=paperKey)
        return engine

    def test_retriever_gets_paper_filter(self):
        engine = self.buildEngine("k1")
        filters = engine.vectorRetriever._filters.filters
        self.assertEqual([(f.key, f.value, f.operator) for f in filters],
                         [("paper_key", "k1", fts.FilterOperator.EQ)])
        self.assertIsInstance(engine.leakGuard, fts.PaperKeyGuard)

    def test_retrieval_returns_only_that_paper(self):
        for key in ("k1", "k2"):
            engine = self.buildEngine(key)
            found = engine.vectorRetriever.retrieve("passage")
            self.assertEqual(len(found), 4)
            self.assertEqual({n.node.metadata["paper_key"] for n in found}, {key})

    def test_no_paper_key_means_no_filter_or_guard(self):
        engine = self.buildEngine(None)
        self.assertIsNone(engine.vectorRetriever._filters)
        self.assertIsNone(engine.leakGuard)


class GuardedFakeAgent(FakeAgent):
    """A FakeAgent whose retrieval guard reports a violation during the listed calls (0-based)."""
    def __init__(self, replies, leakyCalls):
        super().__init__(replies)
        self.leakGuard = fts.PaperKeyGuard(paperKey="k1")
        self.leakyCalls = set(leakyCalls)

    def chatbot(self, queryStr, history):
        if len(self.queries) in self.leakyCalls:
            try:
                self.leakGuard.postprocess_nodes([chunkNode("x", "k2")])
            except fts.ContextLeakError:
                pass   # the agent library swallows tool errors and carries on
        return super().chatbot(queryStr, history)


@mock.patch.object(fts.time, "sleep")
class LeakAnswerLoopTests(unittest.TestCase):
    def test_violation_recorded_as_failed_question(self, sleep):
        agent = GuardedFakeAgent([GOOD, GOOD], leakyCalls=[0])
        answers = {}
        failed = fts._answerAllQuestions(agent, {"Q1": "q1?", "Q2": "q2?"}, answers, False)
        self.assertEqual(failed, 1)
        self.assertTrue(answers["Q1"]["error"].startswith("context leak"))
        self.assertEqual(answers["Q1"]["ShortAnswer"], "")
        self.assertEqual(answers["Q2"]["ShortAnswer"], "Yes")
        self.assertEqual(len(agent.queries), 2)   # not asked again


TEMPLATE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Config", "ConfigLitScrTemp.cfg")


class SentinelModel:
    """Stands in for a loaded model; records whether cleanup tried to free it."""
    def __init__(self):
        self.metadata = types.SimpleNamespace(context_window=262144)
        self._model = types.SimpleNamespace(cpu=lambda: setattr(self, "movedToCpu", True))
        self.movedToCpu = False


@mock.patch.object(fts, "Settings")
@mock.patch.object(fts.ChatbotAgents, "selectDefaultAgent")
@mock.patch.object(fts.ChatbotAgents, "updateQueryEngine")
@mock.patch.object(fts.ChatbotAgents, "getOpenRouterLlmAndEmbedding")
@mock.patch.object(fts.ChatbotAgents, "getMistralLlmAndEmbedding")
@mock.patch.object(fts.ChatbotAgents, "getHuggingFaceLlmAndEmbedding")
class SharedModelTests(unittest.TestCase):
    def test_shared_models_not_reloaded(self, hf, mistral, openrouter, *_):
        llm, embed = SentinelModel(), SentinelModel()
        agent = fts.ChatbotAgents(configPath=TEMPLATE_CONFIG, models=(llm, embed))
        for loader in (hf, mistral, openrouter):
            loader.assert_not_called()
        self.assertIs(agent.llm, llm)
        self.assertIs(agent.embedModel, embed)
        self.assertFalse(agent.ownsModels)

    def test_without_models_loads_as_before(self, hf, mistral, openrouter, *_):
        def load(agentSelf):
            agentSelf.llm, agentSelf.embedModel = SentinelModel(), SentinelModel()
        hf.side_effect = lambda: load(agentRef[0])
        agentRef = [None]
        original = fts.ChatbotAgents._loadModels
        def capture(agentSelf):
            agentRef[0] = agentSelf
            return original(agentSelf)
        with mock.patch.object(fts.ChatbotAgents, "_loadModels", capture):
            agent = fts.ChatbotAgents(configPath=TEMPLATE_CONFIG)
        hf.assert_called_once()   # the template's service is huggingface
        self.assertTrue(agent.ownsModels)

    def test_cleanup_keeps_shared_models(self, *_):
        llm, embed = SentinelModel(), SentinelModel()
        agent = fts.ChatbotAgents(configPath=TEMPLATE_CONFIG, models=(llm, embed))
        with mock.patch.object(fts.ChatbotAgents, "clearFiles"):
            agent.cleanup()
        self.assertTrue(hasattr(llm, "_model"))
        self.assertTrue(hasattr(embed, "_model"))
        self.assertFalse(llm.movedToCpu or embed.movedToCpu)


class FakeRegistry:
    """Stands in for PaperIndexRegistry; the state is shared by all instances of one test."""
    indexed, calls = set(), []

    def __init__(self, storage):
        pass

    def isIndexed(self, key):
        FakeRegistry.calls.append(("isIndexed", key))
        return key in FakeRegistry.indexed

    def deletePaper(self, key):
        FakeRegistry.calls.append(("deletePaper", key))
        FakeRegistry.indexed.discard(key)

    def markIndexed(self, key, fileName):
        FakeRegistry.calls.append(("markIndexed", key))
        FakeRegistry.indexed.add(key)
        return 5

    def close(self):
        pass


class PersistentFlowTests(unittest.TestCase):
    QUESTIONS = {"Q1": "one?"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.pdfs = os.path.join(root, "pdfs")
        os.makedirs(self.pdfs)
        for name, data in (("a.pdf", b"paper A"), ("a_copy.pdf", b"paper A"), ("b.pdf", b"paper B")):
            with open(os.path.join(self.pdfs, name), "wb") as f:
                f.write(data)
        self.outputDir = os.path.join(root, "output")
        FakeRegistry.indexed, FakeRegistry.calls = set(), []
        patches = [
            mock.patch.object(fts, "ChatbotAgents"),
            mock.patch.object(fts, "PaperIndexRegistry", FakeRegistry),
            mock.patch.object(fts, "countPdfWords", return_value=5000),
            mock.patch.object(fts, "_answerAllQuestions", side_effect=self.fakeAnswers),
        ]
        self.agentClass = patches[0].start()
        for p in patches[1:]:
            p.start()
        for p in patches:
            self.addCleanup(p.stop)
        self.agent = self.agentClass.return_value

    def fakeAnswers(self, agent, questions, answers, enableLogging, paperId=""):
        for q, t in questions.items():
            answers[q] = {"QuestionText": t, "Evidence": "e", "Reasoning": "r", "ShortAnswer": "Yes"}
        return 0

    def config(self, persistent=True):
        path = os.path.join(self.tmp.name, f"test_{persistent}.cfg")
        lines = ["[general]", f"persistent_index = {persistent}",
                 "[dir-structure]", f"outputDir = {self.outputDir}",
                 "[nodeparser]", "nodeParserType = static", "chunk_size = 1000", "chunk_overlap = 50",
                 "bufferSize = 1", "breakpointPercentileThreshold = 95",
                 "[llm-model]", "service = openrouter", "[openrouter]", "embed_model = some/model", ""]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def run_paper(self, name, persistent=True, reindex=False):
        return fts.process_single_paper({"paper_id": "1", "pdf_filename": name}, self.QUESTIONS,
                                        self.config(persistent), self.pdfs, models=("llm", "embed"), reindex=reindex)

    def key(self, name):
        return fts.computePaperKey(os.path.join(self.pdfs, name), fts.Config(self.config()))

    def test_new_paper_is_indexed_and_marked(self):
        self.run_paper("a.pdf")
        key = self.key("a.pdf")
        self.assertEqual(FakeRegistry.calls, [("isIndexed", key), ("deletePaper", key), ("markIndexed", key)])
        self.agent.appendIndex.assert_called_once()
        self.assertEqual(self.agent.appendIndex.call_args.kwargs.get("paperKey"), key)
        self.agent.setPaper.assert_called_once_with(key)
        self.agent.clearFiles.assert_not_called()
        # C1: the PDF itself is read, never whatever happens to be in inputDir
        self.agent.uploadFiles.assert_not_called()
        self.assertEqual(self.agent.appendIndex.call_args.kwargs.get("inputFiles"), [os.path.join(self.pdfs, "a.pdf")])
        self.agentClass.assert_called_once_with(configPath=mock.ANY, models=("llm", "embed"))

    def test_indexed_paper_skips_indexing(self):
        FakeRegistry.indexed.add(self.key("a.pdf"))
        self.run_paper("a.pdf")
        self.agent.uploadFiles.assert_not_called()
        self.agent.appendIndex.assert_not_called()
        self.agent.setPaper.assert_called_once_with(self.key("a.pdf"))

    def test_half_built_paper_is_rebuilt(self):
        # isIndexed is False for a paper whose marker is missing or whose chunk count is off
        self.run_paper("a.pdf")
        names = [c[0] for c in FakeRegistry.calls]
        self.assertLess(names.index("deletePaper"), names.index("markIndexed"))
        self.agent.appendIndex.assert_called_once()

    def test_reindex_flag_forces_indexing(self):
        FakeRegistry.indexed.add(self.key("a.pdf"))
        self.run_paper("a.pdf", reindex=True)
        self.agent.appendIndex.assert_called_once()
        self.assertIn(("deletePaper", self.key("a.pdf")), FakeRegistry.calls)

    def test_second_name_reuses_index(self):
        self.run_paper("a.pdf")
        self.run_paper("a_copy.pdf")
        self.assertEqual(self.agent.appendIndex.call_count, 1)
        self.assertEqual(self.agent.setPaper.call_args_list, [mock.call(self.key("a.pdf"))] * 2)

    def test_different_papers_get_different_keys(self):
        self.run_paper("a.pdf")
        self.run_paper("b.pdf")
        self.assertEqual(self.agent.appendIndex.call_count, 2)
        keys = [c.args[0] for c in self.agent.setPaper.call_args_list]
        self.assertEqual(len(set(keys)), 2)

    def test_non_persistent_wipes_each_paper(self):
        with mock.patch.object(FakeRegistry, "__init__", side_effect=AssertionError("registry used")):
            self.run_paper("a.pdf", persistent=False)
        self.agent.clearFiles.assert_called_once()
        self.agent.appendIndex.assert_called_once()
        self.assertIsNone(self.agent.appendIndex.call_args.kwargs.get("paperKey"))
        self.agent.setPaper.assert_not_called()
        self.assertFalse(self.agent.persistent)

    def test_result_records_paper_key(self):
        result = self.run_paper("a.pdf")
        self.assertEqual(result["a.pdf"]["paper_key"], self.key("a.pdf"))
        with open(os.path.join(self.outputDir, "individual", "a.pdf.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["paper_key"], self.key("a.pdf"))

    def test_failed_marking_is_an_indexing_error(self):
        with mock.patch.object(FakeRegistry, "markIndexed", side_effect=ValueError("No chunks were stored")):
            result = self.run_paper("a.pdf")
        self.assertIn("Error indexing PDF", result["a.pdf"]["answers"]["error"])
        self.agent.setPaper.assert_not_called()

    def test_reindex_cli_flag(self):
        self.assertTrue(fts.parseArgs(["--reindex"]).reindex)
        self.assertFalse(fts.parseArgs([]).reindex)


class ReviewFixTests(unittest.TestCase):
    """Fixes from the whole-branch review (C1, I1-I4)."""

    def test_tag_documents_rejects_other_file(self):
        from llama_index.core import Document
        docs = [Document(text="t", metadata={"file_name": "paper_a.pdf"}),
                Document(text="u", metadata={"file_name": "leftover.pdf"})]
        with self.assertRaises(fts.ContextLeakError):
            fts.tagDocuments(docs, "k1", expectedFile="paper_a.pdf")

    def test_append_reads_only_the_given_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.txt", "b.txt"):
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                    f.write(f"text of {name}")
            index = object.__new__(fts.Index)
            index.llm = None
            captured = []
            index._buildVector = lambda docs, batchSize=1: captured.extend(docs)
            index.append(tmp, paperKey="k1", inputFiles=[os.path.join(tmp, "a.txt")])
        self.assertTrue(captured)
        self.assertEqual({d.metadata["file_name"] for d in captured}, {"a.txt"})
        self.assertEqual({d.metadata["paper_key"] for d in captured}, {"k1"})

    @mock.patch.object(fts, "getOpenRouterContextWindow", return_value=262144)
    @mock.patch.object(fts.ChatbotAgents, "buildHuggingFaceEmbedding", return_value="embed")
    def test_openrouter_client_not_reused(self, *_):
        loader = object.__new__(fts.ChatbotAgents)
        loader.contextWindow, loader.apiKey, loader.llmModel = 0, "test-key", "mistralai/mistral-large-2512"
        loader.temperature, loader.max_new_tokens = 0.0, 1024
        loader.getOpenRouterLlmAndEmbedding()
        self.assertFalse(loader.llm.reuse_client)

    def storageConfig(self, persistent):
        return types.SimpleNamespace(
            neo4j={"url1": "bolt://x", "username": "u", "password": "p"},
            llm_model={"service": "openrouter"}, dict={"openrouter": {"embeddim": 3584}},
            general={"persistent_index": persistent})

    def test_persistent_store_uses_own_label_and_index(self):
        with mock.patch.object(fts, "Neo4jVectorStore") as store, mock.patch.object(fts, "StorageContext"):
            store.return_value.embedding_dimension = 3584
            fts.Storage(self.storageConfig(True))
        kwargs = store.call_args.kwargs
        self.assertEqual(kwargs["node_label"], "PaperChunk3584")
        self.assertEqual(kwargs["index_name"], "paper_chunk_3584")

    def test_non_persistent_store_keeps_defaults(self):
        with mock.patch.object(fts, "Neo4jVectorStore") as store, mock.patch.object(fts, "StorageContext"):
            store.return_value.embedding_dimension = 3584
            fts.Storage(self.storageConfig(False))
        self.assertNotIn("node_label", store.call_args.kwargs)
        self.assertNotIn("index_name", store.call_args.kwargs)

    def test_dimension_mismatch_stops(self):
        with mock.patch.object(fts, "Neo4jVectorStore") as store, mock.patch.object(fts, "StorageContext"):
            store.return_value.embedding_dimension = 1536
            with self.assertRaises(fts.PipelineConfigError):
                fts.Storage(self.storageConfig(True))

    def test_registry_creates_key_index(self):
        db = FakeNeo4j()
        fts.PaperIndexRegistry(storage=None, driver=db, nodeLabel="PaperChunk3584")
        created = [q for q, _ in db.queries if q.startswith("CREATE INDEX")]
        self.assertTrue(any("PaperChunk3584" in q and "paper_key" in q and "IF NOT EXISTS" in q for q in created))

    def test_registry_uses_store_label(self):
        db = FakeNeo4j(chunks={"k1": 2})
        storage = types.SimpleNamespace(vectorStore=types.SimpleNamespace(node_label="PaperChunk3584"))
        reg = fts.PaperIndexRegistry(storage=storage, driver=db)
        reg.countChunks("k1")
        self.assertIn("(n:`PaperChunk3584`", db.queries[-1][0])


if __name__ == "__main__":
    unittest.main()
