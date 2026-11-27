#################################### appVersion = 0.9.1 ######################################
# TODO: harmonize config management
# TODO: make sure the questions are articulated independently
################################ Import Required Dependencies ################################

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.environ["HF_METRICS_CACHE"] = os.path.join(os.environ["HF_HOME"], "metrics")
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.expanduser("~/.cache/llama_index")

import sys
import gc
import shutil
import logging
import time
import random
import re
import json
import csv
import hashlib
import argparse
from typing import Optional
import datetime
import pandas as pd
from configparser import ConfigParser
from tqdm import tqdm
from neo4j import GraphDatabase
import torch

import huggingface_hub
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.utils import iter_batch
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import PrivateAttr
import asyncio
import threading
import concurrent.futures

# Append the root directory to path so that the modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Modules.Readers import CustomCSVReader, CustomPDFReader
from Modules.Tools import touch, deleteFolderContents, ensureFolderExists

from llama_index.core import set_global_handler

# disable token counting globally
set_global_handler("simple")

################################ System Message ################################
systemMessage = """
You are a literature screening assistant. Your role is to analyze academic papers and provide structured answers to a predefined set of questions. Your answers must be based solely on the information in the provided document and must conform to the expected formats and standards below.

You always follow these rules:
    Rule 1: Your main goal is to answer each question as accurately and concisely as possible, using only the context provided.
    Rule 2: NEVER answer from general knowledge or assumptions. If the answer is not supported by the document, return "Unsure".
    Rule 3: Your final answer must be a single JSON object. Do NOT include any surrounding explanation or formatting.
    Rule 4: The JSON object has exactly these keys, in this order, so that you gather the evidence and reason about it before committing to an answer:
        - "Evidence": the exact sentence(s) from the paper that support the answer, or "Not Applicable" if there are none.
        - "Reasoning": a short explanation of how the evidence leads to the answer.
        - "ShortAnswer": the short-form answer (e.g., Yes, No, Unsure, one of the listed options, or a concise phrase or list).
    Rule 5: Use the query engine tool to retrieve the relevant passages for every question, even if an earlier question covered a similar topic. You may use your earlier answers about this paper to stay consistent, but every answer must rest on evidence retrieved from the paper.
    Rule 6: 
    - If the question expects a categorical answer (e.g., "Yes", "No", "Unsure"), choose the best matching label.
    - If the question asks for a description, list, or procedure, write a short, meaningful answer based directly on document content. Avoid repeating placeholder phrases like “a description of the process”.
    Rule 7: For questions that ask for lists (e.g., evaluation measures or architectures), extract only items explicitly stated in the document.
    Rule 8: If a question refers to scope (e.g., "to which scope does it apply?"), infer only from direct or strongly implied evidence.
    Rule 9: If rate limits or tool errors occur, retry until the result is retrieved.
    Rule 10: Never hallucinate or overgeneralize. Stay grounded in the document at all times.
"""

################################ Configurations ################################
class Config:
    """
    Loads configuration parameters from a config file.
    """
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        self.dict = {}
        for section in self.config.sections():
            section_data = self._load_section(section)
            setattr(self, section.replace('-', '_'), section_data)
            self.dict[section.replace('-', '_')] = section_data

    def _load_section(self, section):
        section_data = {}
        if section in self.config:
            for key, value in self.config.items(section):
                section_data[key] = self._convert_value(value)
        return section_data

    @staticmethod
    def _convert_value(value):
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        if ',' in value:
            return [item.strip() for item in value.split(',') if item.strip()]
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

################################ Helper Functions and classes ################################
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)
    
def setChunkingMethod(config):
    # Configure how PDFs are chunked (either "semantic" or "static")
    nodeParserType = config.nodeparser['nodeparsertype']
    bufferSize = config.nodeparser['buffersize']
    breakpointPercentileThreshold = config.nodeparser['breakpointpercentilethreshold']
    Settings.chunk_size  = config.nodeparser['chunk_size']
    Settings.chunk_overlap = config.nodeparser['chunk_overlap']

    if nodeParserType == 'semantic':
        from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
        Settings.transformations = [
            SemanticSplitterNodeParser.from_defaults(
                buffer_size=bufferSize,
                breakpoint_percentile_threshold=breakpointPercentileThreshold,
                embed_model=Settings.embed_model,
            ),
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
        ]
    elif nodeParserType == 'static':
        from llama_index.core.node_parser import SentenceSplitter
        Settings.transformations = [
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
        ]
    else:
        raise Exception("`nodeParserType` must be either `static` or `semantic`")


def setLogging(config):
    enableLogging = config.general['enablelogging']
    logDir = config.dir_structure['logdir']
    logFile = config.dir_structure['logfile']
    ensureFolderExists(logDir)
    if enableLogging:
        logPath = os.path.join(logDir, logFile)
        touch(logPath)
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(logPath),
                                logging.StreamHandler()
                            ])
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return logging.getLogger(__name__)

class NonRetryableApiError(Exception):
    """An API error that retrying cannot fix (e.g., out of credit, invalid key, malformed request)."""
    def __init__(self, statusCode, message):
        super().__init__(f"HTTP {statusCode}: {message}")
        self.statusCode = statusCode

# HTTP status codes that will not succeed on retry:
# 400 bad request (e.g., prompt too long), 401 invalid key, 402 out of credit / spending limit,
# 403 forbidden, 404 unknown model. 408, 429 and 5xx are transient and are retried.
NON_RETRYABLE_STATUS = {400, 401, 402, 403, 404}

class PipelineConfigError(Exception):
    """A setup problem that affects every paper (bad config, unknown model); the run stops at once."""

class RunStopped(Exception):
    """The run is stopping (another paper hit a run-wide error, or Ctrl-C): this paper ends at its next question."""

MAX_WORKERS = 4

def readWorkers(config):
    """`workers` in [general]: how many papers are processed at once (1-4, default 1). Several workers need
    persistent_index = True (without it, each paper wipes the database the other papers are searching)."""
    workers = config.general.get('workers', 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise PipelineConfigError(f"workers in [general] must be a whole number from 1 to {MAX_WORKERS}, not {workers!r}.")
    if workers > 1 and not bool(config.general.get('persistent_index', True)):
        raise PipelineConfigError(
            f"workers = {workers} needs persistent_index = True in [general]: with persistent_index = False each "
            f"paper wipes the database the other papers are searching. Set workers = 1 or persistent_index = True.")
    return workers

class AgentStepLimitError(Exception):
    """The ReAct agent reached `max_iterations` without producing a final answer."""

def isAgentStepLimit(error):
    """True if `error` (or anything it wraps) is the agent's "max iterations reached" error."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, AgentStepLimitError) or re.search(r'max(imum)?[ _]iterations', str(error), re.I):
            return True
        error = error.__cause__ or error.__context__
    return False

def getHttpStatus(error):
    """Find an HTTP status code on an exception or anything it wraps; None if there is none."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for attr in ('status_code', 'status', 'http_status'):
            value = getattr(error, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(error, 'response', None)
        value = getattr(response, 'status_code', None)
        if isinstance(value, int):
            return value
        match = re.search(r'Error code: (\d{3})', str(error))
        if match:
            return int(match.group(1))
        error = error.__cause__ or error.__context__
    return None

def safeApiCall(call, enableLogging, *args, maxAttempts=8, maxWait=60, **kwargs):
    """
    Call `call(*args, **kwargs)`, retrying transient failures with capped exponential backoff
    (1, 2, 4, ... seconds, at most `maxWait` each). Errors that retrying cannot fix
    (see NON_RETRYABLE_STATUS) raise NonRetryableApiError immediately. A cancelled call
    (concurrent.futures.CancelledError: the event-loop thread is shutting down) and RunStopped are
    re-raised at once.
    """
    lastError = None
    for attempt in range(maxAttempts):
        try:
            startTime = time.time()
            result = call(*args, **kwargs)
            endTime = time.time()
            if enableLogging:
                log_str = f"API call successful. Duration: {endTime - startTime:.2f} seconds."
                logging.info(log_str)
            return result
        except (concurrent.futures.CancelledError, RunStopped):
            raise
        except Exception as e:
            status = getHttpStatus(e)
            if status in NON_RETRYABLE_STATUS:
                log_str = f"Non-retryable API error (HTTP {status}): {e}. Stopping."
                logging.error(log_str)
                print(log_str)
                raise NonRetryableApiError(status, str(e)) from e
            if isAgentStepLimit(e):
                # Not an API problem: the agent used up its reasoning steps. Retrying the identical call
                # would mostly repeat the same loop, so hand it back to the caller, which may ask once more.
                raise AgentStepLimitError(str(e)) from e
            lastError = e
            wait = min(2 ** attempt, maxWait) + random.random()
            if enableLogging:
                log_str = f"Error occurred (attempt {attempt + 1}/{maxAttempts}): {e}. Waiting for {wait:.2f} seconds."
                logging.warning(log_str)
                print(log_str)
            time.sleep(wait)
    raise Exception(f"API call failed after {maxAttempts} attempts; last error: {lastError}")

################################ File Management ################################
class FileManagement:
    """
    Manages file operations: uploading, listing, and clearing files.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.getConfigs()
        self.buildFolders()

    def getConfigs(self):
        self.inputDir = self.config.dir_structure['inputdir']
        self.logDir = self.config.dir_structure['logdir']
        self.dataPath = self.config.dir_structure['datapath']
        self.pluginsPath = self.config.dir_structure['pluginspath']
        self.outputDir = self.config.dir_structure['outputdir']
    
    def buildFolders(self):
        foldersToBuild = [self.inputDir, self.logDir, self.dataPath, self.pluginsPath,self.outputDir]
        for folder in foldersToBuild:
            ensureFolderExists(folder)
        
    def append(self, files):
        message = ''
        self.fileNames = []
        for file in files:
            fileName = os.path.basename(file)
            targetFileName = os.path.join(self.inputDir, fileName)
            shutil.copyfile(file, targetFileName)
            message += str(fileName) + '\n'
            self.fileNames.append(fileName)
        print("File(s) uploaded successfully!")
        message += str(len(files)) + " files were uploaded!\n"
        return message

    def upload(self, files):
        self.clear()
        return self.append(files)

    def clear(self):
        deleteFolderContents(self.inputDir)
        self.fileNames = []

    def ls(self):
        return self.fileNames
    
    def createDummyFile(self):
        touch(os.path.join(self.inputDir, 'dummy.txt'))

################################ Storage Management ################################
class Storage:
    """
    Manages storage using only the vector store.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.getConfigs()
        # The persistent index has its own node label and vector index, one per embedding dimension, so that
        # (a) tools that search the default `Chunk` nodes without a paper filter (the chatbot, runs with
        # persistent_index = False, earlier versions) never see the persistent papers, and (b) switching to an
        # embedding model of another dimension starts a new index instead of failing every search.
        storeNames = {}
        if self.persistent:
            storeNames = {"node_label": f"PaperChunk{self.embedDim}", "index_name": f"paper_chunk_{self.embedDim}"}
        self.vectorStore = Neo4jVectorStore(
            username=self.username,
            password=self.password,
            url=self.url,
            embedding_dimension=self.embedDim,
            **storeNames,
        )
        storedDim = getattr(self.vectorStore, 'embedding_dimension', self.embedDim)
        if self.persistent and storedDim != self.embedDim:
            raise PipelineConfigError(
                f"The Neo4j vector index {storeNames['index_name']} has dimension {storedDim}, but the embedding "
                f"model is configured with embedDim = {self.embedDim}. Fix embedDim, or drop that index.")
        self.storageContextVector = StorageContext.from_defaults(
            vector_store=self.vectorStore,
        )

    def getConfigs(self):
        self.url = self.config.neo4j['url1']
        self.username = self.config.neo4j['username']
        self.password = self.config.neo4j['password']
        service = self.config.llm_model['service']
        self.embedDim = self.config.dict[service]['embeddim']
        self.persistent = bool(self.config.general.get('persistent_index', True))

    def clear(self):
        try:
            driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            driver.close()
            return "Database cleared successfully!"
        except Exception as e:
            print(f"Error occurred while clearing the database: {e}")
            return f"Clearing the database failed with error: {e}"

class PaperIndexRegistry:
    """
    Which papers the persistent index holds. A paper's chunks are `Chunk` nodes with a `paper_key`; once all of
    them are written, an `IndexedPaper` marker records their number. A paper counts as indexed only if its marker
    exists and the chunk count still matches, so a paper interrupted mid-indexing is never reused.
    Every query is restricted to one paper key; nothing here touches other papers.
    """
    def __init__(self, storage, driver=None, nodeLabel=None) -> None:
        self._ownsDriver = driver is None
        self.driver = driver or GraphDatabase.driver(storage.url, auth=(storage.username, storage.password))
        self.label = nodeLabel or storage.vectorStore.node_label
        # Without these, the filtered search and the chunk counts scan every chunk of every paper.
        self._run(f"CREATE INDEX {self.label.lower()}_paper_key IF NOT EXISTS FOR (n:`{self.label}`) ON (n.paper_key)")
        self._run("CREATE INDEX indexed_paper_key IF NOT EXISTS FOR (m:IndexedPaper) ON (m.paper_key)")

    def _run(self, query, **params):
        with self.driver.session() as session:
            return [record.data() for record in session.run(query, **params)]

    def countChunks(self, paperKey):
        return self._run(f"MATCH (n:`{self.label}` {{paper_key: $k}}) RETURN count(n) AS n", k=paperKey)[0]["n"]

    def isIndexed(self, paperKey):
        marker = self._run("MATCH (m:IndexedPaper {paper_key: $k}) RETURN m.n_chunks AS n_chunks", k=paperKey)
        if not marker:
            return False
        expected = marker[0]["n_chunks"] or 0
        return expected > 0 and self.countChunks(paperKey) == expected

    def deletePaper(self, paperKey):
        self._run(f"MATCH (n:`{self.label}` {{paper_key: $k}}) DETACH DELETE n", k=paperKey)
        self._run("MATCH (m:IndexedPaper {paper_key: $k}) DETACH DELETE m", k=paperKey)

    def markIndexed(self, paperKey, fileName):
        n = self.countChunks(paperKey)
        if n == 0:
            raise ValueError(f"No chunks were stored for {fileName} (key {paperKey}); it was not indexed.")
        self._run("MERGE (m:IndexedPaper {paper_key: $k}) "
                  "SET m.file_name = $f, m.n_chunks = $n, m.indexed_at = $t",
                  k=paperKey, f=fileName, n=n, t=datetime.datetime.now().isoformat(timespec="seconds"))
        return n

    def close(self):
        if self._ownsDriver:
            self.driver.close()

################################ Index Building ################################
class Index:
    """
    Builds and manages the vector index for document retrieval.
    """
    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.config = config
        self.getConfigs()
        self.storage = Storage(config)
        self.vectorIndex = VectorStoreIndex.from_vector_store(vector_store=self.storage.vectorStore)
        self.fileManager = FileManagement(self.config)
    
    def getConfigs(self):
        self.showProgress = self.config.general['showprogress']
        self.enableLogging = self.config.general['enablelogging']
    
    def _buildVector(self, documents, batchSize=4):
        vectorIndex = None  # Initialize before the loop
        
        for batch in tqdm(iter_batch(documents, batchSize), desc='Building VectorIndex'):
            try:
                if vectorIndex is None:
                    vectorIndex = VectorStoreIndex.from_documents(
                        documents=batch,
                        storage_context=self.storage.storageContextVector,
                        show_progress=self.showProgress
                    )
                else:
                    vectorIndex.insert_nodes(batch)
            except Exception as e:
                print(f"Error processing batch: {e}")
                raise  # Re-raise to stop processing if batch fails
        
        if vectorIndex is None:
            raise ValueError("Failed to build vector index - no successful batches")
        
        self.vectorIndex = vectorIndex
        return vectorIndex
    
    def append(self, inputDocsDir, batchSize=1, paperKey=None, inputFiles=None):
        """Index every file in `inputDocsDir`, or only `inputFiles` (the persistent index reads exactly one PDF)."""
        source = {"input_files": inputFiles} if inputFiles else {"input_dir": inputDocsDir, "recursive": True}
        documents = SimpleDirectoryReader(
            **source,
            filename_as_id=True,
            file_extractor={
                '.csv': CustomCSVReader(),
                '.pdf': CustomPDFReader(llm=self.llm),
            },
        ).load_data()
        if paperKey:
            expected = os.path.basename(inputFiles[0]) if inputFiles and len(inputFiles) == 1 else None
            tagDocuments(documents, paperKey, expectedFile=expected)
        self._buildVector(documents, batchSize)
    
    def get(self):
        return self.vectorIndex
    
    def list_indexes(self):
        driver = GraphDatabase.driver(self.storage.url, auth=(self.storage.username, self.storage.password))
        indexes = []
        try:
            with driver.session() as session:
                result = session.run("SHOW INDEXES")
                for record in result:
                    indexes.append({
                        "name": record["name"],
                        "type": record["type"].upper(),
                        "owningConstraint": record.get("owningConstraint")
                    })
        except Exception as e:
            print(f"Error occurred while listing indexes: {e}")
        finally:
            driver.close()
        return indexes

    def drop_index(self, index_info):
        name = index_info["name"]
        owning_constraint = index_info.get("owningConstraint")
        driver = GraphDatabase.driver(self.storage.url, auth=(self.storage.username, self.storage.password))
        try:
            with driver.session() as session:
                if owning_constraint:
                    query = f"DROP CONSTRAINT {owning_constraint}"
                    print(f"Dropping constraint: {owning_constraint}")
                else:
                    query = f"DROP INDEX `{name}`"
                    print(f"Dropping index: {name}")
                session.run(query)
                print(f"Successfully dropped: {name}")
        except Exception as e:
            print(f"Error occurred while dropping index {name}: {e}")
        finally:
            driver.close()

    def clear(self):
        indexes = self.list_indexes()
        print("Indices to be dropped: ", indexes)
        for index_info in indexes:
            self.drop_index(index_info)
        print("Database cleared successfully!")

################################ Retrieval Guard ################################
class ContextLeakError(Exception):
    """A search for one paper returned a chunk of another paper (or of no known paper), or nothing at all."""


class PaperKeyGuard(BaseNodePostprocessor):
    """
    Second line of defence behind the retriever's paper filter: every retrieved chunk must carry the paper key
    of the paper being screened. Otherwise it raises ContextLeakError and records the violation, because the
    agent library turns tool errors into text and the model could otherwise go on to answer anyway; the answer
    loop reads the violation with consumeViolation() and records the question as failed.
    """
    paperKey: str
    _violation: Optional[str] = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "PaperKeyGuard"

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            message = f"the search for paper {self.paperKey} returned no passages"
        else:
            foreign = sorted({str(n.node.metadata.get('paper_key')) for n in nodes
                              if n.node.metadata.get('paper_key') != self.paperKey})
            message = (f"the search for paper {self.paperKey} returned passages of {', '.join(foreign)}"
                       if foreign else None)
        if message:
            self._violation = message
            raise ContextLeakError(message)
        return nodes

    def consumeViolation(self):
        message, self._violation = self._violation, None
        return message

################################ OpenRouter Helpers ################################
_openRouterContextCache = {}

def getOpenRouterContextWindow(modelName, apiKey=None, timeout=30, attempts=3):
    """
    Return the context length OpenRouter lists for `modelName`.

    Only a successful lookup is cached (once per process, so the listing is fetched once per run).
    The lookup is retried a few times; if it still fails, or the model is not listed, this raises
    PipelineConfigError instead of returning None: falling back to llama-index's 3,900-token default would
    silently truncate the retrieved context. Set `context_window` in the config to skip the lookup.
    """
    if modelName in _openRouterContextCache:
        return _openRouterContextCache[modelName]
    import requests
    headers = {"Authorization": f"Bearer {apiKey}"} if apiKey else {}
    lastError = None
    for attempt in range(attempts):
        try:
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=timeout)
            response.raise_for_status()
            models = response.json().get("data", [])
        except Exception as e:
            lastError = e
            print(f"Context-window lookup for '{modelName}' failed (attempt {attempt + 1}/{attempts}): {e}")
            time.sleep(2 ** attempt)
            continue
        for model in models:
            if model.get("id") == modelName:
                contextLength = model.get("context_length") or (model.get("top_provider") or {}).get("context_length")
                if contextLength:
                    _openRouterContextCache[modelName] = int(contextLength)
                    return _openRouterContextCache[modelName]
        raise PipelineConfigError(
            f"OpenRouter does not list the model '{modelName}' (or lists no context length for it). "
            f"Check `model_name` in the [openrouter] config section (OpenRouter ids look like "
            f"'mistralai/mistral-large-2512'), or set `context_window` there explicitly.")
    raise PipelineConfigError(
        f"Could not look up the context window for '{modelName}' on OpenRouter after {attempts} attempts "
        f"({lastError}). Set `context_window` in the [openrouter] config section to run without the lookup.")

################################ Query Engine ################################
class QueryEngine:
    """
    Builds and manages the query engine for vector-based retrieval.
    """
    # The retrieval tool returns plain text (relevant passages and a short answer). Only the agent
    # writes the JSON answer, so the tool output is never a second, competing JSON object.
    chatTextQaMsgs = [
        (
            "user",
            """You are given excerpts from one academic paper. Using only these excerpts, answer the query.
Quote the exact sentences that support your answer. If the excerpts do not contain the answer, say so plainly.
Do not use outside knowledge, and do not format your reply as JSON.

Excerpts:
{context_str}

Query:
{query_str}
"""
        )
    ]
    textQaTemplate = ChatPromptTemplate.from_messages(chatTextQaMsgs)
    successMsg = 'Query Engine was built successfully!'
    
    def __init__(self, llm, embedModel, config) -> None:
        self.config = config
        self.getConfigs()
        self.index = Index(llm, self.config)
        self.vectorTopK = self.config.retriever['vectortopk']
    
    def getConfigs(self):
        self.inputDir = self.config.dir_structure['inputdir']
        self.vectorTopK = self.config.retriever['vectortopk']
    
    def _buildVectorQueryEngine(self, forceReindex=False, paperKey=None, inputFiles=None):
        """
        With `paperKey`, every search is restricted to that paper's chunks, and a PaperKeyGuard checks every
        retrieved chunk before it reaches the model (self.leakGuard). Without it, the whole index is searched,
        as before (the database then holds only the current paper).
        """
        if forceReindex:
            self.index.append(inputDocsDir=self.inputDir, paperKey=paperKey, inputFiles=inputFiles)
        filters, postprocessors, self.leakGuard = None, [], None
        if paperKey:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="paper_key", value=paperKey, operator=FilterOperator.EQ)])
            self.leakGuard = PaperKeyGuard(paperKey=paperKey)
            postprocessors = [self.leakGuard]
        self.vectorRetriever = VectorIndexRetriever(
            index=self.index.vectorIndex,
            similarity_top_k=self.vectorTopK,
            filters=filters,
        )
        self.vectorQueryEngine = RetrieverQueryEngine.from_args(
            self.vectorRetriever,
            text_qa_template=self.textQaTemplate,
            node_postprocessors=postprocessors,
        )
        print("Vector Query Engine was built successfully!")
        return self.successMsg
    
    def get(self):
        rm = self._buildVectorQueryEngine()
        return self.vectorQueryEngine, rm


################################ Model Loading ################################
def loadModels(configPath):
    """Load the configured LLM and embedding model once, to share across the papers of a run: (llm, embedModel)."""
    loader = object.__new__(ChatbotAgents)
    loader.configPath = configPath
    loader.config = Config(configPath)
    loader.getConfigs()
    loader.getModelConfigs(configPath)
    loader._loadModels()
    # A wrong embedDim would not fail at indexing: every filtered search (which requires that size) would
    # just return nothing, and every question would fail as "no passages". Check it once, here.
    size = len(loader.embedModel.get_text_embedding("embedding size check"))
    if size != loader.embedDim:
        raise PipelineConfigError(
            f"The embedding model {getattr(loader, 'embeddingModelName', '')} returns {size}-dimensional vectors, "
            f"but embedDim = {loader.embedDim} in [{loader.service}]. Set embedDim = {size}.")
    return loader.llm, loader.embedModel

def freeModels(llm, embedModel):
    """Release an LLM and an embedding model (and their GPU memory, for HuggingFace models)."""
    if llm is not None:
        if hasattr(llm, '_model'):
            if hasattr(llm._model, 'cpu'):
                llm._model.cpu()
            del llm._model
        if getattr(llm, '_tokenizer', None) is not None:
            try:
                delattr(llm, '_tokenizer')
            except Exception:
                pass
    if embedModel is not None:
        if hasattr(embedModel, 'client'):
            del embedModel.client
        if hasattr(embedModel, '_model'):
            if hasattr(embedModel._model, 'cpu'):
                # 8-bit/4-bit (bitsandbytes) models cannot be moved; that must not skip the rest
                try:
                    embedModel._model.cpu()
                except Exception:
                    pass
            del embedModel._model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LoopThread:
    """A daemon thread running one asyncio event loop, shared by all paper workers."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.closing = False   # set by close(); runOnLoop then refuses to use the loop
        self.loop._ftsLoopThread = self
        started = threading.Event()
        def _run():
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(started.set)
            self.loop.run_forever()
        self.thread = threading.Thread(target=_run, name="fts-event-loop", daemon=True)
        self.thread.start()
        started.wait()

    def close(self, timeout=10):
        """Cancel pending tasks (so threads blocked in runOnLoop get CancelledError instead of
        hanging), stop the loop, join the thread and close the loop. Returns True on a clean exit."""
        self.closing = True
        if self.loop.is_closed():
            return True
        if self.loop.is_running():
            async def _shutdown():
                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if t is not current]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await self.loop.shutdown_asyncgens()
                finally:
                    self.loop.stop()
            self.loop.call_soon_threadsafe(lambda: self.loop.create_task(_shutdown()))
        self.thread.join(timeout)
        if self.thread.is_alive():
            logging.warning("Event-loop thread did not exit within %s s", timeout)
            return False
        self.loop.close()
        return True

def runOnLoop(loop, coroFactory):
    """Run a coroutine to completion on `loop`: submitted thread-safely when another thread
    runs the loop, otherwise with run_until_complete (the single-threaded behaviour).
    The loop of a LoopThread that is closing or closed is never used (not even run from this thread):
    RunStopped is raised at once."""
    owner = getattr(loop, '_ftsLoopThread', None)
    if owner is not None and (owner.closing or loop.is_closed()):
        raise RunStopped("the event-loop thread is shutting down")
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coroFactory(), loop)
        try:
            # poll, so Ctrl-C stays deliverable on Windows (an untimed wait is not interruptible there)
            while not future.done():
                concurrent.futures.wait([future], timeout=0.5)
            return future.result()
        except BaseException:
            future.cancel()
            raise
    return loop.run_until_complete(coroFactory())

def serializeEmbedding(embedModel, lock):
    """Make one embedding model safe to share between threads: every embedding call holds `lock`.
    The async methods run the (locked) synchronous method in a worker thread, so waiting for the
    lock never blocks the event-loop thread. Idempotent."""
    if getattr(embedModel, '_fts_serialized', False):
        return
    def lockedSync(original):
        def wrapper(*args, **kwargs):
            with lock:
                return original(*args, **kwargs)
        wrapper.__name__ = getattr(original, '__name__', 'wrapper')
        return wrapper
    syncNames = ['_get_query_embedding', '_get_text_embedding', '_get_text_embeddings']
    originals = {n: getattr(embedModel, n) for n in syncNames + ['_aget_query_embedding', '_aget_text_embedding']
                 if hasattr(embedModel, n)}
    wrapped = {n: lockedSync(originals[n]) for n in syncNames if n in originals}
    for name, fn in wrapped.items():
        object.__setattr__(embedModel, name, fn)
    for name in ('_aget_query_embedding', '_aget_text_embedding'):
        if name not in originals:
            continue
        syncFn = wrapped.get(name.replace('_aget_', '_get_'))
        if syncFn is None:
            # no synchronous counterpart: drive the original coroutine in a private loop, under the lock
            asyncOriginal = originals[name]
            def syncFn(*args, _orig=asyncOriginal, **kwargs):
                with lock:
                    return asyncio.run(_orig(*args, **kwargs))
        async def asyncWrapper(*args, _fn=syncFn, **kwargs):
            return await asyncio.to_thread(_fn, *args, **kwargs)
        object.__setattr__(embedModel, name, asyncWrapper)
    object.__setattr__(embedModel, '_fts_serialized', True)

################################ Chatbot Agents ################################

class ChatbotAgents:
    """
    Integrates the language model, query engine, and related tools.
    This version supports only vector-based retrieval.
    """
    successMsg = 'Chatbot Agents updated successfully!'
    availableServices = ['huggingface', 'mistral', 'openrouter']

    def __init__(self, configPath, models=None, loop=None) -> None:
        """
        `models`: an (llm, embedModel) pair from loadModels(), shared by every paper of a run, so the models are
        loaded once instead of once per paper. Without it, this instance loads (and on cleanup frees) its own.
        `loop`: the run's event loop, shared by every paper. Shared models hold async clients bound to the loop
        they first ran on, so all papers must use the same loop; the run closes it at the end. Without it, this
        instance creates (and on cleanup closes) its own.
        """
        self.configPath = configPath
        self.config = Config(configPath)
        # Same setting the database layer (Storage) reads: True keeps the persistent index on cleanup
        self.persistent = bool(self.config.general.get('persistent_index', True))
        self.getConfigs()
        self.getModelConfigs(self.configPath)
        if models is not None:
            self.llm, self.embedModel = models
            self.ownsModels = False
        else:
            self._loadModels()
            self.ownsModels = True
        Settings.llm = self.llm
        Settings.embed_model = self.embedModel
        # One memory buffer per ChatbotAgents instance (i.e., per paper): every question for this paper is
        # answered with the full history of earlier questions and answers. A new paper creates a new
        # instance, so memory starts empty. The event loop is the run's (shared) or this instance's own.
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=self.effectiveMemoryLimit())
        self.ownsLoop = loop is None
        self._loop = asyncio.new_event_loop() if loop is None else loop
        self.updateQueryEngine()
        self.selectDefaultAgent()
    
    def _loadModels(self):
        """Set self.llm and self.embedModel for the configured service."""
        if self.service == 'huggingface':
            self.getHuggingFaceLlmAndEmbedding()
        elif self.service == 'mistral':
            self.getMistralLlmAndEmbedding()
        elif self.service == 'openrouter':
            self.getOpenRouterLlmAndEmbedding()
        else:
            raise ValueError(f'service must be in ({self.availableServices})')

    # Tokens kept free in every request besides the memory: system prompt, tool description,
    # the question, and the retrieved passages (vectorTopK chunks of up to chunk_size tokens each).
    PROMPT_MARGIN_TOKENS = 20000
    MIN_MEMORY_TOKENS = 2000

    def effectiveMemoryLimit(self):
        """
        The configured memory_token_limit, capped so that memory plus a prompt and a reply still fit in
        the model's context window. Without the cap, a model with a smaller window would eventually get
        HTTP 400 (prompt too long), which stops the run.
        """
        limit = self.memory_token_limit
        contextWindow = getattr(getattr(self.llm, 'metadata', None), 'context_window', None)
        if contextWindow:
            cap = contextWindow - self.max_new_tokens - self.PROMPT_MARGIN_TOKENS
            cap = max(cap, self.MIN_MEMORY_TOKENS)
            if cap < limit:
                print(f"memory_token_limit {limit} exceeds what fits in the {contextWindow}-token context window; "
                      f"using {cap}.")
                limit = cap
        return limit

    def getModelConfigs(self, configPath):
        config = ConfigParser()
        config.read(configPath)
        self.service = config.get('llm-model', 'service')
        self.llmModel = config.get(self.service, 'model_name')
        self.embeddingModelName = config.get(self.service, 'embed_model')
        self.embedDim = config.getint(self.service, 'embedDim')
        self.apiKey = config.get(self.service, 'api_key')
        self.max_new_tokens = config.getint('llm-model', 'max_new_tokens')
        self.temperature = config.getfloat('llm-model', 'temperature')
        self.do_sample = config.getboolean('llm-model', 'do_sample')
        self.max_iterations = config.getint('agent', 'max_iterations')
        self.enable_agent = config.getboolean('agent', 'enable_agent')
        # Max tokens of question/answer history kept in the per-paper agent memory
        self.memory_token_limit = config.getint('agent', 'memory_token_limit', fallback=100000)
        # Optional fixed context window for the selected service (0 = look it up / use the default)
        self.contextWindow = config.getint(self.service, 'context_window', fallback=0)
        # Optional: where and at what precision to run the HuggingFace embedding model.
        # Defaults to CPU at full precision; see buildHuggingFaceEmbedding for the options.
        self.embedDevice = config.get(self.service, 'embed_device', fallback='cpu')
        self.embedQuantization = config.get(self.service, 'embed_quantization', fallback='none')
        # Maximum token length of a text passed to a HuggingFace embedding model. The default keeps the
        # earlier behaviour (embedDim); chunks are usually far shorter (chunk_size), so it rarely matters.
        self.embedMaxLength = config.getint(self.service, 'embed_max_length', fallback=self.embedDim)

    def getMistralLlmAndEmbedding(self):
        from llama_index.llms.mistralai import MistralAI
        self.llm = MistralAI(
            model=self.llmModel,
            api_key=self.apiKey,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )
        if 'mistral' in self.embeddingModelName:
            from llama_index.embeddings.mistralai import MistralAIEmbedding
            self.embedModel = MistralAIEmbedding(
                model=self.embeddingModelName,
                api_key=self.apiKey
            )
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.embedModel = HuggingFaceEmbedding(
                model_name=self.embeddingModelName, 
                max_length=self.embedMaxLength,
                trust_remote_code=True
            )


    
    def getHuggingFaceLlmAndEmbedding(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from transformers import BitsAndBytesConfig

        # system_prompt = "<|SYSTEM|>\n" + systemMessage + "\n"
        # query_wrapper_prompt = PromptTemplate(
        #     "<|USER|>\n{query_str}\n<|ASSISTANT|>\n"
        # )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.llm = HuggingFaceLLM(
            context_window=10000,
            max_new_tokens=self.max_new_tokens,
            generate_kwargs={"temperature": self.temperature, "do_sample": self.do_sample},
            # system_prompt=system_prompt,
            # query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.llmModel,
            model_name=self.llmModel,
            device_map="auto",
            model_kwargs={
                "quantization_config": quantization_config,
            },
        )
        self.embedModel = HuggingFaceEmbedding(
            model_name=self.embeddingModelName, 
            max_length=self.embedMaxLength,
            trust_remote_code=True
        )

    def getOpenRouterLlmAndEmbedding(self):
        from llama_index.llms.openrouter import OpenRouter

        # Context window: an explicit `context_window` in the [openrouter] config section wins;
        # otherwise use the model's own context length from OpenRouter's model listing.
        # Without this, llama-index falls back to its 3,900-token default for every OpenRouter model.
        # The lookup raises a clear error rather than falling back to the default.
        contextWindow = self.contextWindow or getOpenRouterContextWindow(self.llmModel, self.apiKey)

        # LLM using OpenRouter API
        self.llm = OpenRouter(
            api_key=self.apiKey,
            model=self.llmModel,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            context_window=contextWindow,
        )
        print(f"OpenRouter model {self.llmModel}: context_window = {self.llm.metadata.context_window}")

        # Embedding model (still from HuggingFace): on the GPU when available, otherwise on the CPU
        self.embedModel = self.buildHuggingFaceEmbedding()

        print("OpenRouter LLM and embeddings loaded successfully.")

    def buildHuggingFaceEmbedding(self):
        """
        Load the HuggingFace embedding model.

        Default (no config needed): CPU, full precision (fp32), exactly as before.

        Optional config keys in the service section (e.g., [openrouter]):
          embed_device       = cpu (default) | cuda | auto
                               cuda: use the GPU; auto: use the GPU only if PyTorch sees one.
                               Either way, it falls back to the CPU if no GPU is available or
                               loading on the GPU fails (e.g., out of memory).
          embed_quantization = none (default) | fp16 | 8bit | 4bit
                               Precision on the GPU only; the CPU always uses fp32.
                               A 7B model such as gte-Qwen2-7B-instruct needs 8bit or 4bit to fit
                               on a 16 GB card. Lower precision changes the embeddings slightly,
                               which can change which chunks are retrieved.
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        device = (self.embedDevice or 'cpu').strip().lower()
        quantization = (self.embedQuantization or 'none').strip().lower()
        if device not in ('cpu', 'cuda', 'auto'):
            print(f"Unknown embed_device '{device}'; using the CPU.")
            device = 'cpu'
        if quantization not in ('none', 'fp16', '8bit', '4bit'):
            print(f"Unknown embed_quantization '{quantization}'; using none.")
            quantization = 'none'
        cudaAvailable = torch.cuda.is_available()

        if device == 'cuda' and not cudaAvailable:
            print("embed_device = cuda, but no CUDA GPU is available to PyTorch; using the CPU.")
        if device == 'cpu' and quantization != 'none':
            print(f"embed_quantization = {quantization} applies only on the GPU; the CPU uses fp32.")

        if cudaAvailable and device in ('auto', 'cuda'):
            modelKwargs = {}
            if quantization in ('8bit', '4bit'):
                from transformers import BitsAndBytesConfig
                if quantization == '8bit':
                    modelKwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    modelKwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
            elif quantization == 'fp16':
                modelKwargs['torch_dtype'] = torch.float16
            try:
                embedModel = HuggingFaceEmbedding(
                    model_name=self.embeddingModelName,
                    max_length=self.embedMaxLength,
                    trust_remote_code=True,
                    device='cuda',
                    model_kwargs=modelKwargs,
                )
                precision = 'cuda-fp32' if quantization == 'none' else f'cuda-{quantization}'
                object.__setattr__(embedModel, '_fts_precision', precision)
                print(f"Embedding model {self.embeddingModelName} loaded on GPU "
                      f"({torch.cuda.get_device_name(0)}), precision: {precision}.")
                return embedModel
            except Exception as e:
                print(f"Could not load the embedding model on the GPU ({type(e).__name__}: {e}); "
                      f"falling back to the CPU.")
                gc.collect()
                torch.cuda.empty_cache()

        embedModel = HuggingFaceEmbedding(
            model_name=self.embeddingModelName,
            max_length=self.embedMaxLength,
            trust_remote_code=True,
            device='cpu',
        )
        object.__setattr__(embedModel, '_fts_precision', 'cpu-fp32')
        print(f"Embedding model {self.embeddingModelName} loaded on CPU, precision: cpu-fp32.")
        return embedModel

        
    def getConfigs(self):
        self.inputDir = self.config.dir_structure['inputdir']
    
    def updateQueryEngine(self):
        self.queryEngine = QueryEngine(self.llm, self.embedModel, self.config)
        self.queryEngine._buildVectorQueryEngine(forceReindex=False)
        self._buildAll()
    
    def _buildAll(self):
        customPromptMessages = [
            ChatMessage(role="system", content=systemMessage),
        ]
        self.customPromptTemplate = ChatPromptTemplate(customPromptMessages)
        self.vectorQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.vectorQueryEngine,
            metadata=ToolMetadata(
                name="vector_rag_query_engine",
                description="Searches the paper being screened and returns the most relevant passages. "
                            "Use it for every question: answers must rest on what it returns, not on prior knowledge.",
            ),
        )
        # llama-index >= 0.13 workflow ReActAgent: the system prompt is passed as
        # `system_prompt` (the old `prefix_messages` argument was silently ignored),
        # and max_iterations/memory are passed per call in `chatbot()`.
        # early_stopping_method="generate": when the agent reaches max_iterations it makes one final
        # call to write an answer from what it has gathered, instead of raising an error.
        self.vectorAgent = ReActAgent(
            tools=[self.vectorQueryEngineTool],
            llm=self.llm,
            system_prompt=systemMessage,
            verbose=True,
            early_stopping_method="generate",
        )
    
    def setPaper(self, paperKey):
        """Restrict the agent's search tool to one paper's chunks (with the leak guard)."""
        self.queryEngine._buildVectorQueryEngine(paperKey=paperKey)
        self._buildAll()
        self.selectDefaultAgent()

    @property
    def leakGuard(self):
        queryEngine = getattr(self, 'queryEngine', None)
        return getattr(queryEngine, 'leakGuard', None)

    def appendIndex(self, vectorTopK, cutoffScore, paperKey=None, inputFiles=None):
        self.updateParams(vectorTopK, cutoffScore)
        self.queryEngine._buildVectorQueryEngine(forceReindex=True, paperKey=paperKey, inputFiles=inputFiles)
        self._buildAll()
        return self.successMsg
    
    def getDefault(self):
        self.queryEngine.index.fileManager.createDummyFile()
        self.vectorAgent, _ = self.getAll(forceReindex=False)
        return self.vectorAgent
    
    def getAll(self, forceReindex=False):
        if forceReindex:
            self.appendIndex(self.config.retriever['vectortopk'], self.config.retriever['cutoffscore'])
        return self.vectorAgent, self.successMsg
    
    def updateParams(self, vectorTopK, cutoffScore):
        self.config.retriever['vectortopk'] = vectorTopK
        self.config.retriever['cutoffscore'] = cutoffScore
        self.updateQueryEngine()
    
    def chatbot(self, queryStr, history):
        # messages = [
        #     {"role": "system", "content": systemMessage},
        #     {"role": "user", "content": queryStr},
        # ]
        # The query should include instructions to output a JSON with the required keys.
        if self.enable_agent:
            # Run the workflow agent with this paper's shared memory, so the answer can
            # draw on all earlier questions and answers for the same paper.
            async def _run():
                handler = self.selectedAgent.run(
                    user_msg=queryStr,
                    memory=self.memory,
                    max_iterations=self.max_iterations,
                )
                return await handler
            # Snapshot the memory so that a failed attempt (e.g., an API error) does not leave
            # its question or partial reasoning behind; a retry then starts from a clean history.
            memorySnapshot = list(self.memory.get_all())
            try:
                agentOutput = runOnLoop(self._loop, _run)
            except BaseException:
                self.memory.set(memorySnapshot)
                raise
            return str(agentOutput.response.content)
        else:
            chatOutput = self.queryEngine.vectorQueryEngine.query(queryStr)
        # print("=======> chatOutput",chatOutput)
        # Here we assume the LLM returns a JSON-formatted string.
        # Optionally, you might want to process references; for now we pass the raw output.
        output = chatOutput.response
        return output
    
    def uploadFiles(self, files):
        try:
            msg = self.queryEngine.index.fileManager.upload(files)
            print(msg)
            return msg
        except Exception as e:
            print(e)
            return "Uploading files failed!"
    
    def clearFiles(self):
        try:
            self.queryEngine.index.storage.clear()
            # self.queryEngine.index.fileManager.clear()
            self.queryEngine.index.clear()
            return "Database and input directory were cleared successfully!"
        except Exception as e:
            print("Exception occurred while clearing the database: ", e)
            return "Clearing the database and input directory failed!"
    
    def getFilesList(self):
        return self.queryEngine.index.fileManager.ls()
    
    def selectAgent(self, indexType=None):
        self.selectedAgent = self.vectorAgent
    
    def selectDefaultAgent(self):
        self.selectAgent()
    
    def getSelectedAgent(self):
        return self.selectedAgent
    
    def cleanup(self):
        """
        Comprehensive cleanup of resources to free GPU memory
        """
        try:
            # Clear existing data (a persistent index keeps every paper's chunks)
            if not self.persistent:
                self.clearFiles()

            # Drop this paper's agent memory, and close its event loop if it is this instance's own
            if getattr(self, 'ownsLoop', True) and getattr(self, '_loop', None) is not None                     and not self._loop.is_closed():
                self._loop.close()
            self.memory = None

            # Free the LLM and embedding model only if this instance loaded them; models shared
            # across papers (passed in with `models=`) are freed once at the end of the run.
            if self.ownsModels:
                freeModels(getattr(self, 'llm', None), getattr(self, 'embedModel', None))
            self.llm = None
            self.embedModel = None

            # Reset agent components
            if hasattr(self, 'vectorQueryEngineTool'):
                del self.vectorQueryEngineTool
                self.vectorQueryEngineTool = None
                
            if hasattr(self, 'vectorAgent'):
                del self.vectorAgent
                self.vectorAgent = None
                
            if hasattr(self, 'queryEngine'):
                # Clean up vectorRetriever
                if hasattr(self.queryEngine, 'vectorRetriever'):
                    del self.queryEngine.vectorRetriever
                
                # Clean up index and its components
                if hasattr(self.queryEngine, 'index'):
                    # Access any vector store
                    if hasattr(self.queryEngine.index, 'vector_store'):
                        del self.queryEngine.index.vector_store
                    
                    # Clean up any docstore
                    if hasattr(self.queryEngine.index, 'docstore'):
                        del self.queryEngine.index.docstore
                        
                    # Delete the index itself
                    del self.queryEngine.index
                
                # Clear the vector query engine
                if hasattr(self.queryEngine, 'vectorQueryEngine'):
                    del self.queryEngine.vectorQueryEngine
                    
                # Finally delete the query engine
                del self.queryEngine
                self.queryEngine = None
                
            # Force garbage collection multiple times
            import gc
            gc.collect()
            gc.collect()
            
            # Aggressive CUDA cache clearing
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Additional memory cleanup
                torch.cuda.ipc_collect()
                
            return "Resources cleaned up successfully!"
        except Exception as e:
            print("Exception occurred during cleanup: ", e)
            return "Resource cleanup failed!"

################################ Literature Screening Function ################################
# Keys of a stored answer, in the order the model is asked to produce them (evidence first,
# then reasoning, then the answer). QuestionText is filled in from the questionnaire, not by the model.
ANSWER_KEYS = ("Evidence", "Reasoning", "ShortAnswer")
_KEY_ALIASES = {
    "evidence": "Evidence", "quote": "Evidence", "quotes": "Evidence",
    "reasoning": "Reasoning", "justification": "Reasoning", "explanation": "Reasoning", "rationale": "Reasoning",
    "shortanswer": "ShortAnswer", "short_answer": "ShortAnswer", "answer": "ShortAnswer",
    "questiontext": "QuestionText", "question": "QuestionText",
}

def _decodeJsonObjects(text):
    """Yield every JSON value that can be decoded from `text`: the whole text first, then each '{...}' in it."""
    decoder = json.JSONDecoder(strict=False)
    stripped = text.strip()
    try:
        yield decoder.decode(stripped)
    except ValueError:
        pass
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
            yield value
        except ValueError:
            continue

def _normalizeAnswer(value, depth=0):
    """
    Map a decoded JSON value onto {"Evidence", "Reasoning", "ShortAnswer"}; None if it has no answer.
    Handles alias key names (answer/justification/evidence), a JSON string holding JSON, and an answer
    nested one level deep (e.g., {"Q7": {...}} or {"ShortAnswer": {"ShortAnswer": ...}}).
    """
    if depth > 3:
        return None
    if isinstance(value, str):
        inner = parseAnswer(value, _depth=depth + 1)
        return inner
    if not isinstance(value, dict):
        return None
    mapped = {}
    for key, item in value.items():
        canonical = _KEY_ALIASES.get(re.sub(r'[\s-]', '', str(key)).lower())
        if canonical and canonical not in mapped:
            mapped[canonical] = item
    shortAnswer = mapped.get("ShortAnswer")
    if isinstance(shortAnswer, dict):
        nested = _normalizeAnswer(shortAnswer, depth + 1)
        if nested is not None:
            return nested
        # A structured answer (e.g., {"technique": ..., "description": ...}): keep it as one string
        mapped["ShortAnswer"] = "; ".join(f"{k}: {v}" for k, v in shortAnswer.items())
    if "ShortAnswer" not in mapped:
        # The answer may be wrapped one level deep, e.g. {"Q7": {...}} or {"result": {...}}
        for item in value.values():
            nested = _normalizeAnswer(item, depth + 1)
            if nested is not None:
                return nested
        return None
    answer = {key: mapped.get(key, "") for key in ANSWER_KEYS}
    return answer

def parseAnswer(text, _depth=0):
    """
    Parse the model's reply into {"Evidence", "Reasoning", "ShortAnswer"}; None if no answer can be found.

    Accepts code fences, text before or after the JSON object, escaped newlines inside strings,
    alias key names and an answer nested inside another object.
    """
    if text is None:
        return None
    text = str(text)
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        for value in _decodeJsonObjects(candidate):
            answer = _normalizeAnswer(value, _depth)
            if answer is not None:
                return answer
    # Last resort: a value written as several quoted strings ("a"; "b" or "a", "b"), which is not JSON
    for candidate in candidates:
        repaired = _joinSplitStrings(candidate)
        if repaired == candidate:
            continue
        for value in _decodeJsonObjects(repaired):
            answer = _normalizeAnswer(value, _depth)
            if answer is not None:
                return answer
    return None

_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)

def _joinSplitStrings(text):
    """
    Join a value the model wrote as several quoted strings separated by ';' or ',' (e.g. "Evidence": "a"; "b")
    into one string ("a; b"). A string counts as a continuation only if it follows such a value and is not a
    key (not followed by ':'). Strings in arrays and valid key/value pairs are left as they are.
    """
    tokens = list(_JSON_STRING.finditer(text))
    out, last, i = [], 0, 0
    while i < len(tokens):
        token = tokens[i]
        isValue = text[:token.start()].rstrip().endswith(':')
        out.append(text[last:token.start()])
        end, j, parts = token.end(), i + 1, [token.group()]
        if isValue:
            while j < len(tokens):
                gap = text[end:tokens[j].start()].strip()
                isKey = text[tokens[j].end():].lstrip().startswith(':')
                if gap not in (';', ',') or isKey:
                    break
                parts.append(tokens[j].group())
                end, j = tokens[j].end(), j + 1
        if len(parts) > 1:
            try:
                out.append(json.dumps("; ".join(json.loads(part, strict=False) for part in parts), ensure_ascii=False))
            except ValueError:
                out.append(text[token.start():end])
        else:
            out.append(token.group())
        last, i = end, j
    out.append(text[last:])
    return "".join(out)

def cleanAndConvertJson(jsonText):
    """Backward-compatible wrapper: the parsed answer, or the original text if it cannot be parsed."""
    parsed = parseAnswer(jsonText)
    return parsed if parsed is not None else jsonText

################################ Save individual responses ################################
def writeJsonAtomic(path, data):
    """Write JSON to `path` via a temporary file, so a crash mid-write never leaves a truncated file."""
    tmpPath = f"{path}.tmp"
    with open(tmpPath, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, cls=DateTimeEncoder)
    os.replace(tmpPath, path)

def save_individual_result(paper_id, paper_data, output_dir, suffix=""):
    """
    Save individual paper results to a JSON file named after the paper_id.

    Args:
        paper_id (str): The ID/filename of the paper
        paper_data (dict): The paper's data including metadata and answers
        output_dir (str): Directory where individual results should be saved
        suffix (str): Optional suffix before ".json" (e.g., ".partial" for an incomplete paper)
    """
    ensureFolderExists(output_dir)
    output_path = os.path.join(output_dir, f"{paper_id}{suffix}.json")
    writeJsonAtomic(output_path, paper_data)
    print(f"Saved individual results for {paper_id} to {output_path}")

def questionnaireHash(questions):
    """A fingerprint of the questionnaire, stored with each result so that resume can tell versions apart."""
    return hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

# Bump whenever the PDF reader or the chunking code changes, so that every paper is indexed again.
READER_VERSION = "1"

def embeddingPrecision(embedModel):
    """
    What the embedding model actually loaded as: 'cuda-8bit', 'cuda-4bit', 'cuda-fp16', 'cuda-fp32' or 'cpu-fp32'.
    buildHuggingFaceEmbedding records it on the model; a model without it counts as the CPU default.
    """
    return getattr(embedModel, '_fts_precision', 'cpu-fp32')

def computePaperKey(pdfPath, config, embedPrecision='cpu-fp32'):
    """
    The key that identifies a paper's chunks in the persistent index: a fingerprint of the PDF's bytes and of
    every setting that changes how it is chunked or embedded. The file name is not part of it, so a renamed
    copy reuses the index; a replaced PDF or changed settings get a new key, and so a fresh index.
    The embedding precision is part of it, except for the CPU default (fp32), so that keys made before
    the precision existed stay valid.
    """
    nodeParser = config.nodeparser
    service = config.llm_model['service']
    settings = {
        "nodeParserType": nodeParser.get('nodeparsertype'),
        "chunk_size": nodeParser.get('chunk_size'),
        "chunk_overlap": nodeParser.get('chunk_overlap'),
        "bufferSize": nodeParser.get('buffersize'),
        "breakpointPercentileThreshold": nodeParser.get('breakpointpercentilethreshold'),
        "embed_model": config.dict[service].get('embed_model'),
        "reader_version": READER_VERSION,
    }
    if embedPrecision != 'cpu-fp32':
        settings["embed_precision"] = embedPrecision
    digest = hashlib.sha256()
    with open(pdfPath, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]

def tagDocuments(documents, paperKey, expectedFile=None):
    """
    Store `paperKey` on each document (its chunks inherit it), without changing what is embedded or sent to the LLM.
    With `expectedFile`, every document must come from that file; otherwise ContextLeakError (nothing is tagged).
    """
    if expectedFile is not None:
        others = sorted({str(d.metadata.get('file_name')) for d in documents
                         if d.metadata.get('file_name') != expectedFile})
        if others:
            raise ContextLeakError(f"indexing {expectedFile} also read {', '.join(others)}; nothing was indexed")
    for doc in documents:
        doc.metadata['paper_key'] = paperKey
        for keys in (doc.excluded_embed_metadata_keys, doc.excluded_llm_metadata_keys):
            if 'paper_key' not in keys:
                keys.append('paper_key')
    return documents

def checkCompleteResult(data, questions, qHash):
    """
    Whether a saved result can be reused when resuming: (True, "") or (False, reason).
    A result is reusable if it has a dict answer without an error for every question and, when it records
    the questionnaire it was made with, that questionnaire is the current one. Results written before the
    fingerprint was added are accepted if they are otherwise complete.
    """
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        return False, "it has no answers"
    if data.get("questionnaire_sha256") and data["questionnaire_sha256"] != qHash:
        return False, "it was made with a different questionnaire"
    answers = data["answers"]
    missing = [q for q in questions if q not in answers]
    if missing:
        return False, f"{len(missing)} answer(s) are missing"
    broken = [q for q in questions if not isinstance(answers[q], dict) or answers[q].get("error")]
    if broken:
        return False, f"{len(broken)} answer(s) are unreadable or failed ({', '.join(broken[:5])})"
    return True, ""


# Minimum number of words a PDF's text layer must yield before it is screened
MIN_PDF_WORDS = 200

def countPdfWords(pdfPath):
    """Number of words in the PDF's text layer (0 for scanned, image-only PDFs or unreadable files)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdfPath)
        return sum(len((page.extract_text() or '').split()) for page in reader.pages)
    except Exception as e:
        print(f"Could not read text from {pdfPath}: {e}")
        return 0

def process_single_paper(row, questions, configPath, input_pdf_folder, models=None, reindex=False, loop=None,
                         stopEvent=None):
    """
    Process a single paper:
      - Instantiate a new screeningAgent (ChatbotAgents), so memory starts empty; `models` (from loadModels)
        are shared across papers.
      - persistent_index = True (default): index the PDF only if the persistent index does not already hold it
        (or if `reindex`), then restrict every search to this paper's chunks, checked by PaperKeyGuard.
        persistent_index = False: reset the database and index the PDF, as before 0.10.0.
      - Ask every question about the indexed document.
      - Return {pdf_filename: {"metadata", "questionnaire_sha256", "answers"}}.

    The result is keyed by `pdf_filename` (earlier versions used the numeric `paper_id`).

    Saving: a complete result goes to <outputDir>/individual/<pdf>.json. A paper with failed questions,
    or one interrupted by an error, goes to <pdf>.partial.json instead, so a resumed run redoes it.
    NonRetryableApiError, PipelineConfigError and KeyboardInterrupt are re-raised after saving, and so is
    RunStopped, raised before the next question once `stopEvent` (a threading.Event) is set.
    """
    paper_id = row["pdf_filename"]
    pdf_filename = row.get("pdf_filename", f"{paper_id}.pdf")
    print(f"\nProcessing paper {paper_id} with PDF file: {pdf_filename}")
    qHash = questionnaireHash(questions)

    config = Config(configPath)
    individual_output_dir = os.path.join(config.dir_structure['outputdir'], 'individual')

    persistent = bool(config.general.get('persistent_index', True))
    paperKey = None

    # Instantiate a new screeningAgent for this paper (new memory; the models may be shared).
    screeningAgent = ChatbotAgents(configPath=configPath, models=models, loop=loop)
    screeningAgent.persistent = persistent
    try:
        screeningAgent.selectDefaultAgent()

        if not persistent:
            # Reset the database and clear any uploaded files.
            screeningAgent.clearFiles()

        # Build the full path to the PDF file.
        pdf_path = os.path.join(input_pdf_folder, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"PDF file {pdf_path} not found. Skipping paper {paper_id}.")
            return {paper_id: {"metadata": row, "answers": {"error": f"PDF file {pdf_path} not found."}}}

        # Refuse PDFs without a usable text layer (e.g., scanned pages): with nothing to retrieve,
        # the LLM would answer from its own knowledge and the answers would look valid but be invented.
        textWords = countPdfWords(pdf_path)
        if textWords < MIN_PDF_WORDS:
            msg = (f"PDF has no usable text layer ({textWords} words extracted; likely scanned). "
                   f"Run OCR on it first. Skipping paper {paper_id}.")
            print(msg)
            return {paper_id: {"metadata": row, "answers": {"error": msg}}}

        vectorTopK = screeningAgent.config.retriever['vectortopk']
        cutoffScore = screeningAgent.config.retriever['cutoffscore']
        enableLogging = screeningAgent.config.general['enablelogging']

        try:
            if persistent:
                # Index the paper only if the persistent index does not already hold it completely, then
                # restrict every search to its chunks (see computePaperKey and PaperIndexRegistry).
                precision = embeddingPrecision(screeningAgent.embedModel)
                paperKey = computePaperKey(pdf_path, config, precision)
                registry = PaperIndexRegistry(screeningAgent.queryEngine.index.storage)
                try:
                    if reindex or not registry.isIndexed(paperKey):
                        registry.deletePaper(paperKey)   # leftovers of an interrupted or forced indexing
                        # Read exactly this PDF (not the upload folder, which may hold leftovers of other papers)
                        screeningAgent.appendIndex(vectorTopK, cutoffScore, paperKey=paperKey, inputFiles=[pdf_path])
                        nChunks = registry.markIndexed(paperKey, pdf_filename)
                        print(f"Indexed {pdf_filename} ({nChunks} chunks, key {paperKey}, embeddings {precision})")
                    else:
                        print(f"Reusing the index of {pdf_filename} (key {paperKey}, embeddings {precision})")
                finally:
                    registry.close()
                screeningAgent.setPaper(paperKey)
            else:
                # Upload and index the PDF using current vector retrieval settings.
                print(screeningAgent.uploadFiles([pdf_path]))
                screeningAgent.appendIndex(vectorTopK, cutoffScore)
        except RunStopped:
            raise
        except Exception as e:
            # Indexing calls the LLM (citation extraction). An error that retrying cannot fix
            # (e.g., out of credit) would fail every following paper too, so stop the run.
            status = getHttpStatus(e)
            if status in NON_RETRYABLE_STATUS:
                raise NonRetryableApiError(status, f"while indexing {paper_id}: {e}") from e
            print(f"Error indexing PDF: {e}")
            return {paper_id: {"metadata": row, "answers": {"error": f"Error indexing PDF: {e}"}}}

        paperKeyField = {"paper_key": paperKey} if paperKey else {}
        paper_answers = {}
        try:
            failed = _answerAllQuestions(screeningAgent, questions, paper_answers, enableLogging, paper_id,
                                         stopEvent=stopEvent)
        except BaseException as e:
            # Keep what was answered so far in "<paper>.partial.json" (never mistaken for a complete
            # result) and re-raise so the caller can stop the run.
            partial = {"metadata": row, "questionnaire_sha256": qHash, **paperKeyField, "answers": paper_answers,
                       "error": f"{type(e).__name__}: {e}",
                       "answered": f"{len(paper_answers)}/{len(questions)}"}
            save_individual_result(paper_id, partial, individual_output_dir, suffix=".partial")
            raise

        paper_result = {"metadata": row, "questionnaire_sha256": qHash, **paperKeyField, "answers": paper_answers}
        stalePartial = os.path.join(individual_output_dir, f"{paper_id}.partial.json")
        if failed:
            paper_result["error"] = f"{failed} of {len(questions)} question(s) could not be answered"
            print(f"{paper_id}: {paper_result['error']}; saved as a partial result so that a resumed run redoes it.")
            save_individual_result(paper_id, paper_result, individual_output_dir, suffix=".partial")
        else:
            save_individual_result(paper_id, paper_result, individual_output_dir)
            # A complete result supersedes any partial result from an earlier, aborted run
            if os.path.exists(stalePartial):
                os.remove(stalePartial)
        return {paper_id: paper_result}
    finally:
        # Clean up by deleting the agent, collecting garbage, and emptying the GPU cache.
        screeningAgent.cleanup()
        del screeningAgent
        gc.collect()
        torch.cuda.empty_cache()


ANSWER_INSTRUCTIONS = """

Use the query engine tool to find the passages that answer this question; include the valid short answers in your search query.
Then reply in English with a single JSON object with exactly these keys, in this order:
    "Evidence": quote the exact sentences from the paper that support your answer. If there is no evidence, write "Not Applicable".
    "Reasoning": explain briefly how the evidence leads to your answer.
    "ShortAnswer": one of the valid short answers, or only the items explicitly stated in the paper.
Write the evidence and reasoning before deciding on the short answer."""

JSON_REMINDER = ("\n\nYour previous reply could not be read as the required JSON object. Reply with only one JSON object "
                 "with the keys \"Evidence\", \"Reasoning\" and \"ShortAnswer\", in this order, and nothing else.")

def _errorAnswer(q_text, message, raw=None):
    """A placeholder answer recording why a question could not be answered."""
    answer = {"QuestionText": q_text, **{key: "" for key in ANSWER_KEYS}, "error": message}
    if raw is not None:
        answer["raw"] = raw
    return answer

def _answerAllQuestions(screeningAgent, questions, paper_answers, enableLogging, paper_id="", stopEvent=None):
    """
    Ask every question for the current paper, filling `paper_answers` as answers arrive.

    A question whose reply cannot be parsed is asked once more with a reminder of the format. A question
    that still fails, or whose API calls keep failing, gets an answer with an "error" key and the run moves
    on to the next question. Only errors that retrying cannot fix (NonRetryableApiError) end the paper.
    If `stopEvent` is set, RunStopped is raised before the next question.
    Returns the number of questions that ended with an error.
    """
    failed = 0
    leakGuard = getattr(screeningAgent, 'leakGuard', None)
    for q_key, q_text in tqdm(questions.items(), desc=f"Processing questions for paper {paper_id}", total=len(questions)):
        if stopEvent is not None and stopEvent.is_set():
            raise RunStopped(f"the run is stopping; {len(paper_answers)} of {len(questions)} question(s) answered")
        query = q_text + ANSWER_INSTRUCTIONS
        parsed, raw, lastError = None, None, None
        for attempt in range(2):
            try:
                raw = safeApiCall(
                    screeningAgent.chatbot,
                    enableLogging=enableLogging,
                    queryStr=query if attempt == 0 else query + JSON_REMINDER,
                    history=None
                )
            except (NonRetryableApiError, RunStopped):
                raise
            except concurrent.futures.CancelledError as e:
                # the event-loop thread cancelled the call: the run is stopping, do not ask again
                raise RunStopped("the question was cancelled because the run is stopping") from e
            except Exception as e:
                lastError = f"{type(e).__name__}: {e}"
                print(f"{q_key}: attempt {attempt + 1} failed ({lastError})")
                continue
            print(raw)
            if leakGuard is not None:
                violation = leakGuard.consumeViolation()
                if violation:
                    lastError = f"context leak: {violation}"
                    print(f"{q_key}: {lastError}; the answer is discarded.")
                    break
            parsed = parseAnswer(raw)
            if parsed is not None:
                break
            lastError = "the reply could not be parsed as the required JSON object"
            print(f"{q_key}: {lastError}; asking again.")
        if parsed is None:
            failed += 1
            paper_answers[q_key] = _errorAnswer(q_text, lastError, raw)
        else:
            paper_answers[q_key] = {"QuestionText": q_text, **parsed}
    return failed


def literature_screening(metadata_file_path, questions, configPath, input_pdf_folder, output_json_path, reindex=False):
    """
    Screen every paper in the metadata file (csv or xlsx), one after another, or `workers` at a time
    (`workers` in [general], see _screenInParallel):
      - each paper gets a fresh ChatbotAgents instance (new memory), which reuses or builds the paper's index
        and answers every question (see process_single_paper); the LLM and embedding model are loaded once,
        when the first paper needs processing, and shared by all papers;
      - with `skip_existing = True` in [general], papers with a reusable result in
        <outputDir>/individual/<pdf>.json are loaded instead of processed again (see checkCompleteResult);
      - an unexpected error in one paper is recorded and the run moves on to the next paper;
      - errors that affect every paper (NonRetryableApiError, PipelineConfigError) and Ctrl-C stop the run.
    The combined results are written to `output_json_path` however the run ends.
    Logs the total processing time and average time per paper.
    """
    start_time = time.time()

    results = {}
    rows = []
    if metadata_file_path.endswith('.csv'):
        with open(metadata_file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
    elif metadata_file_path.endswith('.xlsx'):
        df = pd.read_excel(metadata_file_path)
        df.fillna('', inplace=True)  # Fill NaN values with empty strings
        rows = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file type: {metadata_file_path}")

    config = Config(configPath)
    workers = readWorkers(config)
    skipExisting = bool(config.general.get('skip_existing', False))
    individualDir = os.path.join(config.dir_structure['outputdir'], 'individual')
    qHash = questionnaireHash(questions)

    def writeCombined():
        writeJsonAtomic(output_json_path, results)
        print(f"\nScreening results for {len(results)} paper(s) saved to {output_json_path}")

    if workers > 1:
        _screenInParallel(rows, questions, configPath, input_pdf_folder, results, writeCombined, workers,
                          reindex, skipExisting, individualDir, qHash)
        _printRunSummary(results, rows, start_time)
        return

    models = None   # loaded once, when the first paper needs processing, and shared by all papers
    loop = None     # the run's event loop, shared by all papers (the shared models' async clients are bound to it)
    try:
        for i, row in enumerate(rows):
            paperId = row["pdf_filename"]
            if skipExisting and _skipIfReusable(row, i, len(rows), individualDir, questions, qHash, results):
                continue

            try:
                if models is None:
                    models = loadModels(configPath)
                    loop = asyncio.new_event_loop()
                result = process_single_paper(row, questions, configPath, input_pdf_folder,
                                              models=models, reindex=reindex, loop=loop)
            except (NonRetryableApiError, PipelineConfigError) as e:
                print(f"\nStopped at paper {i+1}/{len(rows)} ({paperId}): {e}\n"
                      f"Answers for this paper so far are in {individualDir}. "
                      f"Fix the cause and rerun with skip_existing = True to resume.")
                raise
            except Exception as e:
                # An unexpected problem with this paper only: record it and go on with the next paper.
                _recordPaperError(results, row, e, f"{i+1}/{len(rows)} ({paperId})", "Continuing with the next paper.")
                continue
            results.update(result)
            print(f"\nProcessing completed for paper {i+1}/{len(rows)}.")
    finally:
        # Also runs on Ctrl-C and on errors that stop the run, so the combined output is never stale.
        writeCombined()
        if loop is not None and not loop.is_closed():
            loop.close()
        if models is not None:
            freeModels(*models)

    _printRunSummary(results, rows, start_time)


def _printRunSummary(results, rows, start_time):
    print(f"Processing complete for {len(results)} papers.")
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_paper = total_time / len(rows) if rows else 0

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per paper: {avg_time_per_paper:.2f} seconds")


def _skipIfReusable(row, i, total, individualDir, questions, qHash, results):
    """skip_existing: if <individualDir>/<pdf>.json is a reusable complete result (checkCompleteResult),
    put it in `results` and return True; otherwise return False (saying why a saved result is not reused)."""
    paperId = row["pdf_filename"]
    existing = os.path.join(individualDir, f"{paperId}.json")
    if not os.path.exists(existing):
        return False
    try:
        with open(existing, encoding="utf-8") as f:
            data = json.load(f)
        reusable, reason = checkCompleteResult(data, questions, qHash)
    except (OSError, ValueError) as e:
        reusable, reason = False, f"it cannot be read ({e})"
    if reusable:
        results[paperId] = data
        print(f"\nSkipping paper {i+1}/{total} (already processed): {paperId}")
        return True
    print(f"\nThe saved result for {paperId} is not reusable ({reason}); processing the paper again.")
    return False


def _recordPaperError(results, row, error, label, nextStep):
    """An unexpected problem with one paper only: log it and record it in `results` as answers.error."""
    paperId = row["pdf_filename"]
    logging.error(f"Paper {paperId} failed", exc_info=(type(error), error, error.__traceback__))
    print(f"\nPaper {label} failed: {type(error).__name__}: {error}. {nextStep}")
    results[paperId] = {"metadata": row, "answers": {"error": f"{type(error).__name__}: {error}"}}


def _waitAll(futures):
    """Wait until every future is done, polling so that Ctrl-C stays deliverable on Windows."""
    pending = [f for f in futures if not f.done()]
    while pending:
        concurrent.futures.wait(pending, timeout=0.5)
        pending = [f for f in pending if not f.done()]


def _screenInParallel(rows, questions, configPath, input_pdf_folder, results, writeCombined, workers,
                      reindex, skipExisting, individualDir, qHash):
    """
    The `workers > 1` branch of literature_screening: up to `workers` papers are processed at once, in threads
    of one process. They share one LLM and one embedding model (loaded once; embedding calls are serialized
    with a lock, see serializeEmbedding) and one event loop running in its own thread (LoopThread).
    `results` (the combined output) and the progress counter are changed only here, in the main thread.
    An ordinary error in one paper is recorded and the others go on. A run-wide error (NonRetryableApiError,
    PipelineConfigError) or Ctrl-C sets `stopEvent`: papers not yet started are dropped, running papers stop
    at their next question (RunStopped; their answers so far are in <pdf>.partial.json), and the first
    run-wide error, or KeyboardInterrupt, is re-raised.
    """
    total = len(rows)
    stopEvent = threading.Event()
    models = loopThread = executor = None
    futures = {}
    try:
        todo = [row for i, row in enumerate(rows)
                if not (skipExisting and _skipIfReusable(row, i, total, individualDir, questions, qHash, results))]
        finished = total - len(todo)   # papers done so far, for "Processing completed for paper k/N."
        if not todo:
            return
        models = loadModels(configPath)
        serializeEmbedding(models[1], threading.Lock())
        loopThread = LoopThread()

        def work(row):
            if stopEvent.is_set():
                raise RunStopped("the run stopped before this paper started")
            try:
                return process_single_paper(row, questions, configPath, input_pdf_folder, models=models,
                                            reindex=reindex, loop=loopThread.loop, stopEvent=stopEvent)
            except (NonRetryableApiError, PipelineConfigError, KeyboardInterrupt):
                # set here, in the worker, so that this thread does not start the next queued paper
                stopEvent.set()
                raise

        print(f"\nProcessing {len(todo)} paper(s), {workers} at a time.")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fts-paper")
        futures = {executor.submit(work, row): row for row in todo}
        firstError = None

        def stopRun():
            stopEvent.set()
            for f in futures:
                f.cancel()   # only papers not yet started can be cancelled

        collected = set()   # a future is added only once its result or error is recorded

        def collect(future):
            """Put a finished paper's result (or its error) in `results`; return a run-wide error, if any."""
            nonlocal finished
            row = futures[future]
            error = None
            if not future.cancelled():
                try:
                    result = future.result()
                except RunStopped:
                    pass   # stopped because of another paper's error or Ctrl-C: not this paper's error
                except (NonRetryableApiError, PipelineConfigError, KeyboardInterrupt) as e:
                    error = e
                except Exception as e:
                    _recordPaperError(results, row, e, row["pdf_filename"], "Continuing with the other papers.")
                else:
                    results.update(result)
                    finished += 1
                    print(f"\nProcessing completed for paper {finished}/{total}.")
            collected.add(future)
            return error

        pending = set(futures)
        try:
            while pending:
                # poll, so Ctrl-C stays deliverable on Windows (an untimed wait is not interruptible there);
                # wait() never raises a paper's own exception, only .result() in collect() does
                done, pending = concurrent.futures.wait(pending, timeout=0.5,
                                                        return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    error = collect(future)
                    if isinstance(error, KeyboardInterrupt):
                        raise error
                    if error is not None and firstError is None:
                        firstError = error
                        print(f"\nStopped at paper {futures[future]['pdf_filename']}: {error}\n"
                              f"Waiting for the papers in progress to stop at their next question. Answers so far "
                              f"are in {individualDir}. Fix the cause and rerun with skip_existing = True to resume.")
                        stopRun()
        except KeyboardInterrupt:
            print("\nInterrupted: waiting for the papers in progress to stop at their next question; their answers "
                  "so far are saved as partial results. Press Ctrl-C again to stop at once: the questions in "
                  "progress are abandoned, those papers may get no partial result, and the program exits.")
            stopRun()
            _waitAll(futures)
            # papers that finished before or while waiting, but were not collected yet
            for future in futures:
                if future not in collected:
                    collect(future)
            raise
        if firstError is not None:
            raise firstError
    finally:
        # Also runs on Ctrl-C (even a second one while waiting) and on errors that stop the run.
        writeCombined()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if loopThread is not None:
            loopThread.close()
        if models is not None:
            if any(not f.done() for f in futures):
                # a second Ctrl-C: some papers are still running; do not free the models under them
                print("Stopping at once: papers still in progress are abandoned; the models are not freed "
                      "(the program exits once their current step returns).")
            else:
                freeModels(*models)



################################ Main Execution ################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = './Config/ConfigLitScr.cfg'
LEGACY_CONFIG_PATH = './Config/ConfigLitScrInUse.cfg'   # name used by earlier versions
DEFAULT_QUESTIONS_PATH = os.path.join(SCRIPT_DIR, 'Questionnaires', 'questions_v1.json')

def resolveConfigPath(cliPath=None):
    """
    The config file to use: --config, else the FTS_CONFIG environment variable, else
    ./Config/ConfigLitScr.cfg, else the earlier default ./Config/ConfigLitScrInUse.cfg.
    """
    explicit = cliPath or os.environ.get('FTS_CONFIG')
    if explicit:
        if not os.path.isfile(explicit):
            raise PipelineConfigError(f"Config file not found: {explicit}")
        return explicit
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        return DEFAULT_CONFIG_PATH
    if os.path.isfile(LEGACY_CONFIG_PATH):
        print(f"Using {LEGACY_CONFIG_PATH} (the earlier default name). "
              f"Rename it to {DEFAULT_CONFIG_PATH} or pass --config to silence this message.")
        return LEGACY_CONFIG_PATH
    raise PipelineConfigError(
        f"No config file found. Copy Config/ConfigLitScrTemp.cfg to {DEFAULT_CONFIG_PATH} and fill it in, "
        f"or pass --config <path>.")

def resolveQuestionsPath(config):
    """`questions_file_path` from [dir-structure]; configs from before that key existed get questions_v1.json."""
    path = config.dir_structure.get('questions_file_path')
    if not path:
        print(f"No questions_file_path in [dir-structure]; using {DEFAULT_QUESTIONS_PATH} "
              f"(the questions earlier versions had built in).")
        return DEFAULT_QUESTIONS_PATH
    if not os.path.isfile(path):
        raise PipelineConfigError(f"questions_file_path does not exist: {path}")
    return path

def parseArgs(argv=None):
    parser = argparse.ArgumentParser(description="Answer a questionnaire for each paper in a metadata file.")
    parser.add_argument('--config', help="Path to the config file (default: FTS_CONFIG, then "
                                         f"{DEFAULT_CONFIG_PATH}, then {LEGACY_CONFIG_PATH}).")
    parser.add_argument('--reindex', action='store_true',
                        help="Index every paper of the run again, even if the persistent index already holds it.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parseArgs()
    configPath = resolveConfigPath(args.config)
    config = Config(configPath)
    print(f"Loaded configuration from {configPath}:")
    print(config.neo4j)
    print(config.dir_structure)
    print(config.general)
    print(config.retriever)
    print(config.nodeparser)

    # Extract some default parameters
    appVersion = config.general['appversion']
    huggingFaceToken = getattr(config, 'huggingface', {}).get('api_key')

    # logging into huggingface
    # Never print the token itself; only whether one is set
    print(f"###### Hugging Face Token: {'set' if huggingFaceToken else 'not set'}")
    if huggingFaceToken:
        try:
            huggingface_hub.login(huggingFaceToken)
            print("Successfully logged into huggingface!")
        except Exception as e:
            print(f"Error logging into huggingface: {e}")

    logger = setLogging(config) or logging.getLogger(__name__)

    try:
        setChunkingMethod(config)

        metadata_csv_path = config.dir_structure['metadatafile']      # CSV file with paper metadata (must include "pdf_filename")
        input_pdf_folder = config.dir_structure['input_pdf_folder']  # Folder where all paper PDFs are stored
        output_json_path = config.dir_structure['outputfile']        # File to save all screening results
        questions_file_path = resolveQuestionsPath(config)          # JSON file with the questions

        with open(questions_file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # Run the literature screening process.
        literature_screening(metadata_csv_path, questions, configPath, input_pdf_folder, output_json_path,
                             reindex=args.reindex)

    except Exception as e:
        logger.error(f"Exception occurred:\n\n {e}", exc_info=True)
        sys.exit(1)
