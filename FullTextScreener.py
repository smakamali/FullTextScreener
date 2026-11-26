#################################### appVersion = 0.9.0 ######################################
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

import os
import sys
import gc
import shutil
import logging
import time
import random
import re
import json
import csv
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
import asyncio

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
    Rule 3: Output must be in JSON format. Do NOT include any surrounding explanation or formatting.
    Rule 4: Each answer must include:
        - "answer": the short-form answer (e.g., Yes, No, Unsure, or a concise phrase).
        - "justification": a short natural language explanation of why that answer was selected.
        - "evidence": the most semantically similar span or sentence(s) from the source text.
    Rule 5: Use the query engine to find relevant content for every question. Do not rely on prior interactions or history.
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
import json
import datetime

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
    touch(logFile)
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
    (see NON_RETRYABLE_STATUS) raise NonRetryableApiError immediately.
    """
    for attempt in range(maxAttempts):
        try:
            startTime = time.time()
            result = call(*args, **kwargs)
            endTime = time.time()
            if enableLogging:
                log_str = f"API call successful. Duration: {endTime - startTime:.2f} seconds."
                logging.info(log_str)
            return result
        except Exception as e:
            status = getHttpStatus(e)
            if status in NON_RETRYABLE_STATUS:
                log_str = f"Non-retryable API error (HTTP {status}): {e}. Stopping."
                logging.error(log_str)
                print(log_str)
                raise NonRetryableApiError(status, str(e)) from e
            wait = min(2 ** attempt, maxWait) + random.random()
            if enableLogging:
                log_str = f"Error occurred (attempt {attempt + 1}/{maxAttempts}): {e}. Waiting for {wait:.2f} seconds."
                logging.warning(log_str)
                print(log_str)
            time.sleep(wait)
    raise Exception(f"API call failed after {maxAttempts} attempts")

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
        self.vectorStore = Neo4jVectorStore(
            username=self.username,
            password=self.password,
            url=self.url,
            embedding_dimension=self.embedDim,
        )
        self.storageContextVector = StorageContext.from_defaults(
            vector_store=self.vectorStore,
        )

    def getConfigs(self):
        self.url = self.config.neo4j['url1']
        self.username = self.config.neo4j['username']
        self.password = self.config.neo4j['password']
        service = self.config.llm_model['service']
        self.embedDim = self.config.dict[service]['embeddim']

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
    
    def append(self, inputDocsDir, batchSize=1):
        documents = SimpleDirectoryReader(
            inputDocsDir,
            recursive=True,
            filename_as_id=True,
            file_extractor={
                '.csv': CustomCSVReader(),
                '.pdf': CustomPDFReader(llm=self.llm),
            },
        ).load_data()
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

################################ OpenRouter Helpers ################################
_openRouterContextCache = {}

def getOpenRouterContextWindow(modelName, apiKey=None, timeout=30):
    """
    Return the context length OpenRouter lists for `modelName`, or None if it cannot be found.
    The result is cached per process, so the listing is fetched once per run, not once per paper.
    """
    if modelName in _openRouterContextCache:
        return _openRouterContextCache[modelName]
    contextLength = None
    try:
        import requests
        headers = {"Authorization": f"Bearer {apiKey}"} if apiKey else {}
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=timeout)
        response.raise_for_status()
        for model in response.json().get("data", []):
            if model.get("id") == modelName:
                contextLength = model.get("context_length") or (model.get("top_provider") or {}).get("context_length")
                break
        if contextLength is None:
            print(f"OpenRouter model '{modelName}' not found in the model listing; using llama-index's default context window.")
    except Exception as e:
        print(f"Could not look up the context window for '{modelName}' on OpenRouter ({e}); using llama-index's default.")
    _openRouterContextCache[modelName] = int(contextLength) if contextLength else None
    return _openRouterContextCache[modelName]

################################ Query Engine ################################
class QueryEngine:
    """
    Builds and manages the query engine for vector-based retrieval.
    """
    chatTextQaMsgs = [
        (
            "user",
            systemMessage + """
            Context:
            {context_str}
            Question:
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
    
    def _buildVectorQueryEngine(self, forceReindex=False):
        if forceReindex:
            self.index.append(inputDocsDir=self.inputDir)
        self.vectorRetriever = VectorIndexRetriever(
            index=self.index.vectorIndex,
            similarity_top_k=self.vectorTopK
        )
        self.vectorQueryEngine = RetrieverQueryEngine.from_args(
            self.vectorRetriever,
            text_qa_template=self.textQaTemplate,
        )
        print("Vector Query Engine was built successfully!")
        return self.successMsg
    
    def get(self):
        rm = self._buildVectorQueryEngine()
        return self.vectorQueryEngine, rm


################################ Chatbot Agents ################################

# openrouter class
# class OpenRouterLLM:
#     def __init__(self, api_key, model_name, temperature=0.0, max_new_tokens=1024):
#         self.api_key = api_key
#         self.model_name = model_name
#         self.temperature = temperature
#         self.max_new_tokens = max_new_tokens

#     def generate(self, prompt):
#         import requests
#         url = "https://openrouter.ai/api/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": self.temperature,
#             "max_tokens": self.max_new_tokens
#         }
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         return data["choices"][0]["message"]["content"]

class ChatbotAgents:
    """
    Integrates the language model, query engine, and related tools.
    This version supports only vector-based retrieval.
    """
    successMsg = 'Chatbot Agents updated successfully!'
    def __init__(self, configPath) -> None:
        self.availableServices = ['huggingface', 'mistral',"openrouter"]
        self.configPath = configPath
        self.config = Config(configPath)
        self.getConfigs()
        self.getModelConfigs(self.configPath)
        if self.service == self.availableServices[0]:  # huggingface
            self.getHuggingFaceLlmAndEmbedding()
        elif self.service == self.availableServices[1]:  # mistral
            self.getMistralLlmAndEmbedding()
            # add open router
        elif self.service == self.availableServices[2]:  # openrouter
            self.getOpenRouterLlmAndEmbedding()
        else:
            raise ValueError(f'service must be in ({self.availableServices})')
        Settings.llm = self.llm
        Settings.embed_model = self.embedModel
        # One memory buffer and event loop per ChatbotAgents instance (i.e., per paper):
        # every question for this paper is answered with the full history of earlier
        # questions and answers. A new paper creates a new instance, so memory starts empty.
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        self._loop = asyncio.new_event_loop()
        self.updateQueryEngine()
        self.selectDefaultAgent()
    
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
                max_length=self.embedDim,
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
            device_map="cuda:0",
            model_kwargs={
                "quantization_config": quantization_config,
            },
        )
        self.embedModel = HuggingFaceEmbedding(
            model_name=self.embeddingModelName, 
            max_length=self.embedDim,
            trust_remote_code=True
        )

    def getOpenRouterLlmAndEmbedding(self):
        from llama_index.llms.openrouter import OpenRouter

        # Context window: an explicit `context_window` in the [openrouter] config section wins;
        # otherwise use the model's own context length from OpenRouter's model listing.
        # Without this, llama-index falls back to its 3,900-token default for every OpenRouter model.
        contextWindow = self.contextWindow or getOpenRouterContextWindow(self.llmModel, self.apiKey)
        llmKwargs = {}
        if contextWindow:
            llmKwargs['context_window'] = contextWindow

        # LLM using OpenRouter API
        self.llm = OpenRouter(
            api_key=self.apiKey,
            model=self.llmModel,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            **llmKwargs
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
                    max_length=self.embedDim,
                    trust_remote_code=True,
                    device='cuda',
                    model_kwargs=modelKwargs,
                )
                print(f"Embedding model {self.embeddingModelName} loaded on GPU "
                      f"({torch.cuda.get_device_name(0)}), precision: "
                      f"{'fp32' if quantization == 'none' else quantization}.")
                return embedModel
            except Exception as e:
                print(f"Could not load the embedding model on the GPU ({type(e).__name__}: {e}); "
                      f"falling back to the CPU.")
                gc.collect()
                torch.cuda.empty_cache()

        embedModel = HuggingFaceEmbedding(
            model_name=self.embeddingModelName,
            max_length=self.embedDim,
            trust_remote_code=True,
            device='cpu',
        )
        print(f"Embedding model {self.embeddingModelName} loaded on CPU, precision: fp32.")
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
                description="ATTENTION: Always use this tool to answer questions, ignore your memory or any prior knowledge.",
            ),
        )
        # llama-index >= 0.13 workflow ReActAgent: the system prompt is passed as
        # `system_prompt` (the old `prefix_messages` argument was silently ignored),
        # and max_iterations/memory are passed per call in `chatbot()`.
        self.vectorAgent = ReActAgent(
            tools=[self.vectorQueryEngineTool],
            llm=self.llm,
            system_prompt=systemMessage,
            verbose=True,
        )
    
    def appendIndex(self, vectorTopK, cutoffScore):
        self.updateParams(vectorTopK, cutoffScore)
        self.queryEngine._buildVectorQueryEngine(forceReindex=True)
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
                agentOutput = self._loop.run_until_complete(_run())
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
            # Clear existing data
            self.clearFiles()

            # Drop this paper's agent memory and close its event loop
            if getattr(self, '_loop', None) is not None and not self._loop.is_closed():
                self._loop.close()
            self.memory = None

            # For HuggingFaceLLM and HuggingFaceEmbedding specifically
            if hasattr(self, 'llm') and self.llm is not None:
                # Access and clear the underlying model if possible
                if hasattr(self.llm, '_model'):
                    if hasattr(self.llm._model, 'cpu'):
                        # Move model to CPU first
                        self.llm._model.cpu()
                    del self.llm._model
                
                # Clear tokenizer if it exists (only for HuggingFace LLMs)
                tokenizer_attr = getattr(self.llm, "_tokenizer", None)
                if tokenizer_attr is not None:
                    try:
                        delattr(self.llm, "_tokenizer")
                    except Exception:
                        pass

                    
                # Delete the entire LLM instance
                del self.llm
                self.llm = None
                
            if hasattr(self, 'embedModel') and self.embedModel is not None:
                # Access and clear the underlying model
                if hasattr(self.embedModel, 'client'):
                    del self.embedModel.client
                if hasattr(self.embedModel, '_model'):
                    if hasattr(self.embedModel._model, 'cpu'):
                        # Move model to CPU first. 8-bit/4-bit (bitsandbytes) models cannot be moved,
                        # so a failure here must not skip the rest of the cleanup.
                        try:
                            self.embedModel._model.cpu()
                        except Exception:
                            pass
                    del self.embedModel._model
                    
                # Delete the entire embedding model instance
                del self.embedModel
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
def cleanAndConvertJson(jsonText):
    """
    Cleans and converts a JSON string into a Python dictionary.

    Args:
        jsonText (str): JSON text as a string, possibly with extra characters.

    Returns:
        dict: Parsed JSON data as a dictionary, or None if parsing fails.
    """
    cleanedText = jsonText.strip("'```json").strip('```')

    cleanedText = cleanedText.replace("\\'", "'").replace('\\n', '\n')


    cleanedText = '\n'.join([line.strip() for line in cleanedText.splitlines()])
    try:
        return json.loads(cleanedText)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return cleanedText

################################ Save individual responses ################################
def save_individual_result(paper_id, paper_data, output_dir, suffix=""):
    """
    Save individual paper results to a JSON file named after the paper_id.

    Args:
        paper_id (str): The ID/filename of the paper
        paper_data (dict): The paper's data including metadata and answers
        output_dir (str): Directory where individual results should be saved
        suffix (str): Optional suffix before ".json" (e.g., ".partial" for an aborted paper)
    """
    ensureFolderExists(output_dir)
    output_path = os.path.join(output_dir, f"{paper_id}{suffix}.json")
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(paper_data, outfile, indent=2, cls=DateTimeEncoder)
    print(f"Saved individual results for {paper_id} to {output_path}")


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

def process_single_paper(row, questions, configPath, input_pdf_folder):
    """
    Process a single paper:
      - Instantiate a new screeningAgent (ChatbotAgents) to refresh memory.
      - Reset the database and index store.
      - Upload and index the corresponding PDF.
      - Iterate over the questions and query the indexed document.
      - Return a dictionary mapping paper_id to its answers.
    """
    paper_id = row["pdf_filename"]  # Treat paper_id as a string instead of converting to int
    pdf_filename = row.get("pdf_filename", f"{paper_id}.pdf")
    print(f"\nProcessing paper {paper_id} with PDF file: {pdf_filename}")
    
    # Instantiate a new screeningAgent for this paper.
    screeningAgent = ChatbotAgents(configPath=configPath)
    screeningAgent.selectDefaultAgent()
    
    # Reset the database and clear any uploaded files.
    screeningAgent.clearFiles()
    
    # Build the full path to the PDF file.
    pdf_path = os.path.join(input_pdf_folder, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"PDF file {pdf_path} not found. Skipping paper {paper_id}.")
        screeningAgent.cleanup()
        return {paper_id: {"metadata": row, "answers": {"error": f"PDF file {pdf_path} not found."}}}

    # Refuse PDFs without a usable text layer (e.g., scanned pages): with nothing to retrieve,
    # the LLM would answer from its own knowledge and the answers would look valid but be invented.
    textWords = countPdfWords(pdf_path)
    if textWords < MIN_PDF_WORDS:
        msg = (f"PDF has no usable text layer ({textWords} words extracted; likely scanned). "
               f"Run OCR on it first. Skipping paper {paper_id}.")
        print(msg)
        screeningAgent.cleanup()
        return {paper_id: {"metadata": row, "answers": {"error": msg}}}

    # Upload the PDF file.
    upload_msg = screeningAgent.uploadFiles([pdf_path])
    print(upload_msg)
    
    # Index the uploaded PDF using current vector retrieval settings.
    vectorTopK = screeningAgent.config.retriever['vectortopk']
    cutoffScore = screeningAgent.config.retriever['cutoffscore']
    enableLogging = screeningAgent.config.general['enablelogging']

    try:
        screeningAgent.appendIndex(vectorTopK, cutoffScore)
    except Exception as e:
        print(f"Error indexing PDF: {e}")
        screeningAgent.cleanup()
        return {paper_id: {"metadata": row, "answers": {"error": f"Error indexing PDF: {e}"}}}
    
    config = Config(configPath)
    individual_output_dir = os.path.join(config.dir_structure['outputdir'], 'individual')

    # For each question, query the document.
    paper_answers = {}
    try:
        _answerAllQuestions(screeningAgent, questions, paper_answers, enableLogging, paper_id)
    except Exception as e:
        # Keep what was answered so far in "<paper>.partial.json" (never mistaken for a complete
        # result), clean up, and re-raise so the caller can stop the run.
        partial = {"metadata": row, "answers": paper_answers,
                   "error": f"{type(e).__name__}: {e}",
                   "answered": f"{len(paper_answers)}/{len(questions)}"}
        save_individual_result(paper_id, partial, individual_output_dir, suffix=".partial")
        screeningAgent.cleanup()
        del screeningAgent
        gc.collect()
        torch.cuda.empty_cache()
        raise

    # Create the result dictionary for this paper
    paper_result = {paper_id: {"metadata": row, "answers": paper_answers}}

    # Save individual result
    save_individual_result(paper_id, paper_result[paper_id], individual_output_dir)
    # A complete result supersedes any partial result from an earlier, aborted run
    stalePartial = os.path.join(individual_output_dir, f"{paper_id}.partial.json")
    if os.path.exists(stalePartial):
        os.remove(stalePartial)

    # Clean up by deleting the agent, collecting garbage, and emptying the GPU cache.
    screeningAgent.cleanup()
    del screeningAgent
    gc.collect()
    torch.cuda.empty_cache()

    return paper_result


def _answerAllQuestions(screeningAgent, questions, paper_answers, enableLogging, paper_id=""):
    """Ask every question for the current paper, filling `paper_answers` as answers arrive."""
    for q_key, q_text in tqdm(questions.items(), desc=f"Processing questions for paper {paper_id}", total=len(questions)):
        full_query = (
            q_text +
            """\n
            ATTENTION: When answering questions, always use the query engine tool and do not rely on your internal memory.
            Include all valid short answers in your query string when using the query engine tool.
            Answer in English, in the following JSON format:
                QuestionText: exactly repeat the question with short valid answers,
                ShortAnswer: one of the valid short answers or extract only items explicitly stated in the document,
                Reasoning: specify your reasoning or any additional notes for providing the answer,
                Evidence: quote the exact sentences from the paper that support your answer and reasoning. If evidence is not available, say "Not Applicable"."""
        )
        answer = safeApiCall(
            screeningAgent.chatbot,
            enableLogging=enableLogging,
            queryStr=full_query,
            history=None
        )

        print(answer)

        parsed_answer = cleanAndConvertJson(answer)
        paper_answers[q_key] = parsed_answer


def literature_screening(metadata_file_path, questions, configPath, input_pdf_folder, output_json_path):
    """
    For each paper in the metadata file (csv or xlsx):
      - Process the paper in a separate subprocess.
      - The per-paper process creates a fresh ChatbotAgents instance, resets the state,
        uploads and indexes the PDF, then answers each question.
      - Adds a 3-minute delay between processing each paper.
    Finally, all results are saved in a single JSON file.
    Additionally, logs the total processing time and average time per paper.
    """
    import time

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
    
    # Optional resume: with `skip_existing = True` in [general], papers that already have a complete
    # result in <outputDir>/individual/<paper>.json are not processed again; their saved results are
    # loaded so the combined output still contains every paper. Partial results (*.partial.json) are redone.
    config = Config(configPath)
    skipExisting = bool(config.general.get('skip_existing', False))
    individualDir = os.path.join(config.dir_structure['outputdir'], 'individual')

    def writeCombined():
        with open(output_json_path, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=2, cls=DateTimeEncoder)

    # Process papers sequentially with delay instead of using multiprocessing
    for i, row in enumerate(rows):
        paperId = row["pdf_filename"]
        existing = os.path.join(individualDir, f"{paperId}.json")
        if skipExisting and os.path.exists(existing):
            with open(existing, encoding="utf-8") as f:
                results[paperId] = json.load(f)
            print(f"\nSkipping paper {i+1}/{len(rows)} (already processed): {paperId}")
            continue

        # Process the current paper; on an error that retrying cannot fix (e.g., out of API credit),
        # save the results collected so far and stop instead of moving on to the next paper.
        try:
            result = process_single_paper(row, questions, configPath, input_pdf_folder)
        except NonRetryableApiError as e:
            writeCombined()
            print(f"\nStopped at paper {i+1}/{len(rows)} ({paperId}): {e}\n"
                  f"Results for {len(results)} completed paper(s) were saved to {output_json_path}; "
                  f"answers for this paper so far are in {individualDir}. "
                  f"Fix the cause and rerun with skip_existing = True to resume.")
            raise
        results.update(result)

        # Add delay after each paper except the last one
        if i < len(rows) - 1:
            delay_minutes = 0
            delay_seconds = delay_minutes * 60
            print(f"\nProcessing completed for paper {i+1}/{len(rows)}. Waiting {delay_minutes} minutes before next paper...")
            time.sleep(delay_seconds)
    
    print(f"Processing complete for {len(results)} papers.")

    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2, cls=DateTimeEncoder)
    print(f"\nScreening results saved to {output_json_path}")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_paper = total_time / len(rows) if rows else 0

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per paper: {avg_time_per_paper:.2f} seconds")



################################ Main Execution ################################

import json


if __name__ == "__main__":
    # Path to the config file
    configPath = './Config/ConfigLitScr.cfg'
    config = Config(configPath)
    print("Loaded configuration:")
    print(config.neo4j)
    print(config.dir_structure)
    print(config.general)
    print(config.retriever)
    print(config.nodeparser)

    # Extract some default parameters
    appVersion = config.general['appversion']
    huggingFaceToken = config.huggingface['api_key']

    # logging into huggingface
    # Never print the token itself; only whether one is set
    print(f"###### Hugging Face Token: {'set' if huggingFaceToken else 'not set'}")
    try:
        huggingface_hub.login(huggingFaceToken)
        print("Successfully logged into huggingface!")
    except Exception as e:
        print(f"Error logging into huggingface: {e}")

    logger = setLogging(config)

    try:
        # Optionally, set up Neo4j container if required
        # buildNeo4jContainer(config)
        
        # Initialize the ChatbotAgents (which now functions as the screening engine)
        setChunkingMethod(config)
        
        # Define the metadata CSV file path and the input folder containing the PDFs.
        # Adjust these paths as necessary.
        metadata_csv_path = config.dir_structure['metadatafile']      # CSV file with paper metadata (must include "paper_id" column)
        input_pdf_folder = config.dir_structure['input_pdf_folder']  # Folder where all paper PDFs are stored
        output_json_path = config.dir_structure['outputfile']  # File to save all screening results
        questions_file_path = config.dir_structure['questions_file_path']  # File to save all screening results

            # Define the set of questions as a key-value dictionary.
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Run the literature screening process.
        literature_screening(metadata_csv_path, questions, configPath, input_pdf_folder, output_json_path)
    
    except Exception as e:
        logger.error(f"Exception occurred:\n\n {e}", exc_info=True)