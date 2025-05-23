#################################### appVersion = 0.9.0 ######################################
# TODO: harmonize config management
# TODO: make sure the questions are articulated independently
################################ Import Required Dependencies ################################

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import os
import sys
import gc
import subprocess
import multiprocessing
import requests
import shutil
import logging
import time
import random
import json
import csv
import pandas as pd
from configparser import ConfigParser
from urllib.parse import urlparse
import docker
from tqdm import tqdm
from neo4j import GraphDatabase
import torch

import huggingface_hub
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.utils import iter_batch
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# Append the root directory to path so that the modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Modules.Readers import CustomCSVReader, CustomPDFReader
from Modules.Tools import touch, deleteFolderContents, ensureFolderExists

################################ System Message ################################
systemMessage = """
You are a literature screening assistant. You are designed to take a paper and a set of questions as input and provide answers, reasoning, and evidence from the text based on the format requested by the user.
You always follow these rules:
    Rule 1: Your main goal is to provide answers as accurately as possible, based on the instructions and context I have been given. 
    Rule 2: If a question does not match the provided context or is outside the scope of the document, do not provide answers from my past knowledge. Simply reply "Unsure".
    Rule 3: You always reply in a json format without any additional texts before or after the json.
    Rule 4: If you encounter rate limits, you will retry as many times as you need to get the answer.
    Rule 5: ATTENTION: You must always use the query engine tool to answer questions and ignore any previous conversation history.
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

def safeApiCall(call, enableLogging, *args, **kwargs):
    maxAttempts = 20
    for attempt in range(maxAttempts):
        try:
            startTime = time.time()
            result = call(*args, **kwargs)
            endTime = time.time()
            if enableLogging:
                log_str = f"API call successful. Duration: {endTime - startTime:.2f} seconds."
                logging.info(log_str)
                # print(log_str)
            return result
        except Exception as e:
            wait = 2 ** attempt + random.random()
            if enableLogging:
                log_str = f"Error occurred: {e}. Waiting for {wait:.2f} seconds."
                logging.warning(log_str)
                print(log_str)
            time.sleep(wait)
    raise Exception("API call failed after maximum number of retries")

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
class ChatbotAgents:
    """
    Integrates the language model, query engine, and related tools.
    This version supports only vector-based retrieval.
    """
    successMsg = 'Chatbot Agents updated successfully!'
    def __init__(self, configPath) -> None:
        self.availableServices = ['huggingface', 'mistral',"llama3"]
        self.configPath = configPath
        self.config = Config(configPath)
        self.getConfigs()
        self.getModelConfigs(self.configPath)
        if self.service == self.availableServices[0]:  # huggingface
            self.getHuggingFaceLlmAndEmbedding()
        elif self.service == self.availableServices[1]:  # mistral
            self.getMistralLlmAndEmbedding()
        else:
            raise ValueError(f'service must be in ({self.availableServices})')
        Settings.llm = self.llm
        Settings.embed_model = self.embedModel
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
    
    def getMistralLlmAndEmbedding(self):
        from llama_index.llms.mistralai import MistralAI
        self.llm = MistralAI(
            model=self.llmModel,
            api_key=self.apiKey,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )
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
        self.vectorAgent = ReActAgent.from_tools(
            [self.vectorQueryEngineTool],
            verbose=True,
            max_iterations=self.max_iterations,
            prefix_messages=self.customPromptTemplate.message_templates,
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
        messages = [
            {"role": "system", "content": systemMessage},
            {"role": "user", "content": queryStr},
        ]
        # The query should include instructions to output a JSON with the required keys.
        if self.enable_agent:
            chatOutput = self.selectedAgent.chat(queryStr, tool_choice="vector_rag_query_engine")
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
            
            # For HuggingFaceLLM and HuggingFaceEmbedding specifically
            if hasattr(self, 'llm') and self.llm is not None:
                # Access and clear the underlying model if possible
                if hasattr(self.llm, '_model'):
                    if hasattr(self.llm._model, 'cpu'):
                        # Move model to CPU first
                        self.llm._model.cpu()
                    del self.llm._model
                
                # Clear any tokenizer
                if hasattr(self.llm, '_tokenizer'):
                    del self.llm._tokenizer
                    
                # Delete the entire LLM instance
                del self.llm
                self.llm = None
                
            if hasattr(self, 'embedModel') and self.embedModel is not None:
                # Access and clear the underlying model
                if hasattr(self.embedModel, 'client'):
                    del self.embedModel.client
                if hasattr(self.embedModel, '_model'):
                    if hasattr(self.embedModel._model, 'cpu'):
                        # Move model to CPU first
                        self.embedModel._model.cpu()
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
def save_individual_result(paper_id, paper_data, output_dir):
    """
    Save individual paper results to a JSON file named after the paper_id.
    
    Args:
        paper_id (str): The ID/filename of the paper
        paper_data (dict): The paper's data including metadata and answers
        output_dir (str): Directory where individual results should be saved
    """
    ensureFolderExists(output_dir)
    output_path = os.path.join(output_dir, f"{paper_id}.json")
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(paper_data, outfile, indent=2, cls=DateTimeEncoder)
    print(f"Saved individual results for {paper_id} to {output_path}")


def process_single_paper(row, questions, screeningAgent):
    """
    Process a single paper using a persistent screeningAgent:
      - Assumes the agent is already initialized, PDF uploaded, and indexed.
      - Iterates over the questions and queries the indexed document.
      - Returns a dictionary mapping paper_id to its answers.
    """
    paper_id = row["pdf_filename"]
    print(f"\nProcessing paper {paper_id}...")

    paper_answers = {}
    history = []  # Optional: for simulating conversational context

    for q_key, q_text in tqdm(questions.items(), desc=f"Processing questions for paper {paper_id}", total=len(questions)):
        full_query = (
            q_text +
            """\n
            ATTENTION: When answering questions, always use the query engine tool and do not rely on your internal memory.
            Include all valid short answers in your query string when using the query engine tool.
            Answer in English, in the following JSON format:
                QuestionText: exactly repeat the question with short valid answers,
                ShortAnswer: one of the valid short answers,
                Reasoning: specify your reasoning or any additional notes for providing the answer,
                Evidence: quote the exact sentences from the paper that support your answer and reasoning. If evidence is not available, say "Not Applicable"."""
        )

        try:
            answer = safeApiCall(
                screeningAgent.chatbot,
                enableLogging=screeningAgent.config.general['enablelogging'],
                queryStr=full_query,
                history=history  # Optional: pass past interactions
            )
            print(answer)

            parsed_answer = cleanAndConvertJson(answer)
            paper_answers[q_key] = parsed_answer

            # Optional: Append to simulated conversation history
            history.append({"role": "user", "content": full_query})
            history.append({"role": "assistant", "content": str(parsed_answer)})

        except Exception as e:
            print(f"Error during QA for question {q_key}: {e}")
            paper_answers[q_key] = {"error": str(e)}

    # Construct the final result dictionary
    paper_result = {paper_id: {"metadata": row, "answers": paper_answers}}

    # Save result
    output_dir = os.path.join(screeningAgent.config.dir_structure['outputdir'], 'individual')
    save_individual_result(paper_id, paper_result[paper_id], output_dir)

    # Clean up the agent and GPU memory
    screeningAgent.cleanup()
    del screeningAgent
    gc.collect()
    torch.cuda.empty_cache()

    return paper_result



def literature_screening(metadata_file_path, questions, configPath, input_pdf_folder, output_json_path):
    """
    For each paper in the metadata file (csv or xlsx):
      - Create a persistent ChatbotAgent for that paper.
      - Upload and index the PDF once.
      - Answer all questions using the same agent to preserve context.
      - Wait between papers to manage rate limits or system resources.
    Saves individual results and a combined JSON output.
    """
    import time

    start_time = time.time()
    results = {}

    # Load metadata rows
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

    # Process each paper
    for i, row in enumerate(rows):
        try:
            # Initialize agent once per paper
            screeningAgent = ChatbotAgents(configPath=configPath)
            screeningAgent.selectDefaultAgent()
            screeningAgent.clearFiles()

            # Upload and index the PDF
            screeningAgent.uploadFiles([pdf_path])
            screeningAgent.appendIndex(
                screeningAgent.config.retriever['vectortopk'],
                screeningAgent.config.retriever['cutoffscore']
            )

            # Run QA
            result = process_single_paper(row, questions, screeningAgent)
            results.update(result)

        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            continue

        # # Optional delay between papers
        # if i < len(rows) - 1:
        #     delay_minutes = 1
        #     delay_seconds = delay_minutes * 60
        #     print(f"\nProcessing complete for paper {i+1}/{len(rows)}. Waiting {delay_minutes} minute(s)...")
        #     time.sleep(delay_seconds)

    print(f"\nProcessing complete for {len(results)} papers.")

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
    print(f"###### Hugging Face Token: {huggingFaceToken}")
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
