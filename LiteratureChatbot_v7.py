#################################### appVersion = 0.7.0 ######################################
# TODO: 
# harmonize config management
################################ Import Required Dependencies ################################
import os
import sys
import subprocess
import requests
import shutil
import logging
import time
import random
from configparser import ConfigParser
from urllib.parse import urlparse
import docker
import gradio as gr
# import openai
from tqdm import tqdm
from neo4j import GraphDatabase

import torch
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core.agent import ReActAgent

# from llama_index.llms.azure_openai import AzureOpenAI
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PropertyGraphIndex, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, PGRetriever, VectorContextRetriever, LLMSynonymRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor, ImplicitPathExtractor
from llama_index.core.utils import iter_batch
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPGStore as Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# append the root directory to path so that the modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Modules.Readers import CustomCSVReader, CustomPDFReader
from Modules.Retrievers import HybridRetriever
from Modules.CitationExtractor import formatReferences
from Modules.CitationExtractor import getAgentCitationsN4j as getAgentCitations
from Modules.Tools import touch, deleteFolderContents, ensureFolderExists

################################ Set Configurations ################################

class Config:
    """
    This class represents the configuration for the chatbot. It is responsible for loading and accessing the configuration parameters from the config file.
    """
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)

        self.dict = {}
        # Load parameters from each section into class attributes
        for section in self.config.sections():
            section_data = self._load_section(section)
            # Set the section as an attribute of the class
            setattr(self, section.replace('-', '_'), section_data)
            self.dict[section.replace('-', '_')] = section_data


    def _load_section(self, section):
        """
        Load a section from the config file into a dictionary.
        """
        section_data = {}
        if section in self.config:
            for key, value in self.config.items(section):
                section_data[key] = self._convert_value(value)
        return section_data

    @staticmethod
    def _convert_value(value):
        """
        Convert string values from the config to appropriate types (int, float, bool).
        """
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


systemMessage = """
        You are a Q&A assistant named Document Chatbot, built by AminK.
        You are designed to search in the uploaded documents for the most relevant information related to the user's question.
        and synthesize an informed answer. 
        You follow these rules:
        Rule 0: You ALWAYS answer in the language of the user, defaulting to ENGLISH.
        Rule 1: Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. 
        Rule 2: If the page number is available in the metadata, you report the page number for each piece of information that you provide as an inline citation with format [First Author Last Name et. al., Year of publication, page number(s)].
        Rule 3: You ALWAYS retrieve information from the query engine even if the question is repeated and you have the answer in your memory.
        Rule 4: If a question does not match the provided context or is outside the scope of the document, you do not provide answers from your past knowledge. You advise the user that the provided documents do not contain the requested information.
        """


################################ Helper Functions ################################

def getIndexType(availableIndices):
    if 'vector' in availableIndices and len(availableIndices) == 1:
        indexType = 'vector'
    elif 'graph' in availableIndices and len(availableIndices) == 1:
        indexType = 'graph'
    else:
        indexType = 'graph+vector'
    return indexType

################################ Set Chunking Method ################################
def setChunkingMethod(config):
    # extract configs
    nodeParserType = config.nodeparser['nodeparsertype']
    bufferSize = config.nodeparser['buffersize']
    breakpointPercentileThreshold = config.nodeparser['breakpointpercentilethreshold']
    Settings.chunk_size  = config.nodeparser['chunk_size']
    Settings.chunk_overlap = config.nodeparser['chunk_overlap']

    if nodeParserType == 'semantic':
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
        Settings.transformations = [
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
        ]
    else:
        raise Exception("`nodeParserType` must be either `static` or `semantic`")

#------------------------- Build Neo4j Docker Container -------------------------#

def buildNeo4jContainer(config):
    containerName=config.neo4j['containername']
    username=config.neo4j['username']
    password=config.neo4j['password']
    dataPath=config.dir_structure['datapath']
    pluginsPath=config.dir_structure['pluginspath']
    url1=config.neo4j['url1']
    url2=config.neo4j['url2']

    client = docker.from_env()

    def extract_port(url):
        parsed_url = urlparse(url)
        return parsed_url.port

    def wait_for_container(container):
        while container.status != 'running':
            print(f"Waiting for the container '{containerName}' to be running...")
            time.sleep(5)
            container.reload()

    def wait_for_neo4j():
        while True:
            try:
                response = requests.get(url2)
                if response.status_code == 200:
                    print("Neo4j is ready.")
                    break
            except requests.exceptions.ConnectionError:
                print("Waiting for Neo4j to be available...")
            time.sleep(5)

    # extract port numbers
    port1 = extract_port(url1)
    port2 = extract_port(url2)

    # Check if the container exists and is running
    try:
        container = client.containers.get(containerName)
        if container.status != 'running':
            print(f"The container '{containerName}' is not running. Starting it...")
            container.start()
            wait_for_container(container)
            wait_for_neo4j()
        else:
            print(f"The container '{containerName}' is already running.")
    
    
    except docker.errors.NotFound:
        print(f"The container '{containerName}' does not exist. Creating and starting it...")

        dataPath = os.path.abspath(dataPath)
        pluginsPath = os.path.abspath(pluginsPath)

        dockerCommand = [
            "docker", "run", "--restart", "always", "--name", containerName,
            f"--publish={port1}:{port1}", f"--publish={port2}:{port2}",
            "--env", "NEO4J_AUTH=" + username + "/" + password,
            "-e", "NEO4J_apoc_export_file_enabled=true",
            "-e", "NEO4J_apoc_import_file_enabled=true",
            "-e", "NEO4J_apoc_import_file_use__neo4j__config=true",
            "-e", "NEO4J_PLUGINS=[\"apoc\"]",
            "-v", f"{dataPath}:/data",
            "-v", f"{pluginsPath}:/plugins",
            "neo4j:latest"
        ]

        try:
            subprocess.Popen(dockerCommand)
            print(f"Started the Docker container '{containerName}'.")
        except Exception as e:
            print(f"Error occurred while starting the container: {e}")

        # Wait for the container to be created and started
        while True:
            try:
                container = client.containers.get(containerName)
                wait_for_container(container)
                wait_for_neo4j()
                break
            except docker.errors.NotFound:
                print(f"Waiting for the container '{containerName}' to be created...")
                time.sleep(5)
    
    client.close()

# function to set logging
def setLogging(config):
    enableLogging=config.general['enablelogging']
    logDir=config.dir_structure['logdir']
    logFile=config.dir_structure['logfile']
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

# Function to handle rate limits with exponential backoff
def safeApiCall(call, enableLogging, *args, **kwargs):
    maxAttempts = 10
    for attempt in range(maxAttempts):
        try:
            startTime = time.time()
            result = call(*args, **kwargs)
            endTime = time.time()
            if enableLogging:
                logging.info(f"API call successful. Duration: {endTime - startTime:.2f} seconds.")
            return result
        except Exception as e:
            wait = 2 ** attempt + random.random()
            if enableLogging:
                logging.warning(f"Rate limit hit. Waiting for {wait:.2f} seconds.")
            time.sleep(wait)
    raise Exception("API call failed after maximum number of retries")


class FileManagement:
    """
    This class provide the required file management functionalities.
    """
    def __init__(self,config) -> None:
        """
        Initializes the `FileManagement` object with the provided configuration.
        """
        self.config = config
        self.getConfigs()
        self.buildFolders()

    def getConfigs(self):
        """
        Retrieves the necessary configurations for file management from the global configuration.
        """
        self.inputDir = self.config.dir_structure['inputdir']
        self.logDir = self.config.dir_structure['logdir']
        self.dataPath = self.config.dir_structure['datapath']
        self.pluginsPath = self.config.dir_structure['pluginspath']
    
    def buildFolders(self):
        """
        Creates the necessary folders for file management.
        """
        foldersToBuild=[self.inputDir,self.logDir,self.dataPath,self.pluginsPath]
        for folder in foldersToBuild:
            ensureFolderExists(folder)
        
    def append(self,files):
        """
        Copies the provided files to the input directory and returns a message indicating the successful upload.
        """
        message = ''
        self.fileNames=[]
        for file in files:
            fileName = os.path.basename(file)
            targetFileName = os.path.join(self.inputDir, fileName)
            shutil.copyfile(file, targetFileName)
            message += str(fileName) + '\n'
            self.fileNames.append(fileName)
        
        gr.Info("File(s) uploaded successfully!")
        message += str(len(files)) + " files were uploaded!\n"
        
        return message

    def upload(self,files):
        """
        Clears the existing files and uploads the provided files.
        """
        self.clear()
        return self.append(files)

    def clear(self):
        """
        Deletes all the files in the input directory.
        """
        deleteFolderContents(self.inputDir)
        self.fileNames = []

    def ls(self):
        """
        Returns a list of the names of the uploaded files.
        """
        return self.fileNames
    
    def createDummyFile(self):
        """
        Creates a dummy file in the input directory. This is to avoid failures when creating index from an empty directory.
        """
        touch(os.path.join(self.inputDir, 'dummy.txt'))



class Storage:
    """
    This class is responsible for managing the storage and retrieval of data used by the Literature Chatbot.
    """
    def __init__(self,config) -> None:
        """
        The constructor method initializes the object and sets the configuration parameters. It takes a `config` object as a parameter. Inside the constructor, it initializes the Neo4jPropertyGraphStore and Neo4jVectorStore objects for storing property graph data and vector data respectively. It also sets up the storage context for both graph and vector data.
        """
        self.config = config
        self.getConfigs()
        self.graphStore = Neo4jPropertyGraphStore(
            username=self.username,
            password=self.password,
            url=self.url,
        )
        self.storageContextGraph = StorageContext.from_defaults(
            graph_store=self.graphStore,
        )
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
        """
        This method retrieves the configuration parameters from the `config` object and assigns them to instance variables. These parameters include the Neo4j database URL, username, password, and the embedding dimension.
        """
        self.url = self.config.neo4j['url1']
        self.username = self.config.neo4j['username']
        self.password = self.config.neo4j['password']
        service = self.config.llm_model['service']
        self.embedDim = self.config.dict[service]['embeddim']

    def clear(self):
        """
        This method clears the Neo4j graph database by deleting all nodes and relationships. It uses the provided URL, username, and password to connect to the database and execute a Cypher query to delete all nodes and relationships.
        """
        try:
            driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            driver.close()
            return "Database cleared successfully!"
        except Exception as e:
            print(f"Error occurred while clearing the database: {e}")
            return f"Clearing the database failed with error: {e}"


class Index:
    """
    This class is responsible for building and managing the indices used for searching and retrieving information from the uploaded documents.
    """
    def __init__(self,llm,config) -> None:
        """
        The constructor method initializes the attributes of the class, such as the LLMSynonymRetriever object, configuration object, storage, vectorIndex, pgIndex, kgExtractors, and fileManager.
        """
        self.llm = llm
        self.config = config
        self.getConfigs()
        self.storage = Storage(config)
        self.vectorIndex = VectorStoreIndex.from_vector_store(vector_store=self.storage.vectorStore)
        self.pgIndex = PropertyGraphIndex.from_existing(
            property_graph_store=self.storage.graphStore,
            vector_store=self.storage.vectorStore,
            embed_kg_nodes=True,
        )
        self.kgExtractors = [
            SimpleLLMPathExtractor(
                llm=self.llm,
                max_paths_per_chunk=self.maxPathsPerChunk,
                num_workers=4,
            ),
            ImplicitPathExtractor()
        ]
        self.fileManager = FileManagement(self.config)
    
    def getConfigs(self):
        """
        This method retrieves the configuration values from the config object and assigns them to the corresponding attributes of the class.
        """
        self.maxPathsPerChunk = self.config.indices['maxpathsperchunk']
        self.showProgress = self.config.general['showprogress']
        self.allIndices = self.config.indices['allindices']
        self.enableLogging = self.config.general['enablelogging']
        self.max_new_tokens = self.config.llm_model['max_new_tokens']


    def _buildVector(self, documents, batchSize=4):
        """
        This method builds the vector index using the provided documents. It iterates over the documents in batches and calls the `from_documents` method of the VectorStoreIndex class to build the vector index.
        """
        for batch in tqdm(iter_batch(documents, batchSize), 
                        total=len(documents)//batchSize, 
                        desc='Build VectorIndex for node batches'):
            startBatchTime = time.time()
            
            vectorIndex = safeApiCall(
                call=VectorStoreIndex.from_documents,
                enableLogging=self.enableLogging,
                documents=batch,
                storage_context=self.storage.storageContextVector,
                show_progress=self.showProgress
            )
            endBatchTime = time.time()
            logMsg = f"Batch processed. Duration: {endBatchTime - startBatchTime:.2f} seconds."
            if self.enableLogging:
                logging.info(logMsg)
            else:
                print(logMsg)
        self.vectorIndex=vectorIndex
    
    def _buildPg(self, documents, batchSize=4):
        """
        This method builds the property graph index using the provided documents. It calls the `from_documents` method of the PropertyGraphIndex class to build the property graph index.
        """
        for batch in tqdm(iter_batch(documents, batchSize), 
                        total=len(documents)//batchSize, 
                        desc='Build PgIndex for node batches'):
            startBatchTime = time.time()
            pgIndex = safeApiCall(
                call=PropertyGraphIndex.from_documents,
                enableLogging=self.enableLogging,
                documents=batch,
                kg_extractors=self.kgExtractors,
                property_graph_store=self.storage.graphStore,
                storage_context=self.storage.storageContextGraph,
                show_progress=self.showProgress
            )
            
            endBatchTime = time.time()
            logMsg = f"Batch processed. Duration: {endBatchTime - startBatchTime:.2f} seconds."
            if self.enableLogging:
                logging.info(logMsg)
            else:
                print(logMsg)
        self.pgIndex=pgIndex

    def _build(self, documents, batchSize=4):
        """
        his helper method calls both `_buildVector` and `_buildPg` methods to build both the vector and property graph indices.
        """
        self._buildVector(documents,batchSize)
        self._buildPg(documents,batchSize)

    def append(self,inputDocsDir,batchSize=1,indexType='graph+vector'):
        """
        This method adds new documents to the existing indices. It loads the documents from the input directory using the SimpleDirectoryReader class and calls the appropriate method based on the index type to build the corresponding index.
        """
        # self.llm.max_new_tokens = 100
        documents = SimpleDirectoryReader(
            inputDocsDir,
            recursive=True,
            filename_as_id=True,
            file_extractor={
                '.csv': CustomCSVReader(),
                '.pdf': CustomPDFReader(llm=self.llm),
            },
        ).load_data()
        # self.llm.max_new_tokens = self.max_new_tokens
        if indexType == self.allIndices[0]:
            self._buildVector(documents,batchSize)
        elif indexType == self.allIndices[1]:
            self._buildPg(documents,batchSize)
        elif indexType == self.allIndices[2]:
            self._build(documents, batchSize)
        else:
            supportedTypes = ','.join(self.allIndices)
            raise Exception(f'indexType must be in ({supportedTypes})')
    
    def get(self, indexType='graph+vector'):
        """
        This method retrieves the built indices. It takes the index type as a parameter and returns the corresponding index object(s) based on the index type.
        """
        if indexType == self.allIndices[0]:
            return self.vectorIndex
        elif indexType == self.allIndices[1]:
            return self.pgIndex
        elif indexType == self.allIndices[2]:
            return self.vectorIndex, self.pgIndex
        else:
            supportedTypes = ','.join(self.allIndices)
            raise Exception(f'indexType must be in ({supportedTypes})')
        

class QueryEngine:
    """
    This class is responsible for building and managing the query engines used by the Literature Chatbot.
    """
    LLMSynonymRetrieverPrompt = (
        "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
        "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
        "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
        "Note, result should be in one line, separated by '^' symbols."
        "----\n"
        "QUERY: {query_str}\n"
        "----\n"
        "KEYWORDS: "
    )

    
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
    print("########## textQaTemplate:",textQaTemplate)

    successMsg = 'Query Engine was built successfully!'

    def __init__(self,llm,embedModel,config) -> None:
        """
        The constructor method initializes the object and sets the configuration parameters. It takes three parameters: `llm`, which is the language model used for generating responses, `embedModel`, which is the embedding model used for calculating text embeddings, and `config`, which is the configuration object that contains various settings for the query engine.
        """
        self.config = config
        self.getConfigs()
        self.index=Index(llm,self.config)
        self.embedModel=embedModel

    def getConfigs(self):
        """
        This method retrieves the configuration parameters from the `config` object and assigns them to instance variables. These parameters include the input directory, the top-k value for vector retrieval, the top-k value for graph context retrieval, the maximum number of synonyms, and the path depth. These parameters are used in the building of the query engines.
        """
        self.inputDir=self.config.dir_structure['inputdir']
        self.vectorTopK=self.config.retriever['vectortopk']
        self.contextTopK=self.config.retriever['contexttopk']
        self.maxSynonyms=self.config.retriever['maxsynonyms']
        self.pathDepth=self.config.retriever['pathdepth']
        self.allIndices = self.config.indices['allindices']

    def _buildVectorQueryEngine(self, forceReindex=False):
        """
        This method builds a query engine using a vector index. It creates a `VectorIndexRetriever` and a `RetrieverQueryEngine` based on the vector index. If `forceReindex` is set to True, it appends the input documents to the vector index before building the query engine.
        """
        if forceReindex:
            self.index.append(inputDocsDir=self.inputDir,indexType='vector')
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
    
    def _buildPgQueryEngine(self, forceReindex=False):
        """
        This method builds a query engine using a property graph index. It creates a `VectorContextRetriever` and an `LLMSynonymRetriever` as sub-retrievers, and a `PGRetriever` based on these sub-retrievers. It also creates a `RetrieverQueryEngine` based on the `PGRetriever`. If `forceReindex` is set to True, it appends the input documents to the property graph index before building the query engine.
        """
        if forceReindex:
            self.index.append(inputDocsDir=self.inputDir,indexType='graph')
        subRetrievers = [
            VectorContextRetriever(
                self.index.pgIndex.property_graph_store,
                include_text=True,
                embed_model=self.embedModel,
                similarity_top_k=self.contextTopK,
                path_depth=self.pathDepth
            ),
            LLMSynonymRetriever(
                self.index.pgIndex.property_graph_store,
                include_text=True,
                synonym_prompt=self.LLMSynonymRetrieverPrompt,
                max_keywords=self.maxSynonyms,
                path_depth=self.pathDepth,
            ),
        ]
        self.graphRetriever = PGRetriever(
            sub_retrievers=subRetrievers, 
            use_async=False
        )
        self.graphQueryEngine = RetrieverQueryEngine.from_args(
            self.graphRetriever,
            text_qa_template=self.textQaTemplate,
        )
        print("Graph Query Engine was built successfully!")
        return self.successMsg
    
    def _buildHybridQueryEngine(self):
        """
        This method builds a query engine using a property graph index. It creates a `VectorContextRetriever` and an `LLMSynonymRetriever` as sub-retrievers, and a `PGRetriever` based on these sub-retrievers. It also creates a `RetrieverQueryEngine` based on the `PGRetriever`. If `forceReindex` is set to True, it appends the input documents to the property graph index before building the query engine.
        """
        self.graphVecRetriever = HybridRetriever(
            self.vectorRetriever, 
            self.graphRetriever, 
            mode='OR'
        )
        self.graphVecQueryEngine = RetrieverQueryEngine.from_args(
            self.graphVecRetriever,
            text_qa_template=self.textQaTemplate,
        )
        print("Hybrid Query Engine was built successfully!")
        return self.successMsg

    def _buildAll(self, forceReindex=False):
        """
        This method builds all three types of query engines: vector, property graph, and hybrid. It calls the `_buildVectorQueryEngine`, `_buildPgQueryEngine`, and `_buildHybridQueryEngine` methods.
        """
        _ = self._buildVectorQueryEngine(forceReindex=forceReindex)
        _ = self._buildPgQueryEngine(forceReindex=forceReindex)
        rm = self._buildHybridQueryEngine()
        print(rm)
        return self.successMsg

    def get(self,indexType = 'vector'):
        """
        This method returns the specified query engine based on the provided index type. It can return the vector query engine, the property graph query engine, or the hybrid query engine. If the index type is not supported, an exception is raised.
        """
        if indexType == self.allIndices[0]:
            rm = self._buildVectorQueryEngine()
            return self.vectorQueryEngine, rm
        elif indexType == self.allIndices[1]:
            rm = self._buildPgQueryEngine()
            return self.graphQueryEngine, rm
        elif indexType == self.allIndices[2]:
            rm = self._buildHybridQueryEngine()
            return self.graphVecQueryEngine, rm
        else:
            supportedTypes = ','.join(self.allIndices)
            raise Exception(f'indexType must be in ({supportedTypes})')
        
    def getAll(self):
        """
        This method builds all three types of query engines and returns them.
        """
        rm = self._buildAll()
        return self.vectorQueryEngine,self.graphQueryEngine,self.graphVecQueryEngine, rm

class ChatbotAgents:
    successMsg = 'Chatbot Agents updated successfully!'
    def __init__(self,configPath) -> None:
        """
        Initializes the `ChatbotAgents` object by loading the configuration file and setting up the chatbot agents.
        """
        self.availableServices = ['huggingface','mistral']
        self.configPath = configPath
        self.config = Config(configPath)

        self.getConfigs()

        self.getModelConfigs(self.configPath)

        if self.service == self.availableServices[0]:
            self.getHuggingFaceLlmAndEmbedding()
        elif self.service == self.availableServices[1]:
            self.getMistralLlmAndEmbedding()
        else:
            raise ValueError(f'service must be in ({self.availableServices})')
        
        Settings.llm = self.llm
        Settings.embed_model = self.embedModel

        self.updateQueryEngine()
        # select the agent with the first available index type by default 
        self.selectDefaultAgent()

    def getMistralLlmAndEmbedding(self):
        from llama_index.llms.mistralai import MistralAI
        from llama_index.embeddings.mistralai import MistralAIEmbedding
                
        self.llm = MistralAI(
            model=self.llmModel,
            api_key=self.apiKey,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )
        self.embedModel = MistralAIEmbedding(
            model=self.embeddingModelName,
            api_key=self.apiKey
        )

    def getModelConfigs(self,configPath):
        """
        retrieves the model configurations for the chatbot agents from the global configuration.
        """
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

    def getHuggingFaceLlmAndEmbedding(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from transformers import BitsAndBytesConfig

        system_prompt = "<|SYSTEM|>\n" + systemMessage + "\n"
        query_wrapper_prompt = PromptTemplate(
            "<|USER|>\n{query_str}\n<|ASSISTANT|>\n"
        )

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
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.llmModel,
            model_name=self.llmModel,
            device_map="auto",
            model_kwargs={
                "quantization_config":quantization_config
                },
        )

        self.embedModel = HuggingFaceEmbedding(
            model_name=self.embeddingModelName, max_length=self.embedDim

        )
        

    def getConfigs(self):
        """
        retrieves the necessary configurations for the chatbot agents from the global configuration.
        """
        self.inputDir = self.config.dir_structure['inputdir']
        self.allIndices = self.config.indices['allindices']
        self.availableIndices = self.config.indices['availableindices']
        self.cutoffScore = self.config.retriever['cutoffscore']

    def updateQueryEngine(self):
        self.queryEngine = QueryEngine(self.llm,self.embedModel,self.config)
        self.queryEngine._buildAll()
        self._buildAll()
    
    def _buildAll(self):
        """
        builds the necessary components of the chatbot. It creates the agents and tools required for the chatbot to function.
        """
        customPromptMessages = [
            ChatMessage(role="system", content=systemMessage),
        ]
        self.customPromptTemplate = ChatPromptTemplate(customPromptMessages)

        self.vectorQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.vectorQueryEngine,
            metadata=ToolMetadata(
                name="vector_rag_query_engine",
                description="Useful for answering all queries based on the uploaded document. Do not use this tool to answer trivial questions or prompts.",
            ),
        )
        self.vectorAgent = ReActAgent.from_tools(
            [self.vectorQueryEngineTool],
            verbose=True,
            max_iterations = self.max_iterations,
            prefix_messages=self.customPromptTemplate.message_templates,
        )

        graphQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.graphQueryEngine,
            metadata=ToolMetadata(
                name="graph_rag_query_engine",
                description="Useful for answering all queries based on the uploaded document. Do not use this tool to answer trivial questions or prompts.",
            ),
        )
        self.graphAgent = ReActAgent.from_tools(
            [graphQueryEngineTool],
            verbose=True,
            max_iterations = self.max_iterations,
            prefix_messages=self.customPromptTemplate.message_templates,
        )

        graphVectorQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.graphVecQueryEngine,
            metadata=ToolMetadata(
                name="graph_vector_rag_query_engine",
                description="Useful for answering all queries based on the uploaded document. Do not use this tool to answer trivial questions or prompts.",
            ),
        )
        self.graphVectorAgent = ReActAgent.from_tools(
            [graphVectorQueryEngineTool],
            verbose=True,
            max_iterations = self.max_iterations,
            prefix_messages=self.customPromptTemplate.message_templates,
        )
    
    def appendIndex(self,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore):
        """
        Appends the index of the input documents to the query engine based on the provided parameters. It also rebuilds the query engine and the chatbot agents.
        """
        self.updateParams(vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore)
        indexType=getIndexType(self.availableIndices)
        if indexType in [self.allIndices[0],self.allIndices[2]]:
            self.queryEngine._buildVectorQueryEngine(forceReindex=True)
        if indexType in [self.allIndices[1],self.allIndices[2]]:
            self.queryEngine._buildPgQueryEngine(forceReindex=True)
        if indexType == self.allIndices[2]:
            self.queryEngine._buildHybridQueryEngine()
        # self.queryEngine._buildAll(forceReindex=True)
        self._buildAll()
        return self.successMsg

    def getDefault(self):
        """
        Creates a dummy file and returns the default chatbot agent.
        """
        self.queryEngine.index.fileManager.createDummyFile()
        self.vectorAgent,_,_= self.getAll(forceReindex=False)
        return self.vectorAgent

    def getAll(self, forceReindex=False):
        """
        Returns all the chatbot agents. If `forceReindex` is set to `True`, it appends the index and rebuilds the agents before returning them.
        """
        if forceReindex:
            self.appendIndex()
        return self.vectorAgent, self.graphAgent, self.graphVectorAgent, self.successMsg
    
    def updateParams(self,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore):
        """
        Updates the parameters of the chatbot agents based on the provided values. It also updates the query engine.
        """
        self.config.retriever['vectortopk']=vectorTopK
        self.config.retriever['contexttopk']=contextTopK
        self.config.retriever['maxsynonyms']=maxSynonyms
        self.config.retriever['pathdepth']=pathDepth
        self.config.retriever['cutoffscore']=cutoffScore
        self.updateQueryEngine()
    
    def chatbot(self, queryStr, history):
        """
         Takes a user query and a history of previous interactions as input and returns the chatbot's response. It uses the selected chatbot agent to generate the response. Passing the history is not needed since that is captured by the agent's memory. However, the history argument is still needed to be compatible with the Gradio interface component.
        """
        chatOutput = self.selectedAgent.chat(queryStr)
        references = formatReferences(getAgentCitations(chatOutput, cutoff_score=self.cutoffScore))
        output = chatOutput.response + references
        print("Chatbot Answer:\n",output)
        return output
    
    def uploadFiles(self,files):
        """
        Uploads the input files to the query engine's file manager.
        """
        try:
            msg = self.queryEngine.index.fileManager.upload(files)
            print(msg)
            return msg
        except:
            return "Uploading files failed!"


    def clearFiles(self):
        """
        Clears the input files and the query engine's storage.
        """
        try:
            self.queryEngine.index.storage.clear()
            self.queryEngine.index.fileManager.clear()
            return "Database and input directory were cleared successfully!"
        except:
            return "Clearing the database and input directory failed!"

    def getFilesList(self):
        """
        Returns a list of the uploaded input files.
        """
        return self.queryEngine.index.fileManager.ls()
    
    def selectAgent(self,indexType):
        """
        Selects the chatbot agent based on the provided index type.
        """
        if indexType == 'vector':
            self.selectedAgent = self.vectorAgent
        if indexType == 'graph':
            self.selectedAgent = self.graphAgent
        if indexType == 'graph+vector':
            self.selectedAgent = self.graphVectorAgent
    
    def selectDefaultAgent(self):
        """
        Selects the default chatbot agent.
        """
        self.selectAgent(self.availableIndices[0])

    def getSelectedAgent(self):
        """
        Returns the currently selected chatbot agent.
        """
        return self.selectedAgent

# This is a mock like functionality, the data is not stored
def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: ", data.value)
    else:
        print("You downvoted this response: ", data.value)

if __name__ == "__main__":
    
    configPath = './Config/ConfigNew.cfg'
    config = Config(configPath)

    # Accessing loaded configuration
    print(config.neo4j)          # Neo4j-related parameters
    print(config.dir_structure)  # Directory structure parameters
    print(config.general)        # General parameters
    print(config.indices)        # Indices-related parameters
    print(config.retriever)      # Retriever-related parameters
    print(config.nodeparser)     # Node parser parameters

    # extract the default parameters, these can be changes via the UI
    appVersion=config.general['appversion']
    vectorTopK=config.retriever['vectortopk']
    contextTopK=config.retriever['contexttopk']
    maxSynonyms=config.retriever['maxsynonyms']
    pathDepth=config.retriever['pathdepth']
    cutoffScore=config.retriever['cutoffscore']
    forceReindex=config.general['forcereindex']
    availableIndices = config.indices['availableindices']
    defaultIndex=availableIndices[0]

    logger = setLogging(config)

    try:
        # Setup Neo4j
        # buildNeo4jContainer(config)
        
        # Initialize chatbot agents
        chatbotAgents = ChatbotAgents(configPath=configPath)

        setChunkingMethod(config)

        if forceReindex:
            chatbotAgents.appendIndex(vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore)
        
        chatbotAgents.selectAgent(defaultIndex)
        


        ################################ Gradio UI ################################
        # The Gradio user interface consists of several components:

        # 1. Menu: This column contains various options for interacting with the chatbot. It includes buttons for resetting the database, uploading files, and indexing files. There is also an accordion menu for adjusting retrieval parameters such as the index type, top k values, and relevance score cutoff.

        # 2. Chatbot: This column displays the chatbot interface where users can ask questions and receive answers. The chatbot is powered by the `chatbotAgents.chatbot` object and includes a like/dislike feature for user feedback.

        # 3. Reset Button: This button is used to clear the database and input directory. When clicked, it triggers the `chatbotAgents.clearFiles` function and displays the status of the reset process.

        # 4. Upload Button: This button allows users to upload files for the chatbot to process. It supports multiple file uploads and accepts files with the extensions .pdf and .csv. When files are uploaded, the `chatbotAgents.uploadFiles` function is called and the status of the upload process is displayed.

        # 5. Index Button: This button triggers the `chatbotAgents.appendIndex` function, which indexes the uploaded files. It takes into account the retrieval parameters set in the accordion menu. The status of the indexing process is displayed.

        # 6. Retrieval Parameters Accordion: This accordion menu contains options for adjusting the retrieval parameters used by the chatbot. Users can select the index type (vector, graph, or graph+vector), set the top k values for vector and graph retrieval, specify the maximum number of synonyms to search, set the graph path depth, and define the minimum relevance score for references. The "Update Chatbot" button applies the changes.

        # Overall, the Gradio user interface provides a user-friendly way to interact with the literature chatbot and perform various actions such as resetting the database, uploading files, indexing files, and adjusting retrieval parameters.

        # CSS style to extend the size of the chatbot box to fit the height of the screen and aligning the menu to the bottom
        css = """
        #chatbot {
            flex-grow: 1 !important;
            overflow: auto !important;
        }
        #col {
            height: calc(100vh - 112px - 16px) !important;
        }
        #menu {
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            height: calc(100vh - 112px - 16px) !important;
        }
        #reset {
            background-color: #FF5733; /* Orange background */
            color: white; /* White text */
        }

        #reset:hover {
            background-color: #C70039; /* Darker red when hovered */
        }
        """
        with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
            with gr.Row():
                with gr.Column(scale=3, elem_id="menu"):
                    resetButton = gr.Button("Reset the Database", render=True, elem_id="reset")
                    resetStatus = gr.Markdown("", line_breaks=True)
                    uploadButton = gr.UploadButton("Upload Files", file_count="multiple", file_types=['.pdf', '.csv'])
                    uploadStatus = gr.Markdown("", line_breaks=True)
                    indexButton = gr.Button(value="Ingest Files")
                    indexStatus = gr.Markdown("", line_breaks=True)

                    with gr.Accordion("Retrieval Parameters", open=False):
                        indexSelector = gr.Dropdown(
                            choices=availableIndices,
                            value=defaultIndex,
                            label='Retriever Index Type'
                        )
                        vectorTopKSelector = gr.Slider(label='Vector Top k', minimum=1, maximum=10, value=vectorTopK, step=1)
                        graphTopKSelector = gr.Slider(label='Graph Top k', minimum=1, maximum=10, value=contextTopK, step=1)
                        maxSynonymsSelector = gr.Slider(label='Max Synonyms to Search', minimum=1, maximum=10, value=maxSynonyms, step=1)
                        pathDepthSelector = gr.Slider(label='Graph Path Depth', minimum=1, maximum=5, value=pathDepth, step=1)
                        cutoffScoreSelector = gr.Slider(label='References Min Relevance Score', minimum=0, maximum=1, value=cutoffScore, step=0.01)
                        submitRetrievalParams = gr.Button(value='Update Chatbot')

                with gr.Column(scale=7, elem_id="col"):
                    gr.Markdown(f'<h1 style="text-align: center;">Chat with Your Documents v{appVersion}</h1>')
                    chatbot = gr.Chatbot(elem_id="chatbot", render=False)
                    chatbot.like(vote, None, None)
                    gr.ChatInterface(
                        chatbotAgents.chatbot,
                        chatbot=chatbot,
                        fill_height=True
                    )

            resetButton.click(
                fn=chatbotAgents.clearFiles,
                outputs=resetStatus,
                api_name="Reset Files",
                show_progress=True,
            )

            uploadButton.upload(
                fn=chatbotAgents.uploadFiles,
                inputs=uploadButton,
                outputs=uploadStatus,
                show_progress="full",
            )

            indexButton.click(
                fn=chatbotAgents.appendIndex,
                inputs=[
                    vectorTopKSelector,
                    graphTopKSelector,
                    maxSynonymsSelector,
                    pathDepthSelector,
                    cutoffScoreSelector
                ],
                outputs=[indexStatus],
                api_name="Index Files",
                show_progress=True
            )

            submitRetrievalParams.click(
                fn=chatbotAgents.updateParams,
                inputs=[
                    vectorTopKSelector,
                    graphTopKSelector,
                    maxSynonymsSelector,
                    pathDepthSelector,
                    cutoffScoreSelector
                ],
                api_name="Update Chatbot1",
                show_progress=True
            )

            submitRetrievalParams.click(
                chatbotAgents.selectAgent,
                inputs=[indexSelector],
                api_name="Update Chatbot2",
                show_progress=True
            )

        demo.queue().launch(share=True)
    
    except Exception as e:
        logger.error(f"Exception occured:\n\n {e}", exc_info=True)