#################################### appVersion = 0.8.0 ######################################
# This version supports the vector index only
# TODO: harmonize config management
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
from tqdm import tqdm
from neo4j import GraphDatabase

import torch

# Import only the vector index–related modules from llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.utils import iter_batch
from llama_index.core.agent import ReActAgent

# Import vector store
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# Append the root directory to path so that the modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Modules.Readers import CustomCSVReader, CustomPDFReader
from Modules.CitationExtractor import formatReferences
from Modules.CitationExtractor import getAgentCitationsN4j as getAgentCitations
from Modules.Tools import touch, deleteFolderContents, ensureFolderExists

################################ System Message ################################
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

################################ Configurations ################################
class Config:
    """
    This class represents the configuration for the chatbot.
    It loads and converts configuration parameters from the config file.
    """
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)

        self.dict = {}
        # Load parameters from each section into class attributes
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

################################ Helper Functions ################################
def setChunkingMethod(config):
    # extract configs for node parsing/chunking
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

def buildNeo4jContainer(config):
    containerName = config.neo4j['containername']
    username = config.neo4j['username']
    password = config.neo4j['password']
    dataPath = config.dir_structure['datapath']
    pluginsPath = config.dir_structure['pluginspath']
    url1 = config.neo4j['url1']
    url2 = config.neo4j['url2']

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

    port1 = extract_port(url1)
    port2 = extract_port(url2)

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
                print(log_str)
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
    Provides file management functionalities: uploading, listing, and clearing files.
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
    
    def buildFolders(self):
        foldersToBuild = [self.inputDir, self.logDir, self.dataPath, self.pluginsPath]
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
        gr.Info("File(s) uploaded successfully!")
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
    Manages storage and retrieval of data using only the vector store.
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
        self.vectorIndex = vectorIndex
    
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
    Integrates the language model, query engine, and UI tools into a functional chatbot.
    """
    successMsg = 'Chatbot Agents updated successfully!'
    def __init__(self, configPath) -> None:
        self.availableServices = ['huggingface', 'mistral']
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
    
    def getMistralLlmAndEmbedding(self):
        from llama_index.llms.mistralai import MistralAI
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
            model_kwargs={"quantization_config": quantization_config},
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
                description="Useful for answering queries based on the uploaded document.",
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
        chatOutput = self.selectedAgent.chat(queryStr)
        references = formatReferences(getAgentCitations(chatOutput, cutoff_score=self.config.retriever['cutoffscore']))
        output = chatOutput.response + references
        print("Chatbot Answer:\n", output)
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
            self.queryEngine.index.fileManager.clear()
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

# Mock like functionality for feedback
def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: ", data.value)
    else:
        print("You downvoted this response: ", data.value)

################################ Main Execution ################################
if __name__ == "__main__":
    configPath = './Config/ConfigNew.cfg'
    config = Config(configPath)
    print(config.neo4j)          # Neo4j-related parameters
    print(config.dir_structure)  # Directory structure parameters
    print(config.general)        # General parameters
    print(config.indices)        # Indices-related parameters
    print(config.retriever)      # Retriever-related parameters
    print(config.nodeparser)     # Node parser parameters

    appVersion = config.general['appversion']
    vectorTopK = config.retriever['vectortopk']
    cutoffScore = config.retriever['cutoffscore']
    forceReindex = config.general['forcereindex']

    logger = setLogging(config)

    try:
        # Setup Neo4j container if needed
        # buildNeo4jContainer(config)
        
        # Initialize chatbot agents with only vector index support
        chatbotAgents = ChatbotAgents(configPath=configPath)
        setChunkingMethod(config)

        if forceReindex:
            chatbotAgents.queryEngine.index.clear()
            chatbotAgents.appendIndex(vectorTopK, cutoffScore)

        chatbotAgents.selectDefaultAgent()
        
        ################################ Gradio UI ################################
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
            background-color: #FF5733;
            color: white;
        }
        #reset:hover {
            background-color: #C70039;
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
                        vectorTopKSelector = gr.Slider(label='Vector Top k', minimum=1, maximum=10, value=vectorTopK, step=1)
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
                inputs=[vectorTopKSelector, cutoffScoreSelector],
                outputs=[indexStatus],
                api_name="Index Files",
                show_progress=True
            )

            submitRetrievalParams.click(
                fn=chatbotAgents.updateParams,
                inputs=[vectorTopKSelector, cutoffScoreSelector],
                api_name="Update Chatbot",
                show_progress=True
            )

        demo.queue().launch(share=True)
    
    except Exception as e:
        logger.error(f"Exception occured:\n\n {e}", exc_info=True)
