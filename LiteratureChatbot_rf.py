# TODO:
# Add logging
# Error Handling
#     

################################ Import Required Dependencies ################################
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import subprocess
import shutil
import logging
import time
import random
from configparser import ConfigParser

import docker
import gradio as gr
import openai
from tqdm import tqdm
from neo4j import GraphDatabase

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, PGRetriever, VectorContextRetriever, LLMSynonymRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor, ImplicitPathExtractor
from llama_index.core.utils import iter_batch
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPGStore as Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from Modules.AI.Readers import CustomCSVReader, CustomPDFReader
from Modules.AI.Tools import touch, deleteFolderContents, ensureFolderExists
from Modules.AI.Retrievers import HybridRetriever
from Modules.AI.CitationExtractor import formatReferences
from Modules.AI.CitationExtractor import getAgentCitationsN4j as getAgentCitations
from Modules.AI.NodeParser import CustomSemanticSplitterNodeParser

################################ Set Configurations ################################

appVersion = '0.3.0'

# Directories
inputDir = './input/selected'
logDir = './logs/'
logFile = "processing_log.txt"
dataPath = "./neo4j_vol1/data"
pluginsPath = "./neo4j_vol1/plugins"
configDir = './Config/'
configFile = 'Config.cfg'
configPath = os.path.join(configDir, configFile)

# Neo4j config
username = "neo4j"
password = "neo4j_rag_poc"
url = "bolt://localhost:7687"
embedDim = 1536
containerName = "neo4j-apoc"

################################ Set Parameters ################################

# Parameters
forceReindex = False  # whether to index documents in `inputDir` before launching the chatbot UI
enableLogging = False  # whether to log INFO and WARNING messages
showProgress = True  # whether to show low level progress bars, if set to false, only high level progress bar will be rendered

# Node Parser parameters
nodeParserType = 'semantic'  # node parser approach, can be either `static` or `semantic`
batchSize = 1  # number of documents to process at a time, larger values may lead to hitting the embedding API rate limit
Settings.chunk_size = 512  # Number of tokens to include in each chunk
Settings.chunk_overlap = 50  # The overlap between subsequent chunks

# Semantic Splitter parameters
bufferSize = 1  # Default = 1, Adjust buffer size as needed
breakpointPercentileThreshold = 75  # Default = 95, lower values result in too many small chunks

# PropertyGraphIndex parameters
maxPathsPerChunk = 10

# Default retriever parameters - These can be adjusted in the UI
vectorTopK = 10
contextTopK = 2
maxSynonyms = 10
pathDepth = 1
cutoffScore = 0.9

systemMessage = """
        I am a Q&A assistant named Literature Chatbot, created by DSAT AI Lab at PMRA to assist scientist with their literature review tasks. 
        I am designed to search in the uploaded documents for the most relevant information related to the user's question.
        and synthesize an informed answer. 
        I follow these rules:
        Rule 1: My main goal is to provide answers as accurately as possible, based on the instructions and context I have been given. 
        Rule 2: If the page number is available in the metadata, I report the page number for each piece of information that I provide as an inline citation with format [First Author Last Name et. al., Year of publication, page number(s)].
        Rule 3: I ALWAYS retrieve information from the query engine even if the question is repeated and I have the answer in my memory.
        Rule 4: If a question does not match the provided context or is outside the scope of the document, I do not provide answers from my past knowledge. I advise the user that the provided documents do not contain the requested information.
        """


################################ Helper Functions ################################

#------------------------- Build Neo4j Docker Container -------------------------#

def buildNeo4jContainer():
    
    client = docker.from_env()

    # Check if the container exists and is running
    try:
        container = client.containers.get(containerName)
        if container.status != 'running':
            print(f"The container '{containerName}' is not running. Starting it...")
            container.start()
        else:
            print(f"The container '{containerName}' is already running.")
    except docker.errors.NotFound:
        print(f"The container '{containerName}' does not exist. Creating and starting it...")

        dataPath = os.path.abspath(dataPath)
        pluginsPath = os.path.abspath(pluginsPath)

        dockerCommand = [
            "docker", "run", "--restart", "always", "--name", containerName,
            "--publish=7474:7474", "--publish=7687:7687",
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

    client.close()

#----------------------- Ensure Folder Structure Exists -----------------------#

def setLogging(enableLogging):
    if enableLogging:
        logPath = os.path.join(logDir, "processing_log.txt")
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(logPath),
                                logging.StreamHandler()
                            ])
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

# Function to handle rate limits with exponential backoff
def safeApiCall(call, *args, **kwargs):
    maxAttempts = 10
    for attempt in range(maxAttempts):
        try:
            startTime = time.time()
            result = call(*args, **kwargs)
            endTime = time.time()
            if enableLogging:
                logging.info(f"API call successful. Duration: {endTime - startTime:.2f} seconds.")
            return result
        except openai.RateLimitError as e:
            wait = 2 ** attempt + random.random()
            if enableLogging:
                logging.warning(f"Rate limit hit. Waiting for {wait:.2f} seconds.")
            time.sleep(wait)
    raise Exception("API call failed after maximum number of retries")


class fileManagement:
    def __init__(self,inputDir,logDir,dataPath,pluginsPath) -> None:
        self.inputDir=inputDir
        self.logDir=logDir
        self.dataPath=dataPath
        self.pluginsPath=pluginsPath
        self.buildFolders()
    
    def buildFolders(self):
        foldersToBuild=[self.inputDir,self.logDir,self.dataPath,self.pluginsPath]
        for folder in foldersToBuild:
            ensureFolderExists(folder)
        
    def append(self,files):
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
        self.clear()
        return self.append(files)

    def clear(self):
        deleteFolderContents(self.inputDir)
        self.fileNames = []

    def ls(self):
        return self.fileNames
    
    def createDummyFile(self):
        touch(os.path.join(self.inputDir, 'dummy.txt'))



class storage:
    def __init__(self,url,username,password) -> None:
        self.url=url
        self.username=username
        self.password=password
        self.graphStore = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url,
        )
        self.storageContextGraph = StorageContext.from_defaults(
            graph_store=self.graphStore,
        )
        self.vectorStore = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            embedding_dimension=embedDim,
        )

        self.storageContextVector = StorageContext.from_defaults(
            vector_store=self.vectorStore,
        )
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


class Index:
    def __init__(self,llm) -> None:
        self.llm = llm
        self.storage = storage(url,username,password)
        self.vectorIndex = VectorStoreIndex.from_vector_store(vector_store=self.storage.vectorStore)
        self.pgIndex = PropertyGraphIndex.from_existing(
            property_graph_store=self.storage.graphStore,
            vector_store=self.storage.vectorStore,
            embed_kg_nodes=True,
        )
        self.kgExtractors = [
            SimpleLLMPathExtractor(
                llm=self.llm,
                max_paths_per_chunk=maxPathsPerChunk,
                num_workers=4,
            ),
            ImplicitPathExtractor()
        ]
        self.fileManager = fileManagement(inputDir=inputDir,logDir=logDir,dataPath=dataPath,pluginsPath=pluginsPath)

    def _buildVector(self, documents, batchSize=4):
        for batch in tqdm(iter_batch(documents, batchSize), 
                        total=len(documents)//batchSize, 
                        desc='Build VectorIndex for node batches'):
            startBatchTime = time.time()
            
            vectorIndex = safeApiCall(
                VectorStoreIndex.from_documents,
                documents=batch,
                storage_context=self.storage.storageContextVector,
                show_progress=showProgress
            )
            endBatchTime = time.time()
            logMsg = f"Batch processed. Duration: {endBatchTime - startBatchTime:.2f} seconds."
            if enableLogging:
                logging.info(logMsg)
            else:
                print(logMsg)
        self.vectorIndex=vectorIndex
    
    def _buildPg(self, documents, batchSize=4):
        for batch in tqdm(iter_batch(documents, batchSize), 
                        total=len(documents)//batchSize, 
                        desc='Build PgIndex for node batches'):
            startBatchTime = time.time()
            pgIndex = safeApiCall(
                PropertyGraphIndex.from_documents,
                batch,
                kg_extractors=self.kgExtractors,
                property_graph_store=self.storage.graphStore,
                storage_context=self.storage.storageContextGraph,
                show_progress=showProgress
            )
            
            endBatchTime = time.time()
            logMsg = f"Batch processed. Duration: {endBatchTime - startBatchTime:.2f} seconds."
            if enableLogging:
                logging.info(logMsg)
            else:
                print(logMsg)
        self.pgIndex=pgIndex

    def _build(self, documents, batchSize=4):
        
        self._buildVector(documents,batchSize)
        self._buildPg(documents,batchSize)

    def append(self,inputDocsDir=inputDir,batchSize=batchSize,indexType='both'):
        documents = SimpleDirectoryReader(
            inputDocsDir,
            recursive=True,
            filename_as_id=True,
            file_extractor={
                '.csv': CustomCSVReader(),
                '.pdf': CustomPDFReader(llm=self.llm),
            },
        ).load_data()
        
        indexTypes=['vector','graph','both']
        
        if indexType == indexTypes[0]:
            self._buildVector(documents,batchSize)
        elif indexType == indexTypes[1]:
            self._buildPg(documents,batchSize)
        elif indexType == indexTypes[2]:
            self._build(documents, batchSize)
        else:
            supportedTypes = ','.join(indexTypes)
            raise Exception(f'indexType must be in ({supportedTypes})')
    
    def get(self, indexType='both'):
        indexTypes=['vector','graph','both']
        if indexType == indexTypes[0]:
            return self.vectorIndex
        elif indexType == indexTypes[1]:
            return self.pgIndex
        elif indexType == indexTypes[2]:
            return self.vectorIndex, self.pgIndex
        else:
            supportedTypes = ','.join(indexTypes)
            raise Exception(f'indexType must be in ({supportedTypes})')
        

class QueryEngine:
    
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

    successMsg = 'Query Engine was build successfully!'

    def __init__(self,llm,embedModel,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore) -> None:
        self.vectorTopK=vectorTopK 
        self.contextTopK=contextTopK
        self.maxSynonyms=maxSynonyms
        self.pathDepth=pathDepth
        self.cutoffScore=cutoffScore
        self.index=Index(llm)
        self.embedModel=embedModel

    def _buildVectorQueryEngine(self, forceReindex=False):
        if forceReindex:
            self.index.append(inputDocsDir=inputDir,indexType='vector')
        self.vectorRetriever = VectorIndexRetriever(
            index=self.index.vectorIndex,
            similarity_top_k=self.vectorTopK
        )
        self.vectorQueryEngine = RetrieverQueryEngine.from_args(
            self.vectorRetriever,
            text_qa_template=self.textQaTemplate,
        )
        return self.successMsg
    
    def _buildPgQueryEngine(self, forceReindex=False):
        if forceReindex:
            self.index.append(inputDocsDir=inputDir,indexType='graph')
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
        return self.successMsg
    
    def _buildHybridQueryEngine(self):
        self.graphVecRetriever = HybridRetriever(
            self.vectorRetriever, 
            self.graphRetriever, 
            mode='OR'
        )
        self.graphVecQueryEngine = RetrieverQueryEngine.from_args(
            self.graphVecRetriever,
            text_qa_template=self.textQaTemplate,
        )
        return self.successMsg

    def _buildAll(self, forceReindex=False):
        _ = self._buildVectorQueryEngine(forceReindex=forceReindex)
        _ = self._buildPgQueryEngine(forceReindex=forceReindex)
        rm = self._buildHybridQueryEngine()
        print(rm)
        return self.successMsg

    def get(self,indexType = 'vector'):
        indexTypes = ['vector','graph','hybrid']
        if indexType == indexTypes[0]:
            rm = self._buildVectorQueryEngine()
            return self.vectorQueryEngine, rm
        elif indexType == indexTypes[1]:
            rm = self._buildPgQueryEngine()
            return self.graphQueryEngine, rm
        elif indexType == indexTypes[3]:
            rm = self._buildHybridQueryEngine()
            return self.graphVecQueryEngine, rm
        else:
            supportedTypes = ','.join(indexTypes)
            raise Exception(f'indexType must be in ({supportedTypes})')
        
    def getAll(self):
        rm = self._buildAll()
        return self.vectorQueryEngine,self.graphQueryEngine,self.graphVecQueryEngine, rm

class ChatbotAgents:
    successMsg = 'Chatbot Agents updated successfully!'
    def __init__(self,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore,configPath) -> None:
        self.vectorTopK=vectorTopK
        self.contextTopK=contextTopK
        self.maxSynonyms=maxSynonyms
        self.pathDepth=pathDepth
        self.cutoffScore=cutoffScore
        # The LLM API endpoint for synthesizing output using Azure OpenAI
        self.getModelConfigs(configPath)
        self.llm = AzureOpenAI(
            model=self.llmModelName,
            deployment_name=self.llmModelDeployment,
            api_key=self.apiKey,
            azure_endpoint=self.azureEndpoint,
            api_version=self.llmModelApiVersion,
        )
        print("self.llm",self.llm)
        # The embedding model API endpoint using Azure OpenAI
        self.embedModel = AzureOpenAIEmbedding(
            model=self.embeddingModelName,
            deployment_name=self.embeddingModelDeployment,
            api_key=self.apiKey,
            azure_endpoint=self.azureEndpoint,
            api_version=self.embeddingModelApiVersion,
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embedModel

        self.updateQueryEngine()
    
    def getModelConfigs(self,configPath):
        # Azure Endpoint
        config = ConfigParser()
        config.read(configPath)

        self.apiKey = config.get('azure-openai', 'api_key')
        self.azureEndpoint = config.get('azure-openai', 'azure_endpoint')

        self.embeddingModelName = config.get('embedding-model', 'model')
        self.embeddingModelDeployment = config.get('embedding-model', 'deployment_name')
        self.embeddingModelApiVersion = config.get('embedding-model', 'api_version')

        llmModel = 'gpt4o'  # can be set to `gpt35turbo` or `gpt4o`, `gpt4o` is much stronger but is also more expensive
        self.llmModelName = config.get(llmModel, 'model')
        self.llmModelDeployment = config.get(llmModel, 'deployment_name')
        self.llmModelApiVersion = config.get(llmModel, 'api_version')

    def updateQueryEngine(self):
        self.queryEngine = QueryEngine(self.llm,self.embedModel,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore)
        self.queryEngine._buildAll()
        customPromptMessages = [
            ChatMessage(role="system", content=systemMessage),
        ]
        self.customPromptTemplate = ChatPromptTemplate(customPromptMessages)
        self._buildAll()
    
    def _buildAll(self):
        self.vectorQueryEngineTool = QueryEngineTool(
        query_engine=self.queryEngine.vectorQueryEngine,
        metadata=ToolMetadata(
            name="vector_rag_query_engine",
            description="Useful for answering all queries based on the given context.",
        ),
        )
        self.vectorAgent = OpenAIAgent.from_tools(
            [self.vectorQueryEngineTool],
            verbose=True,
            prefix_messages=self.customPromptTemplate.message_templates,
        )

        graphQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.graphQueryEngine,
            metadata=ToolMetadata(
                name="graph_rag_query_engine",
                description="Useful for answering all queries based on the given context.",
            ),
        )
        self.graphAgent = OpenAIAgent.from_tools(
            [graphQueryEngineTool],
            verbose=True,
            prefix_messages=self.customPromptTemplate.message_templates,
        )

        graphVectorQueryEngineTool = QueryEngineTool(
            query_engine=self.queryEngine.graphVecQueryEngine,
            metadata=ToolMetadata(
                name="graph_vector_rag_query_engine",
                description="Useful for answering all queries based on the given context.",
            ),
        )
        self.graphVectorAgent = OpenAIAgent.from_tools(
            [graphVectorQueryEngineTool],
            verbose=True,
            prefix_messages=self.customPromptTemplate.message_templates,
        )
        self.selectedAgent=self.vectorAgent
    
    def appendIndex(self,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore):
        self.updateParams(vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore)
        self.queryEngine.index.append(inputDocsDir=inputDir,indexType='both')
        self.queryEngine._buildAll(forceReindex=True)
        self._buildAll()
        return self.successMsg

    def getDefault(self):
        self.queryEngine.index.fileManager.createDummyFile()
        self.vectorAgent,_,_= self.getAll(forceReindex=False)
        return self.vectorAgent

    def getAll(self, forceReindex=False):
        if forceReindex:
            self.appendIndex()
        return self.vectorAgent, self.graphAgent, self.graphVectorAgent, self.successMsg
    
    def updateParams(self,vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore):
        self.vectorTopK=vectorTopK
        self.contextTopK=contextTopK
        self.maxSynonyms=maxSynonyms
        self.pathDepth=pathDepth
        self.cutoffScore=cutoffScore
        self.updateQueryEngine()
    
    def chatbot(self, queryStr, history, agent):
        chatOutput = agent.chat(queryStr)
        references = formatReferences(getAgentCitations(chatOutput, cutoff_score=self.cutoffScore))
        output = chatOutput.response + references
        print("Chatbot Answer:\n",output)
        return output
    
    def uploadFiles(self,files):
        # try:
        msg = self.queryEngine.index.fileManager.upload(files)
        print(msg)
        return msg
        # except:
        #     return "Uploading files failed!"


    def clearFiles(self):
        try:
            self.queryEngine.index.storage.clear()
            self.queryEngine.index.fileManager.clear()
            return "Database and input directory were cleared successfully!"
        except:
            return "Clearing the database and input directory failed!"

    def getFilesList(self):
        return self.queryEngine.index.fileManager.ls()
    
    def selectAgent(self,indexType):
        if indexType == 'vector index':
            self.selectedAgent = self.vectorAgent
        if indexType == 'graph index':
            self.selectedAgent = self.graphAgent
        if indexType == 'graph+vector index':
            self.selectedAgent = self.graphVectorAgent
        return self.selectedAgent

# This is a mock like functionality, the data is not stored
def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: ", data.value)
    else:
        print("You downvoted this response: ", data.value)

if __name__ == "__main__":
    
    ################################ Gradio UI ################################

    # Initialize chatbot agents
    chatbotAgents = ChatbotAgents(vectorTopK,contextTopK,maxSynonyms,pathDepth,cutoffScore,configPath=configPath)

    vectorAgnt, graphAgnt, graphVectorAgnt, indexStatus = chatbotAgents.getAll(forceReindex=False)

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
        # Initialize states to store agents
        vectorAgent = gr.State(vectorAgnt)
        graphAgent = gr.State(graphAgnt)
        graphVectorAgent = gr.State(graphVectorAgnt)
        selectedAgent = gr.State(vectorAgnt)

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
                        choices=['vector index', 'graph index', 'graph+vector index'],
                        value='vector index',
                        label='Retriever Index'
                    )
                    vectorTopKSelector = gr.Slider(label='Vector Top k', minimum=1, maximum=10, value=vectorTopK, step=1)
                    graphTopKSelector = gr.Slider(label='Graph Top k', minimum=1, maximum=10, value=contextTopK, step=1)
                    maxSynonymsSelector = gr.Slider(label='Max Synonyms to Search', minimum=1, maximum=10, value=maxSynonyms, step=1)
                    pathDepthSelector = gr.Slider(label='Graph Path Depth', minimum=1, maximum=5, value=pathDepth, step=1)
                    cutoffScoreSelector = gr.Slider(label='References Min Relevance Score', minimum=0, maximum=1, value=cutoffScore, step=0.01)
                    submitRetrievalParams = gr.Button(value='Update Chatbot')

            with gr.Column(scale=7, elem_id="col"):
                gr.Markdown(f'<h1 style="text-align: center;">Literature Chatbot Prototype v{appVersion}</h1>')
                chatbot = gr.Chatbot(elem_id="chatbot", render=False)
                chatbot.like(vote, None, None)
                gr.ChatInterface(
                    chatbotAgents.chatbot,
                    chatbot=chatbot,
                    additional_inputs=[selectedAgent],
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
            outputs=[selectedAgent],
            api_name="Update Chatbot2",
            show_progress=True
        )

    demo.queue().launch(share=True)
