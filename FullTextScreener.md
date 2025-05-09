# Full Text Screener:

### **Introduction**

This Python file is a RAG script instead of a notebook which is optimized to run efficiently and can be deployed on a server

### **Installation & Dependencies**

To install enter into terminal: `pip install -r requirements.txt`

Here are the categories of the imports:

- PyTorch & Deep Learning Frameworks: These libraries are used for deep learning, model training, and inference
    - `torch`: used to perform parallel processing and significantly speed up tasks like training deep learning models
    - `transformers`: used to import BitsAndBytesConfig which is used for quantization
    - `huggingface_hub[inference]`: used to authenticate your environment with Hugging Face's Hub using a personal access token.
    - `bitsandbytes`: maybe remove this
- LlamaIndex (Data Indexing & Retrieval): LlamaIndex is used for building retrieval-augmented generation (RAG) systems
    - `llama-index`: used to build RAG systems
    - `llama-index-llms-huggingface`: used to load LLMs from huggingface
    - `llama-index-embeddings-huggingface`: used to load embeddings from huggingface
    - `llama-index-graph-stores-neo4j`: used to create and store graph databases for RAG systems
    - `llama-index-vector-stores-neo4jvector`: used to enable the storage and searching of vector data
    - `llama-index-llms-mistralai` : used to connect to mistal AI model
    - `llama-index-embeddings-mistralai`: used to connect to mistral AI embeddings
- Neo4j (Graph Database Integration): These libraries are for interacting with the Neo4j database
    - `neo4j`: used to create GraphDatabase driver with the provided URL and authentication credentials, enabling interactions with the database for storage and indexing purposes
- Data Processing & Analysis: Used for handling structured data and files
    - `pandas`: used to store Data as tables
- Docker (Containerization & Deployment): Used for container management and deployment
    - `docker`: used as a container to run the script

### **Usage**

- Show how to run the script (e.g., `python script.py`).
- Include example command-line arguments if applicable.
- Provide sample inputs and expected outputs.

## **Classes & Functions**

### **Class Config**

- **Purpose:**
    
    Loads configuration parameters from a config file and makes them available as attributes.
    
- **Attributes:**
    - **`self.config`**: The parsed configuration file object (using `ConfigParser()`).
    - **`self.dict`**: A dictionary holding each configuration section (with keys processed for Python naming).
    - **Dynamic Attributes**: Each section (with hyphens replaced by underscores) becomes an attribute (e.g., `self.general`, `self.huggingface`).
- **Methods:**
    - **`__init__(self, config_file)`**:
        - Reads the config file.
        - Iterates through sections, processing each via `_load_section`.
        - Stores each processed section as both an attribute and in `self.dict`.
    - **`_load_section(self, section)`**:
        - Iterates over key-value pairs in a given section.
        - Uses `_convert_value` to parse each value into the correct type.
    - **`_convert_value(value)`** *(Static Method)*:
        - Converts string values to appropriate types:
            - Boolean conversion for `"true"`/`"false"`.
            - Comma-separated values to a list.
            - Numeric conversion (int or float) when possible.
            - Falls back to the original string if no conversion applies.

---

### **Class DateTimeEncoder**

- **Purpose:**
    
    Extends Python’s built-in `json.JSONEncoder` to support encoding of `datetime` objects.
    
- **Functionality:**
    - Overrides the `default` method: if an object is a `datetime.datetime`, returns its ISO 8601 formatted string; otherwise, defers to the base implementation.

---

### **Class FileManagement**

- **Purpose:**
    
    Manages file operations (uploading, listing, and clearing files) and ensures necessary directories exist.
    
- **Attributes:**
    - **`self.config`**: Configuration object with directory paths.
    - **`self.inputDir`**: Directory for input files.
    - **`self.logDir`**: Directory for log files.
    - **`self.dataPath`**: Directory for main data storage.
    - **`self.pluginsPath`**: Directory for plugin files.
    - **`self.outputDir`**: Directory for output files.
    - **`self.fileNames`**: List of filenames uploaded to the input directory.
- **Methods:**
    - **`__init__(self, config)`**:
        - Stores the configuration.
        - Calls `getConfigs()` to extract paths.
        - Calls `buildFolders()` to create missing directories.
    - **`getConfigs(self)`**:
        - Extracts and sets directory paths from the configuration.
    - **`buildFolders(self)`**:
        - Ensures each required folder exists (using a helper like `ensureFolderExists`).
    - **`append(self, files)`**:
        - Copies given files to `self.inputDir` without deleting existing files.
        - Updates `self.fileNames` with the new filenames.
    - **`upload(self, files)`**:
        - Clears the input directory before calling `append(files)`.
    - **`clear(self)`**:
        - Deletes all contents in the input directory and resets `self.fileNames`.
    - **`ls(self)`**:
        - Returns the list of filenames currently in the input directory.
    - **`createDummyFile(self)`**:
        - Creates a dummy file (e.g., `dummy.txt`) in the input directory (useful for testing).

---

### **Class Storage**

- **Purpose:**
    
    Initializes and maintains the Neo4j vector database (vector store) for embedding storage.
    
- **Attributes:**
    - **`self.config`**: Configuration settings.
    - **`self.url`**: Neo4j database URL.
    - **`self.username`**: Neo4j username.
    - **`self.password`**: Neo4j password.
    - **`self.embedDim`**: Embedding dimension for vector storage.
    - **`self.vectorStore`**: Instance of `Neo4jVectorStore` (manages embeddings).
    - **`self.storageContextVector`**: A `StorageContext` (provides a standardized interface for storage operations).
- **Methods:**
    - **`__init__(self, config)`**:
        - Retrieves Neo4j credentials and embedding settings from the configuration.
        - Initializes `self.vectorStore` using those credentials.
        - Creates a storage context (`self.storageContextVector`) based on the vector store.
    - **`getConfigs(self)`**:
        - Loads Neo4j connection details and embedding dimension from the config.
    - **`clear(self)`**:
        - Connects to the Neo4j database and deletes all nodes and relationships, effectively clearing the database.

---

### **Class Index**

- **Purpose:**
    
    Manages and builds a vector index used for document (information) retrieval.
    
- **Attributes:**
    - **`self.llm`**: The Large Language Model (LLM) passed to the constructor.
    - **`self.config`**: Configuration settings for index behavior.
    - **`self.storage`**: Instance of the `Storage` class for vector storage.
    - **`self.vectorIndex`**: The vector index, built from the vector store (initialized via `VectorStoreIndex.from_vector_store`).
    - **`self.fileManager`**: Instance of the `FileManagement` class for handling document files.
- **Methods:**
    - **`__init__(self, llm, config)`**:
        - Initializes the index class by setting the LLM and config.
        - Creates a `Storage` instance.
        - Loads an existing vector index from the storage.
        - Initializes the file manager.
    - **`getConfigs(self)`**:
        - Retrieves configuration settings (e.g., retriever parameters) needed by the index.
    - **`_buildVector(self, documents, batchSize=4)`**:
        - Processes documents in batches.
        - For the first batch, creates a new vector index via `VectorStoreIndex.from_documents`.
        - For subsequent batches, adds nodes to the existing index via `insert_nodes`.
        - Updates `self.vectorIndex` with the final index.
    - **`append(self, inputDocsDir, batchSize=1)`**:
        - Loads documents from a directory (using `SimpleDirectoryReader` with custom file extractors) and calls `_buildVector` to update the index.
    - **`list_indexes(self)`**:
        - Lists current indexes in the Neo4j database.
    - **`drop_index(self, index_info)`**:
        - Drops a specific index or constraint from the database.
    - **`get(self)`**:
        - Returns the current vector index.
    - **`clear(self)`**:
        - Clears all indexes in the Neo4j database.

---

### **Class QueryEngine**

- **Purpose:**
    
    Builds and manages a query engine for efficient retrieval using vector-based methods.
    
- **Attributes:**
    - **`self.config`**: Configuration settings.
    - **`self.index`**: Instance of the `Index` class (provides access to the vector index).
    - **`self.vectorTopK`**: Parameter specifying how many top results to retrieve.
    - **`self.vectorRetriever`**: Instance of `VectorIndexRetriever`, which retrieves documents from the vector index based on similarity.
    - **`self.vectorQueryEngine`**: Instance of `RetrieverQueryEngine`, which processes queries and returns results using the retriever and a prompt template.
- **Methods:**
    - **`__init__(self, llm, embedModel, config)`**:
        - Initializes the QueryEngine by setting config and creating an `Index` instance.
        - Retrieves the `vectorTopK` parameter from the configuration.
    - **`getConfigs(self)`**:
        - Retrieves additional configuration settings (like `inputDir`).
    - **`_buildVectorQueryEngine(self, forceReindex=False)`**:
        - If `forceReindex` is `True`, re-uploads documents to rebuild the index.
        - Creates a `VectorIndexRetriever` using `self.index.vectorIndex` and `vectorTopK`.
        - Initializes `self.vectorQueryEngine` with the retriever and a text-based QA prompt template.
        - Prints a success message.
    - **`get(self)`**:
        - Calls `_buildVectorQueryEngine()` and returns the built query engine along with a success message.

---

### **Class ChatbotAgents**

- **Purpose:**
    
    Acts as an orchestrator that integrates the LLM, embeddings, query engine, and file management components to provide a chatbot interface using vector-based retrieval.
    
- **Attributes:**
    
    *(Key attributes include, but are not limited to):*
    
    - **`availableServices`**: List of supported LLM services (e.g., HuggingFace, Mistral, llama3).
    - **`configPath`**: Path to the configuration file.
    - **`config`**: Instance of the `Config` class.
    - **`llm`**: Loaded Large Language Model.
    - **`embedModel`**: Embedding model for generating vector representations.
    - **`queryEngine`**: Instance of `QueryEngine` for handling queries.
    - **`vectorAgent`**: An agent (often a `ReActAgent`) that processes queries and manages conversation state.
- **Methods:**
    - **`__init__(self, configPath)`**:
        - Reads configuration and model parameters.
        - Based on the service specified (e.g., HuggingFace or Mistral), loads the appropriate LLM and embedding model.
        - Updates global settings (e.g., `Settings.llm`).
        - Calls `updateQueryEngine()` to initialize the query engine and then selects a default agent.
    - **`getModelConfigs(self, configPath)`**:
        - Reads and extracts model-specific configuration (service type, model names, API keys, parameters such as temperature, max tokens, etc.).
    - **`getMistralLlmAndEmbedding(self)` & `getHuggingFaceLlmAndEmbedding(self)`**:
        - Load the corresponding LLM and embedding model based on the service.
        - Configure parameters such as temperature, max tokens, quantization settings (for HuggingFace).
    - **`getConfigs(self)`**:
        - Retrieves basic configurations such as the input directory.
    - **`updateQueryEngine(self)`**:
        - Instantiates the `QueryEngine` with the LLM, embedding model, and configuration.
        - Calls `_buildVectorQueryEngine()` to set up the vector-based retrieval system.
        - Calls `_buildAll()` to configure the query agent.
    - **`_buildAll(self)`**:
        - Sets up the custom prompt template for the agent.
        - Initializes the `QueryEngineTool` and creates a `ReActAgent` using the tool.
    - **`appendIndex(self, vectorTopK, cutoffScore)`**:
        - Updates retriever parameters and forces reindexing of documents.
    - **`getDefault(self)` & `getAll(self, forceReindex=False)`**:
        - Provide access to the default or current agent, optionally reindexing documents.
    - **`updateParams(self, vectorTopK, cutoffScore)`**:
        - Updates retriever parameters in the configuration and rebuilds the query engine.
    - **`chatbot(self, queryStr, history)`**:
        - Processes a user query.
        - Depending on whether `enable_agent` is enabled, uses either the agent (`selectedAgent.chat`) or a direct query engine call.
        - Returns the response (assumed to be a JSON-formatted string).
    - **File and Database Operations:**
        - **`uploadFiles(self, files)`**: Uploads files to the document index.
        - **`clearFiles(self)`**: Clears the database and file directory.
        - **`getFilesList(self)`**: Returns the list of currently uploaded files.
    - **Agent Selection and Cleanup:**
        - **`selectAgent(self, indexType=None)`**: Selects an agent.
        - **`selectDefaultAgent(self)`**: Sets the default agent.
        - **`getSelectedAgent(self)`**: Returns the active agent.
        - **`cleanup(self)`**: Cleans up resources (GPU memory, models, query engine, etc.) to free memory.

---

### **Functions**

- **`setChunkingMethod(config)`**
    - **Purpose:** Configures the method used to split PDF documents into chunks.
    - **How It Works:**
        - Reads node parser type and chunking parameters (chunk size, overlap, buffer size, breakpoint threshold) from the configuration.
        - Sets global settings (e.g., `Settings.chunk_size` and `Settings.chunk_overlap`).
        - If the node parser type is `"semantic"`, it sets up a combination of a semantic splitter (which uses an embedding model) and a sentence splitter.
        - If it is `"static"`, only a sentence splitter is configured.
        - Raises an exception if an invalid parser type is specified.
- **`buildNeo4jContainer(config)`**
    - **Purpose:** Manages the lifecycle of a Neo4j Docker container.
    - **How It Works:**
        - Reads container details and Neo4j URLs from the configuration.
        - Uses the Docker Python client to check if the container exists and is running.
        - If not running or non-existent, it creates/starts the container.
        - Waits until the container and the Neo4j service are ready before proceeding.
- **`setLogging(config)`**
    - **Purpose:** Configures the logging system based on settings from the configuration.
    - **How It Works:**
        - Reads logging parameters (enableLogging, log directory, log file).
        - Ensures the log directory and file exist.
        - Sets up logging handlers (file and stream) and adjusts log levels for third-party libraries like `requests` and `urllib3`.
        - Returns a logger object for the module.
- **`safeApiCall(call, enableLogging, *args, **kwargs)`**
    - **Purpose:** Safely executes an API call with automatic retries in case of errors.
    - **How It Works:**
        - Attempts the API call up to 20 times.
        - Uses exponential backoff (with a random component) between attempts.
        - Logs the duration of a successful call and any errors encountered.
        - Raises an exception if all attempts fail.
- **`cleanAndConvertJson(jsonText)`**
    - **Purpose:** Cleans a JSON-formatted string (removing extraneous characters) and converts it into a Python dictionary.
    - **How It Works:**
        - Strips markers (e.g., backticks, extra quotes).
        - Replaces escape sequences with proper characters.
        - Tries to parse the cleaned string as JSON; if it fails, returns the cleaned string.
- **`process_single_paper(row, questions, configPath, input_pdf_folder)`**
    - **Purpose:** Processes a single research paper:
        - Instantiates a fresh `ChatbotAgents` instance.
        - Clears any previous data and uploads a PDF corresponding to the paper.
        - Indexes the PDF and iterates through a set of questions, retrieving answers for each.
        - Returns the results in a dictionary mapping the paper ID to its answers.
    - **How It Works:**
        - Uses file management to locate and upload the PDF.
        - Indexes the document using the vector-based retrieval system.
        - Uses `safeApiCall` and the chatbot method to process each question.
        - Cleans up resources after processing.
- **`literature_screening(metadata_file_path, questions, configPath, input_pdf_folder, output_json_path)`**
    - **Purpose:** Processes multiple papers for literature screening.
    - **How It Works:**
        - Reads metadata from a CSV or Excel file.
        - For each paper, spawns a separate process (using multiprocessing) that calls `process_single_paper`.
        - Aggregates results from all papers and writes them to a JSON file.
        - Logs overall processing time and average time per paper.

## **Workflow Overview**

This section describes a mid-level execution flow of the RAG system using an example of processing 4 research papers. The workflow outlines how configuration, file management, storage, indexing, querying, and agent orchestration work together to produce retrieval-augmented answers for each paper.

### **Step 1: Configuration Setup**

- **Class Involved:** `Config`
- **Action:**
    
    The system starts by reading a configuration file (e.g., `config.ini`) using the **Config** class. This loads global parameters such as:
    
    - Directory paths (input, logs, output, etc.)
    - Neo4j connection details (URL, username, password)
    - LLM settings (service, model name, API key, temperature, etc.)
    - Retriever and agent parameters (e.g., `vectortopk`, `enable_agent`, `max_iterations`)

### **Step 2: File Management Initialization**

- **Class Involved:** `FileManagement`
- **Action:**
    
    The **FileManagement** class is initialized with the configuration to:
    
    - Ensure that required directories exist (input, logs, output, etc.)
    - Manage file uploads (tracking filenames and handling clear operations)

### **Step 3: Storage Setup**

- **Class Involved:** `Storage`
- **Action:**
    
    The **Storage** class establishes a connection to the Neo4j database by:
    
    - Retrieving credentials and embedding dimensions from the configuration.
    - Initializing a `Neo4jVectorStore` to store document embeddings.
    - Creating a `StorageContext` to provide a standardized interface for storage operations.

### **Step 4: Building the Vector Index**

- **Class Involved:** `Index`
- **Action:**
    
    The **Index** class builds a vector index from the uploaded documents by:
    
    - Using a file reader (e.g., `SimpleDirectoryReader`) along with custom file extractors.
    - Processing documents in batches via `_buildVector`:
        - **First Batch:** Creates a new vector index using `VectorStoreIndex.from_documents`.
        - **Subsequent Batches:** Adds documents to the existing index with `insert_nodes`.
    - Storing the updated index in the vector store (managed by **Storage**).

### **Step 5: Query Engine Initialization**

- **Class Involved:** `QueryEngine`
- **Action:**
    
    The **QueryEngine** is instantiated with the LLM, embedding model, and configuration. It performs the following:
    
    - Loads the vector index from the **Index** class.
    - Creates a **VectorIndexRetriever** that searches for the top‑K similar document embeddings when a query is issued.
    - Wraps the retriever into a **RetrieverQueryEngine** that applies a prompt template to guide the LLM in generating a context-aware response.

### **Step 6: Agent Integration and Chatbot Setup**

- **Class Involved:** `ChatbotAgents`
- **Action:**
    
    The **ChatbotAgents** class orchestrates the entire process by:
    
    - Loading the appropriate LLM and embedding model based on the configuration (e.g., HuggingFace or Mistral).
    - Initializing the **QueryEngine** (which in turn uses the Index and its retriever components).
    - Setting up a ReAct agent (via `_buildAll`) that integrates the query engine tool.
        
        This agent enables multi-step reasoning and maintains conversational context if `enable_agent` is set to `True`.
        

### **Step 7: Processing 4 Papers (Example Scenario)**

Assume we have 4 research papers with unique IDs and corresponding PDF files. The processing follows these steps:

1. **Paper Metadata:**
    
    The system reads a metadata file (CSV or Excel) that lists details for each paper (e.g., paper_id, pdf_filename).
    
2. **Per-Paper Processing:**
    
    For each paper (handled in a separate process via `multiprocessing`):
    
    - **Initialization:**
        
        A fresh instance of **ChatbotAgents** is created (using the configuration path). This resets the state so each paper is processed independently.
        
    - **File Upload & Indexing:**
        - The **FileManagement** component uploads the PDF corresponding to the paper (using `uploadFiles`).
        - The **Index** class processes the uploaded PDF:
            - If it’s the first paper or a forced reindex is triggered, the system builds a new vector index.
            - Otherwise, it appends the new paper’s embeddings to the existing index.
    - **Querying:**
        
        For each predefined question:
        
        - The **chatbot** function of **ChatbotAgents** is invoked.
        - If agent mode is enabled (`enable_agent = True`), the ReAct agent queries the vector engine tool.
        - The **QueryEngine** uses the **VectorIndexRetriever** to find the most similar sections from the paper’s index, and the **RetrieverQueryEngine** applies the prompt template and LLM to generate a final answer.
        - The answer is cleaned and converted to JSON via `cleanAndConvertJson`.
    - **Result Collection:**
        
        The system aggregates each paper's answers into a dictionary mapping the paper_id to its metadata and query responses.
        
3. **Aggregation and Output:**
    
    After all 4 papers are processed, the results are combined and written to a JSON file by the `literature_screening` function. Additionally, the total processing time and average time per paper are logged.
    

### **Step 8: Resource Cleanup**

- **Class Involved:** `ChatbotAgents` (cleanup method)
- **Action:**
    
    Once processing for a paper is complete, the `cleanup` method is called to:
    
    - Clear the uploaded files and database entries.
    - Release GPU memory by clearing model objects and invoking garbage collection.
    - Ensure that each process leaves no lingering resources before processing the next paper.