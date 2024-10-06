#!/usr/bin/env python
# coding: utf-8

# Literature Chatbot PoC (w/ Neo4j)

# Literature Chatbot using Graph/Vector Index RAG Pipeline - using Neo4j as graph database
# This version provides the following features:
    # 1. Uses Neo4j as both a GraphStore and a VectorStore
    # 2. Uses PropertyGraphIndex to extract and query Knowledge Graphs
    # 3. Incorporates exponential backoff when calling openai APIs in the indexing phase, 
    #    to avoid hitting API rate limit
    # 4. By default, appends the database with the new document
    # 5. Optionally, allows to clear the database and start fresh 
    # 6. The script can hit the API rate limit if an input document is too large. 
    #    This typically does not happen for PDF files since each Document object includes 
    #    only one page. It is likely to happen for documents comming from CSV files, 
    #    which are not split before ingestion.

# Install the necessary dependencies by `pip install -r requirements.txt`.

################################ Import Required dependencies ################################
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
# import openai
from tqdm import tqdm
from neo4j import GraphDatabase

import torch
import transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from llama_index.core.agent import ReActAgent

# from llama_index.llms.azure_openai import AzureOpenAI
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, PGRetriever, VectorContextRetriever, LLMSynonymRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor, ImplicitPathExtractor
from llama_index.core.utils import iter_batch
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPGStore as Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore


from Modules.Readers import CustomCSVReader, CustomPDFReader
from Modules.Tools import touch, delete_folder_contents, ensure_folder_exists
from Modules.Retrievers import HybridRetriever
from Modules.CitationExtractor import format_references
from Modules.CitationExtractor import get_agent_citations_n4j as get_agent_citations
from Modules.NodeParser import CustomSemanticSplitterNodeParser
 
################################ Set Configurations ################################

# Directories
input_dir = './input/selected'
log_dir = './logs/'
data_path = "./neo4j_vol1/data"
plugins_path = "./neo4j_vol1/plugins"

llm_model_id = "tiiuae/falcon-7b-instruct"
# llm_model_id = "EleutherAI/gpt-neo-1.3B"
# # Azure Endpoint
# config = ConfigParser()
# config.read('./config/config.cfg')

# api_key = config.get('azure-openai', 'api_key')
# azure_endpoint = config.get('azure-openai', 'azure_endpoint')

# embedding_model_name = config.get('embedding-model', 'model')
# embedding_model_deployment = config.get('embedding-model', 'deployment_name')
# embedding_model_api_version = config.get('embedding-model', 'api_version')

# llm_model = 'gpt4o'  # can be set to `gpt35turbo` or `gpt4o`, `gpt4o` is much stronger but is also more expensive
# llm_model_name = config.get(llm_model, 'model')
# llm_model_deployment = config.get(llm_model, 'deployment_name')
# llm_model_api_version = config.get(llm_model, 'api_version')

################################ Set Parameters ################################

# Parameters
force_reindex = False  # whether to index documents in `input_dir` before launching the chatbot UI
enable_logging = False  # whether to log INFO and WARNING messages
show_progress = True  # whether to show low level progress bars, if set to false, only high level progress bar will be rendered

# Node Parser parameters
node_parser_type = 'static'  # node parser approach, can be either `static` or `semantic`
batch_size = 1  # number of documents to process at a time, larger values may lead to hitting the embedding API rate limit
Settings.chunk_size = 512  # Number of tokens to include in each chunk
Settings.chunk_overlap = 50  # The overlap between subsequent chunks

# Semantic Splitter parameters
buffer_size = 1  # Default = 1, Adjust buffer size as needed
breakpoint_percentile_threshold = 75  # Default = 95, lower values results into too many small chunks

# PropertyGraphIndex parameters
max_paths_per_chunk = 10

# Default retriever parameters - These can be adjusted in the UI
vector_top_k = 10
context_top_k = 2
max_synonyms = 10
path_depth = 1
cutoff_score = 0.9

######################### Ensure Folder Structure Exists #########################

ensure_folder_exists(input_dir)
ensure_folder_exists(log_dir)
ensure_folder_exists(data_path)
ensure_folder_exists(plugins_path)

######################### Configure Logging #########################

if enable_logging:
    log_path = os.path.join(log_dir, "processing_log.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

################################ Define Endpoints ################################


# The embedding model from Hugging Face
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# model_kwargs1 = {
#     # "temperature":1 ,
#     # "do_sample":True,
#     "min_new_tokens":200-25,
#     "max_new_tokens":200+25,
#     'repetition_penalty':20.0
# }
# Load an open-source LLM from Hugging Face (e.g., LLaMA, GPT-Neo, GPT-J)
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            )
model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(llm_model_id )
# model.to_bettertransformer()

# The LLM model wrapped for llama_index
llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
# pipeline = transformers.pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             pad_token_id=tokenizer.eos_token_id,
#             **model_kwargs1,
#         )
# llm = HuggingFacePipeline(pipeline=pipeline)

# # The embedding model API endpoint using Azure OpenAI
# embed_model = AzureOpenAIEmbedding(
#     model=embedding_model_name,
#     deployment_name=embedding_model_deployment,
#     api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     api_version=embedding_model_api_version,
# )

# # The LLM API endpoint for synthesizing output using Azure OpenAI
# llm = AzureOpenAI(
#     model=llm_model_name,
#     deployment_name=llm_model_deployment,
#     api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     api_version=llm_model_api_version,
# )

# set the service context using llama_index Settings
Settings.llm = llm
Settings.embed_model = embed_model


# print(llm.complete("No pain no "))
# ################################ Set Chunking Method ################################

custom_splitter = CustomSemanticSplitterNodeParser(
    buffer_size=buffer_size,
    breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    embed_model=Settings.embed_model,
    sentence_chunk_size=Settings.chunk_size,
    sentence_chunk_overlap=Settings.chunk_overlap
)

if node_parser_type == 'semantic':
    Settings.transformations = [custom_splitter]
elif node_parser_type == 'static':
    Settings.transformations = [SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap
    )]
else:
    raise Exception("`node_parser_type` must be either `static` or `semantic`")

################################ Setup Neo4j with Docker ################################

username = "neo4j"
password = "neo4j_rag_poc"
url = "bolt://localhost:7687"
embed_dim = 1536
container_name = "neo4j-apoc"

client = docker.from_env()

# Check if the container exists and is running
try:
    container = client.containers.get(container_name)
    if container.status != 'running':
        print(f"The container '{container_name}' is not running. Starting it...")
        container.start()
    else:
        print(f"The container '{container_name}' is already running.")
except docker.errors.NotFound:
    print(f"The container '{container_name}' does not exist. Creating and starting it...")

    data_path = os.path.abspath(data_path)
    plugins_path = os.path.abspath(plugins_path)

    docker_command = [
        "docker", "run", "--restart", "always", "--name", container_name,
        "--publish=7474:7474", "--publish=7687:7687",
        "--env", "NEO4J_AUTH=" + username + "/" + password,
        "-e", "NEO4J_apoc_export_file_enabled=true",
        "-e", "NEO4J_apoc_import_file_enabled=true",
        "-e", "NEO4J_apoc_import_file_use__neo4j__config=true",
        "-e", "NEO4J_PLUGINS=[\"apoc\"]",
        "-v", f"{data_path}:/data",
        "-v", f"{plugins_path}:/plugins",
        "neo4j:latest"
    ]

    try:
        subprocess.Popen(docker_command)
        print(f"Started the Docker container '{container_name}'.")
    except Exception as e:
        print(f"Error occurred while starting the container: {e}")

client.close()

################################ Connect to Neo4j ################################

graph_store = Neo4jPropertyGraphStore(
    username=username,
    password=password,
    url=url,
)

storage_context_graph = StorageContext.from_defaults(
    graph_store=graph_store,
)

vector_store = Neo4jVectorStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=embed_dim,
)

storage_context_vector = StorageContext.from_defaults(
    vector_store=vector_store,
)

system_message = """
    I am a Q&A assistant named Literature Chatbot, created by DSAT AI Lab at PMRA to assist scientist with their literature review tasks. 
    I am designed to search in the uploaded documents for the most relevent inforamation related to the user's question.
    and synthesize an informed answer. 
    I follow these rules:
    Rule 1: My main goal is to provide answers as accurately as possible, based on the instructions and context I have been given. 
    Rule 2: If the page number is available in the metadata, I report the page number for each piece of information that I provide as an inline citation with format [First Author Last Name et. al., Year of publication, page number(s)].
    Rule 3: I ALWAYS retrieve information from the query engine even if the question is repeated and I have the answer in my memory.
    Rule 4: If a question does not match the provided context or is outside the scope of the document, I do not provide answer from my past knowledge. I advise the user that the provided documents do not contain the requested information.
    """

# Function to handle rate limits with exponential backoff
def safe_api_call(call, *args, **kwargs):
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            start_time = time.time()
            result = call(*args, **kwargs)
            end_time = time.time()
            if enable_logging:
                logging.info(f"API call successful. Duration: {end_time - start_time:.2f} seconds.")
            return result
        except openai.RateLimitError as e:
            wait = 2 ** attempt + random.random()
            if enable_logging:
                logging.warning(f"Rate limit hit. Waiting for {wait:.2f} seconds.")
            time.sleep(wait)
    raise Exception("API call failed after maximum number of retries")

kg_extractors = [
    SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=max_paths_per_chunk,
        num_workers=4,
    ),
    ImplicitPathExtractor()
]

def create_index(documents, batch_size=4):
    for batch in tqdm(iter_batch(documents, batch_size), 
                      total=len(documents)//batch_size, 
                      desc='Processing Node Batches'):
        start_batch_time = time.time()
        pg_index = safe_api_call(
            PropertyGraphIndex.from_documents,
            batch,
            kg_extractors=kg_extractors,
            property_graph_store=graph_store,
            storage_context=storage_context_graph,
            show_progress=show_progress
        )
        vector_index = safe_api_call(
            VectorStoreIndex.from_documents,
            documents=batch,
            storage_context=storage_context_vector,
            show_progress=show_progress
        )
        end_batch_time = time.time()
        if enable_logging:
            logging.info(f"Batch processed. Duration: {end_batch_time - start_batch_time:.2f} seconds.")
        else:
            print(f"Batch processed. Duration: {end_batch_time - start_batch_time:.2f} seconds.")
    return pg_index, vector_index

def indexer(directory, force_reindex=False):
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    pg_index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
        embed_kg_nodes=True,
    )

    if force_reindex:
        documents = SimpleDirectoryReader(
            directory,
            recursive=True,
            filename_as_id=True,
            file_extractor={
                '.csv': CustomCSVReader(),
                '.pdf': CustomPDFReader(llm=llm),
            },
        ).load_data()
        pg_index, vector_index = create_index(documents, batch_size=batch_size)

    return vector_index, pg_index

prompt = (
    "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "QUERY: {query_str}\n"
    "----\n"
    "KEYWORDS: "
)

def get_retrievers(force_reindex=False, vector_top_k=10, context_top_k=2, max_synonyms=10, path_depth=1):
    vector_index, pg_index = indexer(input_dir, force_reindex=force_reindex)

    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=vector_top_k
    )

    sub_retrievers = [
        VectorContextRetriever(
            pg_index.property_graph_store,
            include_text=True,
            embed_model=embed_model,
            similarity_top_k=context_top_k,
            path_depth=path_depth
        ),
        LLMSynonymRetriever(
            pg_index.property_graph_store,
            include_text=True,
            synonym_prompt=prompt,
            max_keywords=max_synonyms,
            path_depth=path_depth,
        ),
    ]

    graph_rag_retriever = PGRetriever(sub_retrievers=sub_retrievers, use_async=False)
    graph_vec_retriever = HybridRetriever(vector_retriever, graph_rag_retriever, mode='OR')

    return vector_retriever, graph_rag_retriever, graph_vec_retriever

def get_query_engines(force_reindex=False, **args):
    vector_retriever, graph_rag_retriever, graph_vec_retriever = get_retrievers(force_reindex, **args)

    chat_text_qa_msgs = [
        (
            "user",
            system_message + """
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]

    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    vector_rag_query_engine = RetrieverQueryEngine.from_args(
        vector_retriever,
        text_qa_template=text_qa_template,
    )

    graph_rag_query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever,
        text_qa_template=text_qa_template,
    )

    graph_vector_rag_query_engine = RetrieverQueryEngine.from_args(
        graph_vec_retriever,
        text_qa_template=text_qa_template,
    )

    return vector_rag_query_engine, graph_rag_query_engine, graph_vector_rag_query_engine, "Indexing Complete!"

def get_chatbot_agents(vector_top_k=vector_top_k, context_top_k=context_top_k, max_synonyms=max_synonyms, path_depth=path_depth, force_reindex=False):
    vector_rag_query_engine, graph_rag_query_engine, graph_vector_rag_query_engine, index_status = get_query_engines(
        force_reindex=force_reindex,
        vector_top_k=vector_top_k,
        context_top_k=context_top_k,
        max_synonyms=max_synonyms,
        path_depth=path_depth
    )

    custom_prompt_messages = [
        ChatMessage(role="system", content=system_message),
    ]

    custom_prompt_template = ChatPromptTemplate(custom_prompt_messages)

    vector_query_engine_tool = QueryEngineTool(
        query_engine=vector_rag_query_engine,
        metadata=ToolMetadata(
            name="vector_rag_query_engine",
            description="useful for answering all queries based on the given context.",
        ),
    )
    # vector_agent = OpenAIAgent.from_tools(
    #     [vector_query_engine_tool],
    #     verbose=True,
    #     prefix_messages=custom_prompt_template.message_templates,
    # )
    vector_agent = ReActAgent.from_tools(
        [vector_query_engine_tool],
        verbose=True,
        llm=llm,
        prefix_messages=custom_prompt_template.message_templates,
    )

    graph_query_engine_tool = QueryEngineTool(
        query_engine=graph_rag_query_engine,
        metadata=ToolMetadata(
            name="graph_rag_query_engine",
            description="useful for answering all queries based on the given context.",
        ),
    )
    # graph_agent = OpenAIAgent.from_tools(
    #     [graph_query_engine_tool],
    #     verbose=True,
    #     prefix_messages=custom_prompt_template.message_templates,
    # )
    graph_agent = ReActAgent.from_tools(
        [graph_query_engine_tool],
        verbose=True,
        llm=llm,
        prefix_messages=custom_prompt_template.message_templates,
    )

    graph_vector_query_engine_tool = QueryEngineTool(
        query_engine=graph_vector_rag_query_engine,
        metadata=ToolMetadata(
            name="graph_vector_rag_query_engine",
            description="useful for answering all queries based on the given context.",
        ),
    )
    # graph_vector_agent = OpenAIAgent.from_tools(
    #     [graph_vector_query_engine_tool],
    #     verbose=True,
    #     prefix_messages=custom_prompt_template.message_templates,
    # )
    graph_vector_agent = ReActAgent.from_tools(
        [graph_vector_query_engine_tool],
        verbose=True,
        llm=llm,
        prefix_messages=custom_prompt_template.message_templates,
    )
    print("Chatbot agents updated!")
    return vector_agent, graph_agent, graph_vector_agent, index_status

def get_chatbot_agents_reindex(vector_top_k, context_top_k, max_synonyms, path_depth, 
                            #    progress=gr.Progress()
                               ):
    # progress = gr.Progress()
    # progress(0, desc="Starting...")
    # total_steps = 100  # Define the total number of steps for the progress bar
    # for i in progress.tqdm(range(total_steps)):
    #     time.sleep(0.1)  # Simulate work being done
        # progress(i / total_steps)  # Update progress
    return get_chatbot_agents(
        vector_top_k=vector_top_k,
        context_top_k=context_top_k,
        max_synonyms=max_synonyms,
        path_depth=path_depth,
        force_reindex=True)

def upload_file(files):
    delete_folder_contents(input_dir)
    message = ''
    for file in files:
        file_name = os.path.basename(file)
        target_file_name = os.path.join(input_dir, file_name)
        shutil.copyfile(file, target_file_name)
        message += str(file_name) + '\n'
    gr.Info("File(s) uploaded successfully!")
    message += str(len(files)) + " files were uploaded!\n"
    return message

def get_default_chatbot_agent(force_reindex=False, **args):
    touch(os.path.join(input_dir, 'dummy.txt'))
    vector_agent, _, _ = get_chatbot_agents(force_reindex, **args)
    return vector_agent

def graph_rag_chatbot_stream(query_str, history, agent):
    if agent is None:
        agent = get_default_chatbot_agent()
    chat_output = agent.stream_chat(query_str)
    res = ''
    for char in chat_output.response_gen:
        res += char
        yield res

def graph_rag_chatbot(query_str, history, agnt, cutoff_score=0.9):
    chat_output = agnt.chat(query_str)
    references = format_references(get_agent_citations(chat_output, cutoff_score=cutoff_score))
    output = chat_output.response + references
    print("Chatbot Answer:")
    print(output)
    return output

def select_agent(vector_agent, graph_agent, graph_vector_agent, index_type='graph+vector index'):
    if index_type == 'vector index':
        return vector_agent
    if index_type == 'graph index':
        return graph_agent
    if index_type == 'graph+vector index':
        return graph_vector_agent

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: ", data.value)
    else:
        print("You downvoted this response: ", data.value)

def clear_graph():
    try:
        driver = GraphDatabase.driver(url, auth=(username, password))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        return "Database cleared successfully!"
    except Exception as e:
        print(f"Error occurred while clearing the database: {e}")
        return f"Clearing the database failed with error: {e}"

import asyncio

################################ Gradio UI ################################

# Initialize chatbot agents
vector_agnt, graph_agnt, graph_vector_agnt, index_status = get_chatbot_agents(
    vector_top_k=vector_top_k,
    context_top_k=context_top_k,
    max_synonyms=max_synonyms,
    path_depth=path_depth,
    force_reindex=force_reindex
)

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
"""
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    # Initialize states to store agents
    vector_agent = gr.State(vector_agnt)
    graph_agent = gr.State(graph_agnt)
    graph_vector_agent = gr.State(graph_vector_agnt)
    selected_agent = gr.State(vector_agnt)

    with gr.Row():
        with gr.Column(scale=3, elem_id="menu"):
            reset_button = gr.Button("Reset the Database", render=True)
            reset_status = gr.Markdown("", line_breaks=True)
            upload_button = gr.UploadButton("Upload Files", file_count="multiple", file_types=['.pdf', '.csv'])
            upload_status = gr.Markdown("", line_breaks=True)
            index_button = gr.Button(value="Ingest Files")
            index_status = gr.Markdown("", line_breaks=True)

            with gr.Accordion("Retrieval Parameters", open=False):
                index_selector = gr.Dropdown(
                    choices=['vector index', 'graph index', 'graph+vector index'],
                    value='vector index',
                    label='Retriever Index'
                )
                vector_top_k_selector = gr.Slider(label='Vector Top k', minimum=1, maximum=10, value=vector_top_k, step=1)
                graph_top_k_selector = gr.Slider(label='Graph Top k', minimum=1, maximum=10, value=context_top_k, step=1)
                max_synonyms_selector = gr.Slider(label='Max Synonyms to Search', minimum=1, maximum=10, value=max_synonyms, step=1)
                path_depth_selector = gr.Slider(label='Graph Path Depth', minimum=1, maximum=5, value=path_depth, step=1)
                cutoff_score_selector = gr.Slider(label='References Min Relevence Score', minimum=0, maximum=1, value=cutoff_score, step=0.01)
                submit_retrival_params = gr.Button(value='Update Chatbot')

        with gr.Column(scale=7, elem_id="col"):
            gr.Markdown('<h1 style="text-align: center;">Literature Chatbot Prototype v1.9</h1>')
            chatbot = gr.Chatbot(elem_id="chatbot", render=False)
            chatbot.like(vote, None, None)
            gr.ChatInterface(
                graph_rag_chatbot,
                chatbot=chatbot,
                additional_inputs=[selected_agent, cutoff_score_selector],
                fill_height=True
            )

    reset_button.click(
        fn=clear_graph,
        outputs=reset_status,
        api_name="Reset Files",
        show_progress=True
    )

    upload_button.upload(
        fn=upload_file,
        inputs=upload_button,
        outputs=upload_status,
        show_progress="full",
    )


    index_button.click(
        fn=get_chatbot_agents_reindex,
        inputs=[
            vector_top_k_selector,
            graph_top_k_selector,
            max_synonyms_selector,
            path_depth_selector
        ],
        outputs=[
            vector_agent,
            graph_agent,
            graph_vector_agent,
            index_status
        ],
        api_name="Index Files",
        show_progress=True
    )


    submit_retrival_params.click(
        fn=get_chatbot_agents,
        inputs=[
            vector_top_k_selector,
            graph_top_k_selector,
            max_synonyms_selector,
            path_depth_selector
        ],
        outputs=[
            vector_agent,
            graph_agent,
            graph_vector_agent
        ],
        api_name="Update Chatbot1",
        show_progress=True
    )

    submit_retrival_params.click(
        select_agent,
        inputs=[
            vector_agent,
            graph_agent,
            graph_vector_agent,
            index_selector
        ],
        outputs=[selected_agent],
        api_name="Update Chatbot2",
        show_progress=True
    )

demo.queue().launch(share=True)


# TODO: Identify important parameters -> Done!
# TODO: Implement progress bar for ingest files, how could this be done due to the async implementation of the indexing methods?
#           - tqdm progress bar added in the back-end
#           - front-end now shows a progress visual, but not a progress bar. 
# TODO: Investigate if it is necessary to have the vector indices for graph_store and vector_store in different database schemas?
# TODO: Sort references based on score -> Done!
# TODO: Add functionality to clear the database - Done!
# TODO: Hide functionality to clear the database - Done!
# TODO: Add semantic chuncking -> Done!
# TODO: Expose retrival parameters in the UI - Done!

# important parameters:
    # Indexing:
        # Vector
            # Static:
                # Chunk size
                # Chunk overlap
            # Semantic:
                # buffer_size
                # breakpoint_percentile_threshold
        # Graph:
            # max_paths_per_chunk
    # Retrieval
        # Graph:
            # context_top_k
            # max_synonyms
            # path_depth
        # Vector:
            # vector_top_k 
    # Query Engine
        # NA
    # Agent


# Sample questions:
# Demo 1: Literature Search
# Provide a list of research articles that study the effects of herbicides or pesticides containing Bensulide on organisms or the environment. Exclude articles that are only included in the "references" of other articles but are not available in the context. For each article include: title, authors, year of publication, and a brief summary.

# Demo 2: Full-text Screening
# For the article "Bensulide-induced oxidative stress causes developmental defects of cardiovascular system and liver in zebrafish (Danio rerio)", provide answers for the following questions. Run separate queries for each question.

# 1. What herbicide or pesticide substance was studied?
# 2. Was the test substance guarantee/purity reported?
# 3. were the test results presented in terms of active ingredient?
# 4. was the test substance reported as technical grade or analytical standard?
# 5. Is the test organism reported (Scientific name and common name)?
# 6. Were test organisms exposed (in at least one treatment group) to the single active ingredient of interest (TGAI or formulation) or transformation product (of the active of interest)?
# 7. were test organisms exposed (in at least one treatment group) to the active ingredient in an end-use product containing multiple active ingredients (where the end-use product used in the test contains the same active ingredients as an EP proposed for registration or currently registered in Canada) ?
# 8. Are the measures of effects apical (survival, growth or reproduction)?
# 9. are the measures of effects expected to be directly linked to survival, growth or reproduction.?
# 10. Were appropriate exposure units used (e.g., mg a.i./kg diet, mg/kg bw), or is there sufficient information in the article to accurately estimate exposure in appropriate units?
# 11. Was the duration of exposure and frequency reported?
# 12. Does the study contain an endpoint that would be used in Quantitative risk assessment?

# Demo 3: Text summarization
# provide a summary of all US EPA regulations related to Chlorantraniliprole.

# Demo 4: Chronological Search
# what are the most recent tolerance thresholds for chlorantraniliprole established by the US EPA?

# Demo 5: Holistic Reasoning
# Which of the two ingredients Chlorantraniliprole and Florpyrauxifen-benzyl have stricter regulations established by the US EPA? 
# retrieve information for each substance separately.

# Demo 6: Out of Context
# Do crocodiles live in freshwaters?
# what are the most recent tolerance thresholds for flubendiamide established by the US EPA?
