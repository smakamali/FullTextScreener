# FullTextScreener - Information Retrieval Evaluation

This project evaluates various embeddings for information retrieval (IR) tasks using the BEIR dataset. The goal is to determine which embedding works best for screening literature, specifically in the context of **cardinality estimation papers**. This README guides you through setting up the experiment, running the dataset creation, configuring the evaluation, and executing the IR evaluation.

## Steps to Run the Experiment

### 1. Set Up the Environment

Before you begin, make sure you have Python 3.11 or higher installed. It’s recommended to create a **virtual environment** using Conda to manage dependencies.

```bash
conda create --name ir-eval python=3.11
conda activate ir-eval
```

Once the environment is activated, install the necessary dependencies.

```bash
pip install -r requirements_nv.txt
```

### 2. Create the BEIR Dataset

The **BEIR dataset** is required to perform the information retrieval evaluation. Run the `BEIRdataset.py` script to download and prepare the dataset.

#### Run the Script:

```bash
python3 BEIRdataset.py
```

This script does the following:

* Downloads the specified BEIR dataset (e.g., "nq").
* Creates a subset of the dataset with relevant and non-relevant documents.
* Saves the dataset as `.tsv` and `.txt` files under `./input/beir_v2`.

### 3. Set Up Neo4j Docker Container

To set up the Neo4j database for the IR evaluation, you'll need to run a Docker container with Neo4j and the APOC plugin enabled.

Run the following command to start the Neo4j container:

```bash
docker run --name neo4j-apoc \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/neo4j_rag_poc \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v ./neo4j_vol1/data:/data \
    -v ./neo4j_vol1/plugins:/plugins \
    neo4j:latest
```

**Important:**

* Ensure that the parameters in this command match the configuration settings in the `neo4j` section of your config file.
* The container is set to use the `neo4j_rag_poc` password, so make sure that the credentials in the config file (`neo4j` section) match this.
* The Neo4j container exposes the web interface on port `7474` and the database interface on port `7687`.

Once the Neo4j container is up and running, the system will be ready to store and retrieve documents from the database as part of the evaluation.

### 4. Configure the IR Evaluation

The **configuration file** contains various settings, including paths to the dataset, LLM models, and other parameters for the evaluation. Below is an example configuration file, and it should be saved as `ConfigLitScrIR.cfg`.

#### Configuration Specifications:

Here are the most important sections in the configuration file:

```ini
[general]
appversion = 0.8.0
forcereindex = False
enablelogging = True
showprogress = True

[llm-model]
service = mistral
max_new_tokens = 1100
temperature = 0.0
do_sample = False

[huggingface]
api_key = YOUR_HUGGINGFACE_API_KEY
model_name = meta-llama/Llama-3.1-8B-Instruct
embed_model = intfloat/multilingual-e5-small
embeddim = 1024

[mistral]
api_key = YOUR_MISTRAL_API_KEY
model_name = mistral-large-latest
embeddim = 1024

[neo4j]
username = neo4j
password = neo4j_rag_poc
url1 = bolt://localhost:7687
url2 = http://localhost:7474
containername = neo4j-apoc

[dir-structure]
inputdir = ./input/beir_v2
input_pdf_folder = ./input/pdfs
metadatafile = ./input/metadata.csv
outputdir = ./output
outputfile = ./output/output.json
logdir = ./logs/
logfile = processing_log.txt
datapath = ../neo4j_vol1/data
pluginspath = ../neo4j_vol1/plugins

[agent]
enable_agent = True
max_iterations = 10

[retriever]
vectortopk = 10
cutoffscore = 0.5

[nodeparser]
nodeparsertype = static
batchsize = 1
chunk_size = 1200
chunk_overlap = 200
buffersize = 1
breakpointpercentilethreshold = 95
```

#### Key Fields to Fill:

* **HuggingFace and Mistral API keys**: Provide your API keys under the `huggingface` and `mistral` sections.
* **inputdir**: Set this to `./input/beir_v2` as it points to the directory where the dataset will be saved.

### 4. Run the IR Evaluation

Once the dataset is ready and the configuration is set up, you can run the **IR evaluation script** (`IR_Evaluation.py`).

```bash
python3 IR_Evaluation.py
```

This will:

1. Load the dataset and configuration.
2. Evaluate multiple embeddings on the BEIR dataset using the specified parameters in the config file.
3. Output the evaluation results to a JSON file.

### 5. Configurable Parameters

The configuration file allows for the following customizations:

* **Neo4j database connection**: Set the correct credentials and URLs under the `[neo4j]` section if you're using a local Neo4j instance.
* **Retriever parameters**: Adjust the `vectortopk` and `cutoffscore` under the `[retriever]` section to fine-tune the retrieval process.
* **Chunking and processing**: The `[nodeparser]` section lets you modify batch size, chunk size, overlap, and other parameters for document processing.

### 6. Output

The evaluation results are stored in the specified `outputdir`. A JSON file with evaluation metrics will be generated, containing metrics such as:

* **Precision\@10**
* **Mean Reciprocal Rank at 5 (MRR\@5)**
* **Intra-list Diversity (ILD)**
* **Kendall’s Tau**

Example output format:

```json
{
  "intfloat/e5-large": {
    "Average_P@10": 0.85,
    "Average_MRR@5": 0.72,
    "Average_ILD@10": 0.63,
    "Average_Kendalls_Tau": 0.78,
    "Num_Queries": 500
  }
}
```

---

## Troubleshooting

If you encounter issues:

1. **Missing dataset files**: Ensure the `BEIRdataset.py` script ran successfully and the data is saved under `./input/beir_v2`.
2. **API rate limits**: If you hit rate limits from HuggingFace or Mistral, consider using a paid plan or adjusting the `batchsize` in the config file.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---