# FullTextScreener

A literature screening tool for automated analysis of scientific papers. This tool uses large language models to extract structured information from PDF documents based on predefined questions.

## Version

Current version: 0.8.0

## Overview

FullTextScreener is a Python application designed to automate the process of screening scientific literature. It uses natural language processing and vector-based retrieval to analyze PDF documents and extract relevant information according to a set of predefined questions.

The tool performs the following tasks:
- Loads PDF documents from a specified directory
- Indexes the content using vector embeddings
- Processes each document with a large language model (LLM)
- Extracts structured answers to predefined questions
- Outputs the results in JSON format

## Features

- **PDF Processing**: Automatic extraction and indexing of PDF content
- **Vector-based Retrieval**: Efficient document retrieval using vector embeddings
- **LLM Integration**: Support for HuggingFace and Mistral AI models
- **Structured Output**: JSON-formatted results for easy analysis
- **Configurable**: Flexible configuration options for different use cases

## Prerequisites

- Python 3.11 or higher
- Docker (for Neo4j database)
- GPU with CUDA support (recommended for optimal performance if using open source LLMs from HuggingFace)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/smakamali/ChatWithDocs.git
cd ChatWithDocs
```

2. Install dependencies:
```bash
pip install -r Requirements/requirements_nv.txt
```
`Requirements/requirements_wv.txt` pins the exact versions that ran the survey extractions (Python 3.11).

3. Create the configuration file:
```bash
cp ./Config/ConfigLitScrTemp.cfg ./Config/ConfigLitScr.cfg
```

4. Edit the configuration file:
   - Choose the `service` (`openrouter`, `mistral` or `huggingface`) and add its API key, plus a HuggingFace token for the embedding model
   - Adjust other settings as needed

   Every `Config/*.cfg` except the `*Temp.cfg` templates is gitignored, because live configs hold API keys. Never commit them.


## Setting up Neo4j Docker container

Run the following command. Make sure the parameters used in the following command are consistent with the config parameters under the `neo4j` section of the config file.

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

## Directory Structure

Ensure your project has the following directory structure:
```
fulltext-screener/
├── Config/
│   ├── ConfigLitScrTemp.cfg
│   └── ConfigLitScr.cfg
├── Questionnaires/
│   └── (questionnaire JSON files)
├── Modules/
│   ├── Readers/
│   │   └── (Custom reader modules)
│   └── Tools/
│       └── (Helper tools)
├── input/
│   ├── pdfs/
│   │   └── (PDF files)
│   ├── selected/
│   │   └── (Used for processing)
│   └── metadata.csv
├── output/
│   └── (Output files)
├── logs/
│   └── (Log files)
├── FullTextScreener.py
└── Requirements/
    ├── requirements_nv.txt
    └── requirements_wv.txt
```

## Configuration

The `ConfigLitScr.cfg` file contains all the necessary configuration parameters. Key configuration sections include:

### General Settings
```
[general]
appVersion = 0.9.1
forceReindex = False
enableLogging = True
showProgress = True
skip_existing = False   # True: resume, reusing complete results made with the same questionnaire
```

### LLM Model Settings
```
[llm-model]
service = openrouter # or `mistral` or `huggingface`
max_new_tokens = 1024
temperature = 0.0
do_sample = False
```

### HuggingFace Settings
```
[huggingface]
api_key = YOUR_API_KEY_HERE
model_name = Qwen/Qwen2.5-14B-Instruct-1M
embed_model = Alibaba-NLP/gte-multilingual-base
embedDim = 768
```

### OpenRouter Settings
```
[openrouter]
api_key = YOUR_API_KEY_HERE
model_name = mistralai/mistral-large-2512
embed_model = Alibaba-NLP/gte-Qwen2-7B-instruct
embedDim = 3584
# context_window = 262144     # optional; otherwise read from OpenRouter's model listing
# embed_device = cuda         # optional: cpu (default), cuda or auto
# embed_quantization = 8bit   # optional, GPU only: none, fp16, 8bit or 4bit
```
If the context window cannot be looked up and `context_window` is not set, the run stops with an error instead of falling back to llama-index's 3,900-token default, which would truncate the retrieved context.

### Mistral Settings
```
[mistral]
api_key = YOUR_API_KEY_HERE
model_name = mistral-large-latest
embed_model = Alibaba-NLP/gte-multilingual-base
embedDim = 768
```

### Neo4j Database Settings
```
[neo4j]
username = neo4j
password = neo4j_rag_poc
url1 = bolt://localhost:7687
url2 = http://localhost:7474
containerName = neo4j-apoc
```

### Directory Structure Settings
```
[dir-structure]
inputDir = ./input/selected
input_pdf_folder = ./input/pdfs
metadataFile = ./input/metadata.csv
outputDir = ./output
outputFile = ./output/output.json
questions_file_path = ./Questionnaires/questions_v2.1.json
logDir = ./logs/
logFile = processing_log.txt
dataPath = ../neo4j_vol1/data
pluginsPath = ../neo4j_vol1/plugins
```

### Agent Settings
```
[agent]
enable_agent = True
max_iterations = 10          # reasoning steps per question; at the limit the agent writes a final answer
memory_token_limit = 100000  # per-paper memory; capped to fit the model's context window
```

### Retriever Settings
```
[retriever]
vectorTopK = 10
cutoffScore = 0.5
```

### Node Parser Settings
```
[nodeparser]
nodeParserType = static
batchSize = 1
chunk_size = 1000
chunk_overlap = 50
bufferSize = 1
breakpointPercentileThreshold = 95
```

## Input Data Format

The tool expects the following input data:

1. **Metadata CSV File**: A CSV file containing metadata about the papers to be analyzed, stored in `metadataFile` path. The file must include:
   - `paper_id`: A unique identifier for each paper
   - `pdf_filename`: The name of the PDF file associated with the paper

2. **PDF Files**: The PDF documents to be analyzed, stored in the `input_pdf_folder` directory.

Example metadata CSV format:
```csv
paper_id,pdf_filename,title,authors,year
paper001,paper001.pdf,Example Paper Title,John Doe,2023
paper002,paper002.pdf,Another Paper Title,Jane Smith,2024
```
## Customizing Questions

The questions are read from the JSON file given by `questions_file_path` in `[dir-structure]`: an object mapping question ids (`Q1`, `Q2`, ...) to question texts. `Questionnaires/` holds the survey's versions (see `Questionnaires/README.md`). If the key is missing, `Questionnaires/questions_v1.json` is used.

## Usage

Run the script with Python:

```bash
python FullTextScreener.py                      # uses ./Config/ConfigLitScr.cfg
python FullTextScreener.py --config path/to.cfg # or set the FTS_CONFIG environment variable
```
Without `--config` or `FTS_CONFIG`, the script uses `./Config/ConfigLitScr.cfg`, or the earlier default `./Config/ConfigLitScrInUse.cfg` if only that exists.

The script will:
1. Load configuration settings
2. Process each paper in the metadata CSV file
3. Generate answers to the predefined questions
4. Save each paper's result in `<outputDir>/individual/<pdf_filename>.json`, and all results in `outputFile`

**Failures and resuming:**
- **A question that fails:** a reply that cannot be parsed is asked once more. If the question still fails, or its API calls keep failing, it is recorded with an `error` key and the run continues. The paper is then saved as `<pdf_filename>.partial.json`.
- **Errors that stop the run:** errors retrying cannot fix (out of credit, invalid key, unknown model) and setup errors stop the run. `outputFile` is written however the run ends, including on Ctrl-C.
- **Resuming:** rerun with `skip_existing = True`. Complete results made with the same questionnaire are reused; partial, unreadable or outdated results are redone.

## Output Format

The tool generates a JSON file with the following structure. Results are keyed by `pdf_filename` (versions before 0.9 used the numeric `paper_id`). `QuestionText` is copied from the questionnaire. The model writes `Evidence` and `Reasoning` before `ShortAnswer`, so that it reasons before answering.

```json
{
  "paper001.pdf": {
    "metadata": {
      "paper_id": "paper001",
      "pdf_filename": "paper001.pdf",
      "title": "Example Paper Title",
      "authors": "John Doe",
      "year": "2023"
    },
    "questionnaire_sha256": "3f1c…",
    "answers": {
      "Q1": {
        "QuestionText": "Does the study provide any new definitions for robustness?",
        "Evidence": "On page 3, the authors state: 'We define robustness as...'",
        "Reasoning": "The paper introduces a novel definition of robustness in the context of query optimization.",
        "ShortAnswer": "Yes"
      },
      "Q2": {
        "QuestionText": "How does the study define robustness or risk (implicitly or explicitly)?",
        "Evidence": "The definition appears on page 4: 'A query optimizer is considered robust if...'",
        "Reasoning": "The authors explicitly define robustness in terms of plan stability.",
        "ShortAnswer": "A query optimizer is robust if it produces consistent execution plans for similar queries"
      }
    }
  }
}
```


## Tests

Offline tests cover answer parsing, retries, the context-window lookup, the memory cap, resuming and config resolution. They need no API key, Neo4j or GPU:
```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

Common issues and solutions:

1. **GPU Memory Issues**: If you encounter GPU memory errors, try:
   - Reducing the batch size in the configuration
   - Using a smaller model
   - Processing fewer papers at once

2. **API Rate Limits**: If you hit API rate limits:
   - The tool will automatically retry with exponential backoff
   - Consider using a paid API tier for higher limits

3. **Neo4j Connection Issues**: If you cannot connect to Neo4j:
   - Ensure Docker is running
   - Check that the Neo4j container is started
   - Verify the Neo4j credentials in the configuration file

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This tool uses the following open-source libraries:
- llama-index for document indexing and retrieval
- HuggingFace Transformers for language and embedding models
- Mistral AI for LLMs
- Neo4j for graph database storage
- PyTorch for deep learning operations
- pdfplumber for PDF processing
