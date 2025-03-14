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
pip install -r requirements_wv.txt
```

3. Create the configuration file:
```bash
cp ./Config/ConfigLitScrTemp.cfg ./Config/ConfigLitScr.cfg
```

4. Edit the configuration file:
   - Add your HuggingFace and Mistral API keys
   - Adjust other settings as needed


## Setting up Neo4j Docker container

Run the following command. Make sure to use the config parameters under the `neo4j` section of the config file in the following command:

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
└── requirements_wv.txt
```

## Configuration

The `ConfigLitScr.cfg` file contains all the necessary configuration parameters. Key configuration sections include:

### General Settings
```
[general]
appVersion = 0.8.0
forceReindex = False
enableLogging = True
showProgress = True
```

### LLM Model Settings
```
[llm-model]
service = huggingface # alternatively can be set to `mistral`
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
logDir = ./logs/
logFile = processing_log.txt
dataPath = ../neo4j_vol1/data
pluginsPath = ../neo4j_vol1/plugins
```

### Agent Settings
```
[agent]
enable_agent = True
max_iterations = 10 #sets number of times to try to answer the question
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

1. **Metadata CSV File**: A CSV file containing metadata about the papers to be analyzed. The file must include:
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

The questions are defined in the `questions` dictionary in the `FullTextScreener.py` file. You can modify or add questions by editing this dictionary.

## Usage

Run the script with Python:

```bash
python FullTextScreener.py
```

The script will:
1. Load configuration settings
2. Process each paper in the metadata CSV file
3. Generate answers to the predefined questions
4. Save the results in a JSON file

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "paper_id": {
    "metadata": {
      "paper_id": "paper001",
      "pdf_filename": "paper001.pdf",
      "title": "Example Paper Title",
      "authors": "John Doe",
      "year": "2023"
    },
    "answers": {
      "Q1": {
        "QuestionText": "Does the study provide any new definitions for robustness?",
        "ShortAnswer": "Yes",
        "Reasoning": "The paper introduces a novel definition of robustness in the context of query optimization.",
        "Evidence": "On page 3, the authors state: 'We define robustness as...'"
      },
      "Q2": {
        "QuestionText": "How does the study define robustness or risk (implicitly or explicitly)?",
        "ShortAnswer": "A query optimizer is robust if it produces consistent execution plans for similar queries",
        "Reasoning": "The authors explicitly define robustness in terms of plan stability.",
        "Evidence": "The definition appears on page 4: 'A query optimizer is considered robust if...'"
      }
    }
  }
}
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

MIT License

Copyright (c) 2025 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This tool uses the following open-source libraries:
- llama-index for document indexing and retrieval
- HuggingFace Transformers for language and embedding models
- Mistral AI for LLMs
- Neo4j for graph database storage
- PyTorch for deep learning operations
- pdfplumber for PDF processing
