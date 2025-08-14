# Meta-Analysis of RAG Output for Query Optimization Papers

## Overview
This project performs meta-analysis on the output of a Retrieval-Augmented Generation (RAG) pipeline that answers questions about research papers in query optimization. The analysis includes:
- Data processing of RAG JSON outputs
- Distribution analysis of answers to 25 research questions
- Trend analysis of machine learning task types
- Benchmark/dataset usage analysis
- Visualization of relationships between research dimensions

## Notebook Structure

### 1. Data Import and Transformation
- Loads RAG output from `latest_output.json`
- Parses question answers into structured format (QuestionText, ShortAnswer, Reasoning, Evidence)
- Flattens nested JSON structure
- Converts data to pandas DataFrame

### 2. Distribution Analysis
- Visualizes answer distributions for all 25 questions
- Generates bar charts showing frequency of answers per question

### 3. Task Type Trend Analysis
- Examines distribution of ML task types over time
- Analyzes relationships between task types and:
  - Learning paradigms
  - Model architectures
  - Benchmarks used
  - Uncertainty evaluation techniques
  - Out-of-distribution (OOD) handling
  - Encoding techniques

### 4. Dataset/Benchmark Analysis
- Tracks benchmark usage trends over time
- Shows distribution of popular benchmarks across papers

## Key Visualizations
1. **Answer Distributions**: Bar charts for each research question
2. **Task Type Trends**: 
   - Temporal distribution (stacked area charts)
   - Relationship with learning paradigms (grouped bar charts)
   - Architecture usage patterns
3. **Benchmark Analysis**:
   - Usage frequency across papers
   - Adoption trends over time (stacked area charts)
4. **Technique Relationships**:
   - Uncertainty evaluation by task type
   - OOD handling methods
   - Encoding technique preferences

## Dependencies
- Python 3.7+
- Required packages:
  ```bash
  pandas numpy matplotlib seaborn json re