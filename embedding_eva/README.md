# embedding_eva (evaluation scripts)

Scripts used to choose the embedding and language models for the pipeline. They are research scripts, not part of the screening run.

| Script | What it does |
|---|---|
| `BEIRdataset.py` | Exports a document set in BEIR format and builds an evaluation set |
| `WikiQADataset.py` | Builds an evaluation set from the WikiQA dataset |
| `IR_Evaluation.py` | Compares retrieval quality of several embedding models on the same corpus |
| `LLM_Evaluation.py` | Runs the screening pipeline with several LLMs and scores the answers |

**Running them:**
- **Working directory:** run from the repository root with the root on `PYTHONPATH` (for example `python -m embedding_eva.IR_Evaluation`), because they import `FullTextScreener`.
- **Configs:** they expect their own config files (`Config/ConfigLitScrIR.cfg`, `Config/ConfigLitScrEval.cfg`). Create them from `Config/ConfigLitScrTemp.cfg`; like every live config, they are gitignored.
- **Extra packages:** `beir`, `jsonlines`, `rouge-score`, `wikipedia`, `datasets`, `scikit-learn`, `scipy`, listed separately in `Requirements/requirements_nv.txt`.
