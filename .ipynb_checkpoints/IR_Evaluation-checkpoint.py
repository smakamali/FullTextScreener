import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:64"
)
import json, csv, torch, jsonlines, gc
from collections import defaultdict
from datetime import datetime
import configparser

import numpy as np
from tqdm import tqdm
from sentence_transformers import util
from scipy.stats import kendalltau

from FullTextScreener import Config, QueryEngine
from llama_index.core.settings import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase, basic_auth
from transformers import BitsAndBytesConfig

# ---------- CONSTANTS ----------
EMBED_MODELS = [
    # Category 1: Non-LLM Dense Embedding Models
    "intfloat/e5-large",
    "sentence-transformers/gtr-t5-xl",
    "jinaai/jina-embeddings-v2-base-en",
    "mixedbread-ai/mxbai-embed-large-v1",
    "sentence-transformers/all-roberta-large-v1",
    "BAAI/bge-large-en-v1.5",

    # Category 2: LLM-Based or Instruction-Tuned Models
    "Linq-AI-Research/Linq-Embed-Mistral",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "intfloat/multilingual-e5-large-instruct",
    "Salesforce/SFR-Embedding-Mistral",

    # Category 3: Baseline Sentence-Embedding Models
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "facebook/contriever",
    "sentence-transformers/bert-base-nli-mean-tokens"
]

QUERY_FP = "./input/beir/queries.jsonl"
QRELS_FP = "./input/beir/test.tsv"
CONFIG_PATH = "./Config/ConfigLitScrIR.cfg"
SIM_THRESHOLD = 0.7

LIVE_OUT = "ir_eval_multi_live.json"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j_rag_poc"

def reset_vector_index(dim):
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as s:
        s.run("DROP INDEX corpus_vec IF EXISTS")
        s.run("""
            CREATE VECTOR INDEX corpus_vec
            FOR (d:Document) ON (d.embedding)
            OPTIONS {
              indexConfig: {
                vector.dimensions: $dim,
                vector.similarity_function: 'cosine'
              }
            }
        """, dim=dim)
    driver.close()

def clear_neo4j():
    with GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)) as drv, drv.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")
        idx_res = sess.run("""
            SHOW INDEXES
            YIELD name, type
            WHERE type = 'VECTOR' OR name STARTS WITH 'vector_'
            RETURN name
        """)
        for rec in idx_res:
            idx = rec["name"]
            sess.run(f"DROP INDEX {idx} IF EXISTS")
            print(f"  • dropped index {idx}")
    print("✓ Neo4j cleared (data + vector indexes)")

def save_partial(results: dict):
    with open(LIVE_OUT, "w") as f:
        json.dump(results, f, indent=2)

def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def precision_at_k(retrieved, relevant, k=2):
    return sum(1 for d in retrieved[:k] if d in relevant) / k

def mean_reciprocal_rank(retrieved, relevant, k=5):
    for idx, d in enumerate(retrieved[:k]):
        if d in relevant:
            return 1.0 / (idx + 1)
    return 0.0

def kendalls_tau(r1, r2):
    return kendalltau(r1, r2)[0]

def intra_list_diversity(embs):
    if len(embs) < 2: return 1.0
    sims = util.pytorch_cos_sim(torch.stack(embs), torch.stack(embs)).cpu().numpy()
    upper = sims[np.triu_indices(len(embs), 1)]
    return float(np.mean(1 - upper))

def safe_load_embedder(name):
    try:
        return HuggingFaceEmbedding(
            model_name=name,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "device_map": None
            }
        )
    except (RuntimeError, NotImplementedError) as e:
        print(f"⚠️  {name} failed fast path → retrying with 4-bit: {e}")
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        return HuggingFaceEmbedding(
            model_name=name,
            trust_remote_code=True,
            model_kwargs={
                "quantization_config": bnb_cfg,
                "device_map": "auto",
                "torch_dtype": torch.float16
            }
        )

print("Loading BEIR queries and qrels …")
queries = {}
with jsonlines.open(QUERY_FP) as rd:
    for obj in rd:
        queries[obj["_id"]] = obj["text"]

qrels = defaultdict(set)
with open(QRELS_FP, newline="", encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    next(tsv)
    for qid, doc_id, score in tsv:
        if float(score) > 0:
            qrels[qid].add(doc_id)

judged_queries = {qid: txt for qid, txt in queries.items() if qid in qrels}
print(f"Queries w/ judgements : {len(judged_queries):,}\n")

cfg = Config(CONFIG_PATH)
llm = MistralAI(model=cfg.mistral['model_name'], api_key=cfg.mistral['api_key'], temperature=0.0, max_tokens=1100)
Settings.llm = llm
all_results = {}

for mdl_name in EMBED_MODELS:
    print(f"\n=== Evaluating with embedding: {mdl_name} ===")
    clear_neo4j()
    embed_model = safe_load_embedder(mdl_name)
    Settings.embed_model = embed_model
    dim = len(embed_model.get_text_embedding("probe"))

    for sect in cfg.dict:
        for key in ("embeddim", "embed_dim", "vector_dim"):
            if key in cfg.dict[sect]:
                cfg.dict[sect][key] = dim

    qe = QueryEngine(llm, embed_model, cfg)
    qe._buildVectorQueryEngine(forceReindex=True)
    retriever = qe.vectorRetriever

    p10s, mrr5s, ilds, taus = [], [], [], []

    for qid, qtxt in tqdm(judged_queries.items(), desc=mdl_name[:25]):
        rel_doc_ids = list(qrels[qid])
        res_nodes = retriever.retrieve(qtxt)
        doc_ids = [n.metadata["file_name"].replace(".txt", "") for n in res_nodes]
        doc_text_top = [n.text for n in res_nodes[:10]]

        p10s.append(precision_at_k(doc_ids, rel_doc_ids, k=2))
        mrr5s.append(mean_reciprocal_rank(doc_ids, rel_doc_ids, k=5))

        emb10 = [to_tensor(embed_model.get_text_embedding(t)) for t in doc_text_top]
        ilds.append(intra_list_diversity(emb10))

        q_emb = to_tensor(embed_model.get_text_embedding(qtxt))
        sims = util.pytorch_cos_sim(q_emb, torch.stack(emb10))[0].cpu().tolist()
        taus.append(kendalls_tau(list(range(len(emb10))), [i for i, _ in sorted(enumerate(sims), key=lambda x: -x[1])]))

    all_results[mdl_name] = {
        "Average_P@10": round(float(np.mean(p10s)), 3),
        "Average_MRR@5": round(float(np.mean(mrr5s)), 3),
        "Average_ILD@10": round(float(np.mean(ilds)), 3),
        "Average_Kendalls_Tau": round(float(np.mean(taus)), 3),
        "Num_Queries": len(p10s)
    }

    save_partial(all_results)

    # after you finish evaluating a model
    # ───── utility ────────────────────────────────────────────────────────────
    def report_gpu(tag: str = ""):
        """Pretty-print current CUDA memory usage (in MiB)."""
        alloc    = torch.cuda.memory_allocated()  / 2**20
        reserved = torch.cuda.memory_reserved()   / 2**20
        print(f"{tag:<18} | alloc: {alloc:7.1f} MB  |  reserved: {reserved:7.1f} MB")
    
    # ───── inside your for-model loop ─────────────────────────────────────────
    report_gpu("before cleanup")       # <─ add this just *before* the cleanup block
    
    # ---------- cleanup block ----------
    if hasattr(embed_model, "to"):
        embed_model.to("meta")              # detach weights
    
    Settings.embed_model = None
    for v in ("embed_model", "qe", "retriever"):
        if v in locals():
            del locals()[v]
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # ------------------------------------
    
    report_gpu("after cleanup")        # <─ and this right after the cleanup


ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out = f"ir_eval_multi_{ts}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nEvaluation finished →", os.path.abspath(out))
print("Live snapshot (every model) →", os.path.abspath(LIVE_OUT))