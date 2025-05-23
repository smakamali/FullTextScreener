# ir_eval_multi.py
# --------------------------------------------------------
# Evaluate the same corpus with many embedding models.
# --------------------------------------------------------
import os, json, csv, torch, jsonlines
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sentence_transformers import util
from scipy.stats import kendalltau

from FullTextScreener import Config, QueryEngine
from llama_index.core.settings import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase, basic_auth

# ---------- CONSTANTS ----------
EMBED_MODELS = [
    # ── already working for you ─────────────────────────
    "Alibaba-NLP/gte-multilingual-base",
    "intfloat/multilingual-e5-large-instruct",
    "Lajavaness/bilingual-embedding-large",

    # ── lightweight English & multilingual baselines ───
    "sentence-transformers/all-MiniLM-L6-v2",                   # EN  | 90 MB
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multi | 230 MB
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # Multi | 440 MB

    # ── e5 family (English & multilingual) ──────────────
    "intfloat/e5-base",                         # EN  | 440 MB
    "intfloat/multilingual-e5-base",            # Multi | 490 MB

    # ── current strong English retriever ───────────────
    "BAAI/bge-base-en-v1.5",                    # EN  | 470 MB
]


QUERY_FP = "beir/queries.jsonl"
QRELS_FP = "beir/test.tsv"
CONFIG_PATH = "./Config/ConfigLitScrIR.cfg"
SIM_THRESHOLD = 0.7          # still used for ILD cosine diversity

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
NEO4J_URI      = "bolt://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j_rag_poc"

all_results = {}

LIVE_OUT = "ir_eval_multi_live.json"   # <- will be overwritten after each model

def reset_vector_index(dim):
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    with driver.session() as s:
        # 1 Drop any existing index (ignore if it doesn't exist)
        s.run("DROP INDEX corpus_vec IF EXISTS")

        # 2 Create a fresh cosine-similarity index with the right dimension
        s.run(
            """
            CREATE VECTOR INDEX corpus_vec
            FOR (d:Document) ON (d.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            dim=dim,
        )
    driver.close()


def clear_neo4j():
    """
    Remove all data *and* drop every vector index, regardless of APOC.
    Works on Neo4j 4.x / 5.x without extra plugins.
    """
    with GraphDatabase.driver(
        NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as drv, drv.session() as sess:

        # 1. delete all nodes/relationships
        sess.run("MATCH (n) DETACH DELETE n")

        # 2. list every index; keep only VECTOR ones (or any you recognise)
        idx_res = sess.run("""
            SHOW INDEXES
            YIELD name, type
            WHERE type = 'VECTOR' OR name STARTS WITH 'vector_'
            RETURN name
        """)
        for rec in idx_res:
            idx = rec["name"]
            sess.run(f"DROP INDEX `{idx}` IF EXISTS")
            print(f"  • dropped index {idx}")

    print("✓ Neo4j cleared (data + vector indexes)")

# ---------- RESULT DICT & LIVE-FILE ----------
def save_partial(results: dict):
    """Write current results to disk so you can inspect mid-run."""
    with open(LIVE_OUT, "w") as f:
        json.dump(results, f, indent=2)

# ---------- tensor helper ----------
def to_tensor(x):
    """Ensure x is a 1-D float32 torch tensor (handles list / numpy / tensor)."""
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def embed_dim(model) -> int:
    """Return the *real* output length used in Neo4j."""
    return len(model.get_text_embedding("dimension_probe"))

# ---------- METRIC HELPERS ----------
def precision_at_k(retrieved, relevant, k=10):
    return sum(1 for d in retrieved[:k] if d in relevant) / k

def mean_reciprocal_rank(retrieved, relevant, k=5):
    for idx, d in enumerate(retrieved[:k]):
        if d in relevant:
            return 1.0 / (idx + 1)
    return 0.0

def kendalls_tau(r1, r2):                   # r1, r2 = rank lists
    return kendalltau(r1, r2)[0]

def intra_list_diversity(embs):             # list[Tensor]
    if len(embs) < 2: return 1.0
    sims = util.pytorch_cos_sim(
        torch.stack(embs), torch.stack(embs)
    ).cpu().numpy()
    upper = sims[np.triu_indices(len(embs), 1)]
    return float(np.mean(1 - upper))

# ---------- LOAD BEIR QUERIES & QRELS ----------
print("Loading BEIR queries and qrels …")
queries = {}
with jsonlines.open(QUERY_FP) as rd:
    for obj in rd: 
        queries[obj["_id"]] = obj["text"]

qrels = defaultdict(set)
with open(QRELS_FP, newline="", encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    next(tsv)                      # skip header
    for qid, doc_id, score in tsv:
        if float(score) > 0:
            qrels[qid].add(doc_id)

judged_queries = {qid: txt for qid, txt in queries.items() if qid in qrels}
print(f"Queries w/ judgements : {len(judged_queries):,}\n")

# ---------- INIT LLM ONCE (unchanged) ----------
cfg   = Config(CONFIG_PATH)
llm   = MistralAI(model=cfg.mistral['model_name'],
                  api_key=cfg.mistral['api_key'],
                  temperature=0.0, max_tokens=1100)
Settings.llm = llm

# ---------- RESULT DICT ----------
all_results = {}

# ---------- MAIN LOOP OVER EMBEDDING MODELS ----------
for mdl_name in EMBED_MODELS:
    print(f"\n=== Evaluating with embedding: {mdl_name} ===")
    clear_neo4j() 
    embed_model = HuggingFaceEmbedding(model_name=mdl_name,
                                       trust_remote_code=True)
    Settings.embed_model = embed_model
    dim = len(embed_model.get_text_embedding("probe"))

    # Patch *every* embeddim in cfg **before** QueryEngine is built
    for sect in cfg.dict:
        for key in ("embeddim", "embed_dim", "vector_dim"):
            if key in cfg.dict[sect]:
                cfg.dict[sect][key] = dim
    
    # (Re)build index with current embedder
    qe = QueryEngine(llm, embed_model, cfg)
    qe._buildVectorQueryEngine(forceReindex=True)
    retriever = qe.vectorRetriever

    p10s, mrr5s, ilds, taus = [], [], [], []

    for qid, qtxt in tqdm(judged_queries.items(), desc=mdl_name[:25]):
        rel_doc_ids = list(qrels[qid])

        res_nodes    = retriever.retrieve(qtxt)
        doc_ids      = [n.metadata["file_name"].replace(".txt","") for n in res_nodes]  # adjust if you saved "doc_id"
        doc_text_top = [n.text for n in res_nodes[:10]]

        # -- metrics --
        p10s.append(precision_at_k(doc_ids, rel_doc_ids, k=10))
        mrr5s.append(mean_reciprocal_rank(doc_ids, rel_doc_ids, k=5))

        emb10 = [to_tensor(embed_model.get_text_embedding(t))  # ← use helper
         for t in doc_text_top]
        ilds.append(intra_list_diversity(emb10))

        q_emb = to_tensor(embed_model.get_text_embedding(qtxt))
        sims   = util.pytorch_cos_sim(q_emb, torch.stack(emb10))[0].cpu().tolist()
        taus.append(kendalls_tau(list(range(len(emb10))),
                                 [i for i,_ in sorted(enumerate(sims), key=lambda x:-x[1])]))

    # -- store per-model summary --
    all_results[mdl_name] = {
        "Average_P@10"      : round(float(np.mean(p10s)), 3),
        "Average_MRR@5"     : round(float(np.mean(mrr5s)), 3),
        "Average_ILD@10"    : round(float(np.mean(ilds)), 3),
        "Average_Kendalls_Tau": round(float(np.mean(taus)), 3),
        "Num_Queries"       : len(p10s)
    }
    save_partial(all_results)

# ---------- SAVE ONE JSON ----------
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out = f"ir_eval_multi_{ts}.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nEvaluation finished →", os.path.abspath(out))
print("Live snapshot (every model) →", os.path.abspath(LIVE_OUT))
