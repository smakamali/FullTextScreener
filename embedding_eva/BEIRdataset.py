#!/usr/bin/env python3
# Combined BEIR exporter + evaluation set creator

import os
import json
import csv
import pathlib
import random
import re
from collections import defaultdict
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

######################################
# Settings
######################################

# BEIR download
DATASET_NAME = "nq"          # e.g., "nq", "trec-covid", "scifact"
SPLIT = "test"               # "train", "dev", or "test"

# Target eval corpus settings
TARGET_N = 8000              # total number of documents after expansion
random.seed(42)              # for reproducibility

# Paths
BEIR_DOWNLOAD_DIR = "./beir_datasets"
EXPORT_DIR = f"{DATASET_NAME}_beir_tsv"
OUT_DIR = "beir_v2"

######################################
# Functions
######################################

def download_and_save_beir_dataset():
    zip_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    data_path = util.download_and_unzip(zip_url, BEIR_DOWNLOAD_DIR)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=SPLIT)

    os.makedirs(EXPORT_DIR, exist_ok=True)

    queries_tsv = os.path.join(EXPORT_DIR, f"queries.{SPLIT}.tsv")
    qrels_tsv = os.path.join(EXPORT_DIR, f"qrels.{SPLIT}.tsv")
    coll_tsv = os.path.join(EXPORT_DIR, "collection.tsv")

    # Save queries
    with open(queries_tsv, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            f.write(f"{qid}\t{qtext}\n")
    print(f"✓ saved {len(queries):,} queries → {queries_tsv}")

    # Save qrels
    lines = 0
    with open(qrels_tsv, "w", encoding="utf-8") as f:
        for qid, doc_dict in qrels.items():
            for doc_id, rel in doc_dict.items():
                f.write(f"{qid} 0 {doc_id} {rel}\n")
                lines += 1
    print(f"✓ saved {lines:,} qrel lines → {qrels_tsv}")

    # Save collection (only documents mentioned in qrels)
    rel_doc_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    with open(coll_tsv, "w", encoding="utf-8") as f:
        for doc_id in rel_doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "").replace("\t", " ").strip()
            text  = doc.get("text", "").replace("\t", " ").strip()
            joined = f"{title}. {text}".strip(". ")
            f.write(f"{doc_id}\t{joined}\n")
    print(f"✓ saved {len(rel_doc_ids):,} passages → {coll_tsv}")

    print("\n✅ BEIR dataset prepared!")

def create_small_eval_set():
    # Paths
    corpus_fp = os.path.join(EXPORT_DIR, "collection.tsv")
    qrels_fp = os.path.join(EXPORT_DIR, f"qrels.{SPLIT}.tsv")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect relevant document IDs
    rel_ids = set()
    with open(qrels_fp, newline="", encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter="\t")
        for row in tsv:
            qid, _, doc_id, _ = row
            rel_ids.add(doc_id)

    print(f"Relevant IDs from qrels.{SPLIT}.tsv: {len(rel_ids):,}")
    extras_n = TARGET_N - len(rel_ids)
    print(f"Random non-relevant docs needed: {extras_n:,}\n")

    extras_reservoir = []
    extras_seen = 0
    written_ids = set()

    # Count total lines
    with open(corpus_fp, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Stream corpus and select extras
    with open(corpus_fp, encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Scanning corpus"):
            doc_id, text = line.rstrip("\n").split("\t", 1)

            if doc_id in rel_ids:
                out_fp = os.path.join(OUT_DIR, f"{doc_id}.txt")
                if doc_id not in written_ids:
                    with open(out_fp, "w", encoding="utf-8") as out:
                        out.write(text)
                    written_ids.add(doc_id)
            else:
                extras_seen += 1
                if len(extras_reservoir) < extras_n:
                    extras_reservoir.append((doc_id, text))
                else:
                    j = random.randrange(extras_seen)
                    if j < extras_n:
                        extras_reservoir[j] = (doc_id, text)

    # Write sampled non-relevant extras
    for doc_id, text in extras_reservoir:
        if doc_id in written_ids:
            continue
        out_fp = os.path.join(OUT_DIR, f"{doc_id}.txt")
        with open(out_fp, "w", encoding="utf-8") as out:
            out.write(text)
        written_ids.add(doc_id)

    # Final counts
    print(f"\n✔ Total files written : {len(written_ids):,}")
    print(f"• Relevant passages   : {len(rel_ids):,}")
    print(f"• Random extras       : {len(written_ids) - len(rel_ids):,}")

######################################
# Final Main
######################################

def main():
    download_and_save_beir_dataset()
    create_small_eval_set()

if __name__ == "__main__":
    main()
