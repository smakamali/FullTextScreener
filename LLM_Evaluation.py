#!/usr/bin/env python3

import os
import csv
import json
import re
import time
import multiprocessing

import pandas as pd
from datasets import load_dataset
import huggingface_hub
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import mauve
import numpy as np
from collections import Counter

from FullTextScreener import Config, literature_screening, setLogging, setChunkingMethod

################################ Helper Functions ################################

def sanitize_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join([str(i) for i in x])
    return str(x)

def compute_exact_match_micro_f1(refs, gens):
    total_tp = total_pred = total_ref = 0
    for ref, gen in zip(refs, gens):
        ref_tokens = ref.split()
        gen_tokens = gen.split()
        ref_counts = Counter(ref_tokens)
        gen_counts = Counter(gen_tokens)
        overlap = sum((ref_counts & gen_counts).values())
        total_tp += overlap
        total_pred += sum(gen_counts.values())
        total_ref += sum(ref_counts.values())
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def normalize_answers(data):
    for item in data:
        for key, entry in item.items():
            answers = entry.get("answers", {})
            for qid, ans in answers.items():
                if isinstance(ans, str):
                    parsed = parse_answer_string(ans)
                    answers[qid] = parsed
    return data

def parse_answer_string(answer_str):
    short_answer, reasoning, evidence = "", "", "Not Applicable"
    reasoning_split = re.split(r'\s*Reasoning:\s*', answer_str, maxsplit=1)
    if len(reasoning_split) == 2:
        short_answer = reasoning_split[0].strip()
        rest = reasoning_split[1]
        evidence_split = re.split(r'\s*Evidence:\s*', rest, maxsplit=1)
        if len(evidence_split) == 2:
            reasoning = evidence_split[0].strip()
            evidence = evidence_split[1].strip()
        else:
            reasoning = rest.strip()
    else:
        short_answer = answer_str.strip()
    return {
        "QuestionText": "Unknown",
        "ShortAnswer": short_answer,
        "Reasoning": reasoning,
        "Evidence": evidence
    }

################################ Modularized Main Functions ################################

def setup_environment(configPath):
    config = Config(configPath)
    huggingFaceToken = config.huggingface.get('api_key')
    huggingface_hub.login(huggingFaceToken)
    print("Successfully logged into Hugging Face!\n")

    logger = setLogging(config)
    setChunkingMethod(config)

    return config

def load_wiki_dataset(max_docs=200):
    print("Loading dataset...")
    dataset = load_dataset("microsoft/wiki_qa")
    train_data = dataset['train']
    wiki_data = {}
    for example in train_data:
        if example["label"] != 1:
            continue
        title = example["document_title"]
        wiki_data.setdefault(title, []).append((example["question"], example["answer"]))
    print(f"Loaded {len(wiki_data)} documents.\n")
    return dict(list(wiki_data.items())[:max_docs])

def run_screening(config, wiki_data, model_runs):
    metadata_csv_path = config.dir_structure['metadatafile']
    input_pdf_folder = config.dir_structure['input_pdf_folder']

    for model_name, output_json in model_runs:
        print(f"\n=== Running screening with model: {model_name} ===")
        config.huggingface['model_name'] = model_name

        for idc, (title, qa_pairs) in enumerate(wiki_data.items()):
            print(f"Evaluating document #{idc}: {title}")

            meta = {
                'paper_id': idc,
                'pdf_filename': f"{title}.txt",
                'title': title,
                'authors': 'N/A',
                'year': 'N/A'
            }
            os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)
            with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=meta.keys())
                writer.writeheader()
                writer.writerow(meta)

            questions = {f"Q{idx}": q for idx, (q, _) in enumerate(qa_pairs)}
            literature_screening(
                metadata_csv_path,
                questions,
                config.configPath,
                input_pdf_folder,
                output_json,
                append_mode=True
            )

def normalize_outputs(model_runs):
    normalized_paths = []
    for model_name, output_path in model_runs:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalized_data = normalize_answers(data)
        normalized_path = output_path.replace('output/', 'output/normalized_')
        
        with open(normalized_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2)
        
        normalized_paths.append(normalized_path)
        print(f"Normalized {output_path} -> {normalized_path}")
    return normalized_paths

def evaluate_models(normalized_paths, model_labels, actual_answers):
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = {}

    for model_name, normalized_path in zip(model_labels, normalized_paths):
        print(f"\nEvaluating {model_name}...")
        with open(normalized_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        generated_answers = []
        for idx, item in enumerate(data):
            answers = item.get(str(idx), {}).get('answers', {})
            for val in answers.values():
                generated_answers.append(sanitize_text(val.get("ShortAnswer", "")))

        assert len(generated_answers) == len(actual_answers), \
            f"Mismatch: {len(generated_answers)} generated vs {len(actual_answers)} references"

        rouge_precs, rouge_recs, rouge_f1s = [], [], []
        for ref, gen in zip(actual_answers, generated_answers):
            score = rouge.score(ref, gen)['rougeL']
            rouge_precs.append(score.precision)
            rouge_recs.append(score.recall)
            rouge_f1s.append(score.fmeasure)

        results[model_name] = {
            'rouge_precision': float(np.mean(rouge_precs)),
            'rouge_recall': float(np.mean(rouge_recs)),
            'rouge_f1': float(np.mean(rouge_f1s))
        }

        mauve_score = mauve.compute_mauve(
            p_text=actual_answers,
            q_text=generated_answers,
            device_id=0,
            verbose=False
        ).mauve
        results[model_name]['mauve'] = float(mauve_score)

        ref_embs = embedding_model.encode(actual_answers, convert_to_tensor=False)
        gen_embs = embedding_model.encode(generated_answers, convert_to_tensor=False)
        mean_cos = float(np.diag(cosine_similarity(ref_embs, gen_embs)).mean())
        results[model_name]['cosine'] = mean_cos

        P, R, F1 = bert_score(generated_answers, actual_answers, lang='en', rescale_with_baseline=True)
        results[model_name]['bertscore'] = {
            'precision': float(P.mean().cpu().numpy()),
            'recall': float(R.mean().cpu().numpy()),
            'f1': float(F1.mean().cpu().numpy())
        }

        em_p, em_r, em_f1 = compute_exact_match_micro_f1(actual_answers, generated_answers)
        results[model_name]['exact_match'] = {
            'precision': em_p,
            'recall': em_r,
            'f1': em_f1
        }
    return results

def save_evaluation(results, output_path='output/evaluation_metrics.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to {output_path}")

################################ Final Main Block ################################

def main():
    configPath = './Config/ConfigLitScrEval.cfg'
    config = setup_environment(configPath)

    model_runs = [
        ('meta-llama/Llama-3.2-1B-Instruct', 'output/output_llama-3.2-1B.json'),
        ('deepseek-ai/deepseek-llm-7b-chat', 'output/output_deepseek-7b.json')
    ]

    wiki_data = load_wiki_dataset()
    actual_answers = [sanitize_text(a) for idx, (title, qa_pairs) in enumerate(wiki_data.items()) for (q, a) in qa_pairs]

    run_screening(config, wiki_data, model_runs)
    normalized_paths = normalize_outputs(model_runs)
    model_labels = ['llama', 'deepseek']

    results = evaluate_models(normalized_paths, model_labels, actual_answers)
    save_evaluation(results)

if __name__ == "__main__":
    main()
