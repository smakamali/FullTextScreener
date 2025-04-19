from scipy.stats import kendalltau
import numpy as np
from sentence_transformers import util
from FullTextScreener import Config, QueryEngine
from llama_index.core.settings import Settings
from datasets import load_dataset
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
from datetime import datetime

# ------------------ Evaluation Metrics ------------------
def precision_at_k(retrieved_docs, relevant_docs, k=10):
    """Compute Precision@K (P@K)"""
    retrieved_top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in retrieved_top_k if doc in relevant_docs)
    return relevant_count / k


def mean_reciprocal_rank(retrieved_docs, relevant_docs, k=5):
    """Compute Mean Reciprocal Rank (MRR@K)"""
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0


# def context_recall(retrieved_docs, relevant_docs):
#     """Compute Context Recall = Relevant Retrieved / Total Relevant"""
#     if not relevant_docs:
#         return 0.0
#     relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
#     return relevant_retrieved / len(relevant_docs)


def kendalls_tau(ranking_1, ranking_2):
    """Compute Kendall’s Tau for ranking consistency"""
    return kendalltau(ranking_1, ranking_2)[0]


def intra_list_diversity(retrieved_embeddings):
    """Compute Intra-List Diversity (ILD) using cosine similarity"""
    if len(retrieved_embeddings) < 2:
        return 1.0
    diversity_scores = []
    for i in range(len(retrieved_embeddings)):
        for j in range(i + 1, len(retrieved_embeddings)):
            sim = util.pytorch_cos_sim(retrieved_embeddings[i], retrieved_embeddings[j]).item()
            diversity_scores.append(1 - sim)
    return float(np.mean(diversity_scores)) if diversity_scores else 1.0


def is_relevant_by_similarity(doc_text, answer, embed_model, threshold=0.7):
    """Use embedding cosine similarity to judge relevance"""
    doc_emb = embed_model.get_text_embedding(doc_text)
    ans_emb = embed_model.get_text_embedding(answer)
    sim = util.pytorch_cos_sim(doc_emb, ans_emb).item()
    return sim >= threshold, sim


# ------------------ Main Execution ------------------
if __name__ == "__main__":
    # Load dataset sample
    print("Loading dataset ...")
    dataset = load_dataset("microsoft/wiki_qa")
    train_data = dataset['train']
    wiki_data = {}
    for example in train_data:
        if example["label"] != 1:
            continue
        title = example["document_title"]
        wiki_data.setdefault(title, []).append((example["question"], example["answer"]))
        # if len(wiki_data) >= 5:
            # break
    print("Dataset loaded!\n")

    # Initialize Config and Query Engine
    configPath = './Config/ConfigLitScr.cfg'
    config = Config(configPath)
    llm = MistralAI(model=config.mistral['model_name'], api_key=config.mistral['api_key'], temperature=0.0, max_tokens=1100)
    embed_model = HuggingFaceEmbedding(model_name=config.mistral['embed_model'], max_length=768, trust_remote_code=True)
    Settings.llm = llm
    Settings.embed_model = embed_model
    qe = QueryEngine(llm, embed_model, config)
    qe._buildVectorQueryEngine(forceReindex=True)
    retriever = qe.vectorRetriever
    print("Query engine initialized!\n")

    # Initialize accumulators
    all_p10, all_mrr5, all_recall, all_ild, all_tau = [], [], [], [], []
    similarity_threshold = 0.7

    # Run evaluation
    for title, qa_pairs in wiki_data.items():
        print(f"Evaluating document: {title}\n")
        for question, correct_answer in qa_pairs:
            results = retriever.retrieve(question)
            retrieved_texts = [doc.text for doc in results]

            # Exact and similarity-based relevance
            relevant_docs = []
            sim_scores = []
            ans_emb = embed_model.get_text_embedding(correct_answer)
            for text in retrieved_texts:
                is_rel, sim = is_relevant_by_similarity(text, correct_answer, embed_model, similarity_threshold)
                sim_scores.append(sim)
                if correct_answer.lower() in text.lower() or is_rel:
                    relevant_docs.append(text)

            # Compute metrics per query
            p10 = precision_at_k(retrieved_texts, relevant_docs, k=10)
            mrr5 = mean_reciprocal_rank(retrieved_texts, relevant_docs, k=5)
            # recall = context_recall(retrieved_texts, relevant_docs)
            embeddings_10 = [embed_model.get_text_embedding(t) for t in retrieved_texts[:10]]
            ild10 = intra_list_diversity(embeddings_10)

            # Kendall's Tau: compare retrieval order vs similarity-ranked order
            rank1 = list(range(len(retrieved_texts)))
            rank2 = [idx for idx, _ in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]
            tau = kendalls_tau(rank1, rank2)

            # Accumulate
            all_p10.append(p10)
            all_mrr5.append(mrr5)
            # all_recall.append(recall)
            all_ild.append(ild10)
            all_tau.append(tau)

        # Remove this break to evaluate all documents
        # break

    # Compute overall averages
    avg_p10 = np.mean(all_p10)
    avg_mrr5 = np.mean(all_mrr5)
    # avg_recall = np.mean(all_recall)
    avg_ild = np.mean(all_ild)
    avg_tau = np.mean(all_tau)

    results = {
        "Average_P@10": round(avg_p10, 3),
        "Average_MRR@5": round(avg_mrr5, 3),
        # "Average_Recall": round(avg_recall, 3),
        "Average_ILD@10": round(avg_ild, 3),
        "Average_Kendalls_Tau": round(avg_tau, 3),
        "Num_Queries": len(all_p10)
    }

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f"ir_eval_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n=== Overall Evaluation ===")
    print(f"Average P@10   : {avg_p10:.3f}")
    print(f"Average MRR@5  : {avg_mrr5:.3f}")
    # print(f"Average Recall : {avg_recall:.3f}")
    print(f"Average ILD@10 : {avg_ild:.3f}")
    print(f"Average Tau    : {avg_tau:.3f}")
