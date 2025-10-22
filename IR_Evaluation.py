from scipy.stats import kendalltau
import numpy as np
from sentence_transformers import util
import huggingface_hub
from FullTextScreener import Config, ChatbotAgents, setLogging, QueryEngine
from llama_index.core.settings import Settings
from datasets import load_dataset


# # Information Retrieval Evaluation Functions
def precision_at_k(retrieved_docs, relevant_docs, k=10):
    """Compute Precision@K (P@K)"""
    retrieved_top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in retrieved_top_k if doc in relevant_docs)
    return relevant_count / k

def mean_reciprocal_rank(retrieved_docs, relevant_docs, k=5):
    """Compute Mean Reciprocal Rank (MRR@K)"""
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            return 1 / (i + 1)  # Rank is 1-based
    return 0  # No relevant document found in top K

def context_recall(retrieved_docs, relevant_docs):
    """Compute Context Recall = Relevant Retrieved / Total Relevant"""
    if not relevant_docs:
        return 0
    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    return relevant_retrieved / len(relevant_docs)

def kendalls_tau(ranking_1, ranking_2):
    """Compute Kendall’s Tau for ranking consistency"""
    return kendalltau(ranking_1, ranking_2)[0]  # Extract correlation coefficient

def intra_list_diversity(retrieved_embeddings):
    """Compute Intra-List Diversity (ILD) using cosine similarity"""
    if len(retrieved_embeddings) < 2:
        return 1  # If only one document, no redundancy
    diversity_scores = []
    for i in range(len(retrieved_embeddings)):
        for j in range(i + 1, len(retrieved_embeddings)):
            sim = util.pytorch_cos_sim(retrieved_embeddings[i], retrieved_embeddings[j]).item()
            diversity_scores.append(1 - sim)  # Diversity is 1 - similarity
    return np.mean(diversity_scores) if diversity_scores else 1

# process a single query
def process_single_title_ir(title, qa_pairs, agent, k_values=[10, 5]):
    config = Config('./Config/ConfigLitScr.cfg')
    embeddingModelName = "Alibaba-NLP/gte-multilingual-base"
    embedDim = 768
    llm = MistralAI(
            model="mistral-large-latest",
            api_key="eYSYlGHlJjIhnsVVdhLrK2kSj2BP7wpl",
            temperature=0,
            max_tokens=1100
        )
    embedModel = HuggingFaceEmbedding(
                model_name=embeddingModelName, 
                max_length=embedDim,
                trust_remote_code=True
            )
    Settings.llm = llm
    Settings.embed_model = embedModel
    qe = QueryEngine(llm, embedModel, config)
    engine, _ = qe.get()
    response = engine.query("Your query here")
    print(response)
    return 0
    # # Step 1: Get full article content
    
    # try:
    #     article_path = f"documents/{title}"
    #     with open(article_path, "w") as f:
    #         article = f.read()
    # except Exception as e:
    #     print(f"Could not retrieve page {title}: {e}")
    #     return None

    # # Step 2: Clear previous memory and upload current text as a file
    # agent.clearFiles()

    # # Create a temp text file
    # temp_txt_path = f"/tmp/{title.replace(' ', '_')}.txt"
    # with open(temp_txt_path, 'w') as f:
    #     f.write(context_text)

    # agent.uploadFiles([temp_txt_path])
    
    # # Index it
    # try:
    #     agent.appendIndex(agent.config.retriever['vectortopk'], agent.config.retriever['cutoffscore'])
    # except Exception as e:
    #     print(f"Indexing failed for article {title}: {e}")
    #     return None

    # # Step 3: Query the agent with the question
    # result = agent.queryTool(queryStr=question, topK=10)
    # retrieved_texts = [doc['text'] for doc in result]

    # # Step 4: Evaluate
    # scores = {}
    # for k in k_values:
    #     scores[f"P@{k}"] = precision_at_k(retrieved_texts, ground_truth_answers, k=k)
    #     scores[f"MRR@{k}"] = mean_reciprocal_rank(retrieved_texts, ground_truth_answers, k=k)

    # scores["ContextRecall"] = context_recall(retrieved_texts, ground_truth_answers)

    # return scores


# ################################ Main Execution ################################
if __name__ == "__main__":
    # load dataset
    print("Loading dataset ...")
    dataset = load_dataset("microsoft/wiki_qa")
    train_data = dataset['train']

    wiki_data = {}
    count = 0
    for example in train_data:
        if example["label"] != 1:
            continue

        doc_title = example["document_title"]
        if doc_title not in wiki_data:
            wiki_data[doc_title] = []

        wiki_data[doc_title].append((example["question"], example["answer"]))
        count += 1

        if count == 5:
            break

    print("Dataset loaded!")


    k_values = [10, 5]  # For P@10 and MRR@5

    # intialize arrays for metrics
    precision_scores = []
    mrr_scores = []
    context_recall_scores = []
    kendall_scores = []
    ild_scores = []

    # set up the config
    print("Setting up config ...")
    configPath = './Config/ConfigLitScr.cfg'
    config = Config(configPath)
    print("Config set up!")
    
    # initialze the chatbot
    print("Intializing chatbot ...")
    screeningAgent = ChatbotAgents(configPath=configPath)
    screeningAgent.selectDefaultAgent()
    print("Chatbot setup!")
    
    # loop though each title and answer the questions
    for title, qa_pairs in wiki_data.items():
        print(f"\nEvaluating: {title}")
        
        metrics = process_single_title_ir(title, qa_pairs, screeningAgent)
        
        
        if metrics:
            precision_scores.append(metrics["P@10"])
            mrr_scores.append(metrics["MRR@5"])
            context_recall_scores.append(metrics["ContextRecall"])
