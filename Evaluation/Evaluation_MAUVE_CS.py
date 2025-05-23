import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from mauve import compute_mauve

# Load the JSON data
with open("merged_output.json", encoding="utf-8") as f:
    data = json.load(f)

# Initialize Sentence-BERT for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize a dictionary to store results per question
results = {}

# Iterate through each question (Q1, Q2, ..., Q27)
for q_id in [f"Q{i}" for i in range(1, 28)]:
    short_answers = []
    correct_answers = []
    
    # Collect all answers for this question across all papers
    for paper_id, paper_data in data.items():
        answers = paper_data.get('answers', {})
        if q_id in answers:
            # Handle cases where answers[q_id] is a string or dict
            if isinstance(answers[q_id], dict):
                short_answer = str(answers[q_id].get('ShortAnswer', ''))
                correct_answer = str(answers[q_id].get('correct_short_answer', ''))
            else:
                # If answers[q_id] is a string, assume it's the correct_short_answer
                short_answer = ''
                correct_answer = str(answers[q_id])
            
            # Skip if either answer is empty
            if short_answer and correct_answer:
                short_answers.append(short_answer)
                correct_answers.append(correct_answer)
    
    # Skip if no valid answers found for this question
    if not short_answers or not correct_answers:
        continue
    
    # Compute Cosine Similarity
    short_embeddings = model.encode(short_answers)
    correct_embeddings = model.encode(correct_answers)
    cosine_sims = [
        cosine_similarity([short], [correct])[0][0]
        for short, correct in zip(short_embeddings, correct_embeddings)
    ]
    avg_cosine = np.mean(cosine_sims)
    
    # Compute MAUVE Score (for distributional similarity)
    mauve_score = compute_mauve(
        p_text=short_answers,
        q_text=correct_answers,
        device_id=0,
        max_text_length=512,
        verbose=False
    ).mauve
    
    # Store results for this question
    results[q_id] = {
        "avg_cosine_similarity": float(avg_cosine),
        "mauve_score": float(mauve_score),
        "num_papers": len(short_answers)
    }

# Save results to cs_score.json
with open('cs_score.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to cs_score.json")