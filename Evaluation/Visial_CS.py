import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('cs_score.json', 'r') as f:
    data = json.load(f)

# Prepare data for plotting
questions = list(data.keys())
cosine_similarities = [data[q]["avg_cosine_similarity"] for q in questions]
mauve_scores = [data[q]["mauve_score"] for q in questions]

x = range(len(questions))
width = 0.35  # width of the bars

# Plotting
plt.figure(figsize=(12, 6))
plt.bar([p - width/2 for p in x], cosine_similarities, width, label='Avg Cosine Similarity', color='skyblue')
plt.bar([p + width/2 for p in x], mauve_scores, width, label='Mauve Score', color='salmon')

plt.xlabel('Questions')
plt.ylabel('Score')
plt.title('Comparison of Avg Cosine Similarity and Mauve Score by Question')
plt.xticks(ticks=x, labels=questions)
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
