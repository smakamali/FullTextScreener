import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# === Load Scores ===
with open("Scores.json") as f:
    scores = json.load(f)

# Remove 'overall' to isolate per-question data
question_scores = {k: v for k, v in scores.items() if k != "overall"}

# Metrics to plot
metrics = ["F1", "ROUGE-1", "ROUGE-L", "BERTScore-F1"]

# Set plot style
sns.set(style="whitegrid")

# Create PDF report
pdf = PdfPages("Scores_Visualization_Report.pdf")

# Plot and save individual charts
for metric in metrics:
    plt.figure(figsize=(12, 6))
    values = [v[metric] for v in question_scores.values()]
    questions = list(question_scores.keys())

    sns.barplot(x=questions, y=values, palette="viridis")

    # Draw overall average line
    avg = scores["overall"][metric]
    plt.axhline(avg, color="red", linestyle="--", label=f"Overall Avg: {avg:.4f}")

    plt.title(f"{metric} per Question", fontsize=14)
    plt.ylabel(metric)
    plt.xlabel("Question ID")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save as PNG
    png_filename = f"{metric}_per_question.png"
    plt.savefig(png_filename)
    print(f"Saved chart: {png_filename}")

    # Save to PDF
    pdf.savefig()
    plt.close()

# Save and close the PDF
pdf.close()
print("Combined PDF report saved as 'Scores_Visualization_Report.pdf'")
