import json
from collections import defaultdict
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
import bert_score

# === Load Predictions and References ===
with open("merged_output.json", encoding="utf-8") as f:
    data = json.load(f)

# === Metric Tools ===
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def normalize_text(text):
    if isinstance(text, list):
        return ' '.join(map(str, text))
    elif isinstance(text, (str, int, float)):
        return str(text)
    else:
        return ""

def simple_token_f1(pred, ref):
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    tp = len(pred_tokens & ref_tokens)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

# === Group answers by question ID ===
questions = defaultdict(list)

for paper_id, content in data.items():
    answers = content.get("answers", {})
    for qid, qa in answers.items():
        if not isinstance(qa, dict):
            continue
        pred = normalize_text(qa.get("ShortAnswer", ""))
        ref = normalize_text(qa.get("correct_short_answer", ""))
        questions[qid].append((pred, ref))

# === Compute Metrics Per Question ===
results = {}
all_f1, all_r1, all_rl, all_bert = [], [], [], []

for qid, pairs in questions.items():
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]

    f1s, r1s, rls = [], [], []
    for pred, ref in pairs:
        f1 = simple_token_f1(pred, ref)
        rouge_scores = rouge.score(ref, pred)
        r1 = rouge_scores["rouge1"].fmeasure
        rl = rouge_scores["rougeL"].fmeasure
        f1s.append(f1)
        r1s.append(r1)
        rls.append(rl)

    # BERTScore
    _, _, bert_f1s = bert_score.score(preds, refs, lang="en", verbose=False)
    bert_f1s = bert_f1s.tolist()

    # Averages for this question
    results[qid] = {
        "F1": round(sum(f1s) / len(f1s), 4),
        "ROUGE-1": round(sum(r1s) / len(r1s), 4),
        "ROUGE-L": round(sum(rls) / len(rls), 4),
        "BERTScore-F1": round(sum(bert_f1s) / len(bert_f1s), 4),
        "Count": len(pairs)
    }

    # Accumulate for global average
    all_f1 += f1s
    all_r1 += r1s
    all_rl += rls
    all_bert += bert_f1s

# === Overall Averages Across All Questions ===
results["overall"] = {
    "F1": round(sum(all_f1) / len(all_f1), 4),
    "ROUGE-1": round(sum(all_r1) / len(all_r1), 4),
    "ROUGE-L": round(sum(all_rl) / len(all_rl), 4),
    "BERTScore-F1": round(sum(all_bert) / len(all_bert), 4),
    "TotalAnswers": len(all_f1)
}

# === Save to JSON ===
with open("Scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("Per-question and overall metrics written to Scores.json.")
