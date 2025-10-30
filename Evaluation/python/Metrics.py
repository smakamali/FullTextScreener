import json
from collections import defaultdict
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
from transformers import pipeline

# === Valid labels per question ===
valid_labels_per_question = {
    "Q1": {"Yes", "No", "Unsure"},
    "Q2": {"Not provided", "Unsure"},
    "Q3": {"Unsure"},
    "Q4": {"Yes", "No", "Unsure"},
    "Q5": {"Yes", "No", "Unsure"},
    "Q6": {"Yes", "No", "Unsure"},
    "Q7": {"Not provided"},
    "Q8": {"Yes", "No", "Unsure"},
    "Q9": {"Not provided"},
    "Q10": {"JOB", "TPC-DS", "TPC-H", "Stack", "CEB", "DSB", "Not provided", "Unsure"},
    "Q11": {"Real", "Synthetic", "Both", "Not provided", "Unsure"},
    "Q12": {"Not provided"},
    "Q13": {"Yes", "No", "Unsure"},
    "Q14": {"Yes", "No", "Unsure"},
    "Q15": {"Supervised", "Unsupervised", "Semi-supervised", "Reinforcement", "Other", "Not provided", "Unsure"},
    "Q16": {"Regression", "Classification", "Learning-to-Rank", "Autoregression", "Clustering", "Other", "Not provided", "Unsure"},
    "Q17": {"Yes", "No", "Unsure"},
    "Q18": {"Yes", "No", "Unsure"},
    "Q19": {"Not provided"},
    "Q20": {"Not provided"},
    "Q21": {"Yes", "No", "Unsure"},
    "Q22": {"Yes", "No", "Unsure"},
    "Q23": {"Yes", "No", "Unsure"},
    "Q24": {"MLP", "RNN", "MSCN", "TCNN", "Tree-LSTM", "BDT", "GNN", "Transformer", "Other", "Not provided", "Unsure"},
    "Q25": {"Yes", "No", "Unsure"},
    "Q26": {"Not provided"}
}

# === Load Predictions and References ===
with open("merged_output.json", encoding="utf-8") as f:
    data = json.load(f)

# === Metric Tools ===
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

# Optional: For NLI-based Consistency and Faithfulness
nli = pipeline("text-classification", model="roberta-large-mnli")
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

def normalize_text(text):
    if isinstance(text, list):
        return ' '.join(map(str, text))
    elif isinstance(text, (str, int, float)):
        return str(text)
    else:
        return ""

# Simple token F1 for short answers
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
    for qid, qa_item in answers.items():
        if not isinstance(qa_item, dict):
            continue
        pred = normalize_text(qa_item.get("ShortAnswer", ""))
        ref = normalize_text(qa_item.get("correct_short_answer", ""))
        reasoning = normalize_text(qa_item.get("Reasoning", ""))
        evidence = normalize_text(qa_item.get("Evidence", ""))
        question_text = normalize_text(qa_item.get("QuestionText", ""))

        questions[qid].append((pred, ref, reasoning, evidence, question_text))

# === Compute Metrics Per Question ===
results = {}
all_em, all_f1, all_f1_macro, all_f1_micro, all_r1, all_rl, all_bleu, all_bert = [], [], [], [], [], [], [], []
all_validity, all_nli, all_faithful = [], [], []

for qid, pairs in questions.items():
    preds = [p for p, _, _, _, _ in pairs]
    refs = [r for _, r, _, _, _ in pairs]

    f1s, r1s, rls, bleus, ems, valids, nlis, faithfuls = [], [], [], [], [], [], [], []

    for pred, ref, reasoning, evidence, qtext in pairs:
        # Exact Match
        ems.append(1 if pred == ref else 0)

        # Validity Rate per question
        valid_set = valid_labels_per_question.get(qid, set())
        if valid_set:
            valids.append(1 if pred in valid_set else 0)
        else:
            valids.append(1)

        # Simple Token F1
        f1s.append(simple_token_f1(pred, ref))

        # ROUGE
        rouge_scores = rouge.score(ref, pred)
        r1s.append(rouge_scores["rouge1"].fmeasure)
        rls.append(rouge_scores["rougeL"].fmeasure)

        # BLEU
        bleus.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth))

        # NLI Consistency
        if evidence and evidence != "Not Applicable":
            nli_result = nli(f"Premise: {evidence} Hypothesis: {reasoning}", truncation=True)[0]
            nlis.append(1 if nli_result["label"] == "ENTAILMENT" else 0)

        # Faithfulness via QA reproduction
        if evidence and evidence != "Not Applicable":
            fa = qa({"question": qtext, "context": evidence})
            faithfuls.append(1 if fa["answer"].strip().lower() == pred.lower() else 0)

    # === BERTScore ===
    _, _, bert_f1s = bert_score.score(preds, refs, lang="en", verbose=False)
    bert_f1s = bert_f1s.tolist()

    # === Macro/Micro F1 ===
    try:
        f1_macro = f1_score(refs, preds, average="macro", zero_division=0)
        f1_micro = f1_score(refs, preds, average="micro", zero_division=0)
    except ValueError:
        f1_macro, f1_micro = 0.0, 0.0

    # Store per-question results
    results[qid] = {
        "ExactMatch": round(sum(ems) / len(ems), 4),
        "F1_Token": round(sum(f1s) / len(f1s), 4),
        "F1_Macro": round(f1_macro, 4),
        "F1_Micro": round(f1_micro, 4),
        "ROUGE-1": round(sum(r1s) / len(r1s), 4),
        "ROUGE-L": round(sum(rls) / len(rls), 4),
        "BLEU": round(sum(bleus) / len(bleus), 4),
        "BERTScore-F1": round(sum(bert_f1s) / len(bert_f1s), 4),
        "ValidityRate": round(sum(valids) / len(valids), 4),
        "NLI_Consistency": round(sum(nlis) / len(nlis), 4) if nlis else None,
        "Faithfulness": round(sum(faithfuls) / len(faithfuls), 4) if faithfuls else None,
        "Count": len(pairs)
    }

    # Accumulate for overall
    all_em += ems
    all_f1 += f1s
    all_f1_macro.append(f1_macro)
    all_f1_micro.append(f1_micro)
    all_r1 += r1s
    all_rl += rls
    all_bleu += bleus
    all_bert += bert_f1s
    all_validity += valids
    all_nli += nlis
    all_faithful += faithfuls

# === Overall ===
results["overall"] = {
    "ExactMatch": round(sum(all_em) / len(all_em), 4),
    "F1_Token": round(sum(all_f1) / len(all_f1), 4),
    "F1_Macro": round(sum(all_f1_macro) / len(all_f1_macro), 4),
    "F1_Micro": round(sum(all_f1_micro) / len(all_f1_micro), 4),
    "ROUGE-1": round(sum(all_r1) / len(all_r1), 4),
    "ROUGE-L": round(sum(all_rl) / len(all_rl), 4),
    "BLEU": round(sum(all_bleu) / len(all_bleu), 4),
    "BERTScore-F1": round(sum(all_bert) / len(all_bert), 4),
    "ValidityRate": round(sum(all_validity) / len(all_validity), 4),
    "NLI_Consistency": round(sum(all_nli) / len(all_nli), 4) if all_nli else None,
    "Faithfulness": round(sum(all_faithful) / len(all_faithful), 4) if all_faithful else None,
    "TotalAnswers": len(all_f1)
}

# === Save ===
with open("Scores_AllMetrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("All metrics written to Scores_AllMetrics.json")
