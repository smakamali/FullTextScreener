import json

# Load the full Evaluation.json
with open("Evaluation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extracted result
extracted = {}

for key, content in data.items():
    # Extract metadata
    new_entry = {
        "metadata": content.get("metadata", {}),
        "answers": {}
    }

    # Extract only correct_short_answer from each Q#
    for q_key, q_content in content.get("answers", {}).items():
        if "correct_short_answer" in q_content:
            new_entry["answers"][q_key] = {
                "correct_short_answer": q_content["correct_short_answer"]
            }

    extracted[key] = new_entry

# Save the result to a new JSON file
with open("extracted_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)

print("✅ Extracted data saved to 'extracted_evaluation.json'")
