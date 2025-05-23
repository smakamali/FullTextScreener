import json

# Load both JSON files with UTF-8 encoding
with open('correct.json', 'r', encoding='utf-8') as f:
    correct_data = json.load(f)

with open('output.json', 'r', encoding='utf-8') as f:
    output_data = json.load(f)

# Merge logic
merged_data = {}

for paper, correct_content in correct_data.items():
    normalized_key = paper.replace('.json', '')
    output_content = output_data.get(normalized_key, {})

    merged_entry = {
        "metadata": {**output_content.get("metadata", {}), **correct_content.get("metadata", {})},
        "answers": {}
    }

    correct_answers = correct_content.get("answers", {})
    output_answers = output_content.get("answers", {})

    for question, correct_answer in correct_answers.items():
        output_answer = output_answers.get(question, {})
        
        # Ensure output_answer is a dictionary before merging
        if isinstance(output_answer, dict):
            merged_entry["answers"][question] = {
                "correct_short_answer": correct_answer.get("correct_short_answer"),
                **output_answer
            }
        else:
            # Just use the correct answer if output answer is not a dict
            merged_entry["answers"][question] = {
                "correct_short_answer": correct_answer.get("correct_short_answer")
            }

    merged_data[normalized_key] = merged_entry

# Save the merged result with UTF-8 encoding
with open('merged_output.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print("Merging complete. Output saved to 'merged_output.json'.")
