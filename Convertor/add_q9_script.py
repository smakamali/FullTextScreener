import json

# Load the fixed JSON file
with open("correct_shifted.json", encoding="utf-8") as f:
    data = json.load(f)

# Loop through each paper entry
for paper in data.values():
    answers = paper["answers"]
    new_answers = {}

    # Step 1: Copy Q1 to Q8 as-is
    for i in range(1, 9):
        new_answers[f"Q{i}"] = answers.get(f"Q{i}", {})

    # Step 2: Insert new Q9
    new_answers["Q9"] = {
        "correct_short_answer": "Not provided"
    }

    # Step 3: Shift Q9–Q26 → Q10–Q27
    for i in range(9, 27):
        old_key = f"Q{i}"
        new_key = f"Q{i + 1}"
        if old_key in answers:
            new_answers[new_key] = answers[old_key]

    # Step 4: Replace the answers
    paper["answers"] = new_answers

# Save to new file
with open("correct_enhanced_with_q9.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
