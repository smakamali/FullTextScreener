import json

# Load the file
with open("correct.json", encoding="utf-8") as f:
    data = json.load(f)

# Loop through all paper entries
for paper in data.values():
    answers = paper["answers"]

    # Skip if already has all 27 keys
    if len(answers) == 27:
        continue

    # Find missing question (assumed to be Q8 here)
    missing_index = 8

    # Create a new dict with shifted keys
    new_answers = {}
    for i in range(1, 28):
        if i < missing_index:
            new_answers[f"Q{i}"] = answers.get(f"Q{i}", {})
        elif i > missing_index:
            # Shift Q9→Q8, Q10→Q9, etc.
            prev_key = f"Q{i}"
            new_key = f"Q{i-1}"
            if prev_key in answers:
                new_answers[new_key] = answers[prev_key]
    # Optional: remove Q27 if we now have Q1 to Q27 filled
    if len(new_answers) == 26:
        paper["answers"] = new_answers
        print(paper)

# Save to new file
with open("correct_enhanced_shifted.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
