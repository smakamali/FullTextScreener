import json
import os

def process_and_merge_json_files(input_dir="individual", output_file="output.json"):
    """
    Merge the complete per-paper results in `input_dir` into one JSON file.

    Only complete results (<paper>.json) are merged. Partial results (<paper>.partial.json, from a paper
    that was interrupted or had failed questions) and temporary files from interrupted saves are skipped
    and listed, so that incomplete answers never end up in the merged output.
    """
    merged_data = {}
    skipped = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.partial.json') or filename.endswith('.json.tmp'):
            skipped.append(filename)
            continue
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add "correct_short_answer" to all question objects
        if 'answers' in data:
            for q_key, q_data in data['answers'].items():
                if isinstance(q_data, dict) and 'ShortAnswer' in q_data:
                    q_data['correct_short_answer'] = ''

        # Use the filename (without extension) as the key
        file_key = os.path.splitext(filename)[0]
        merged_data[file_key] = data

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged {len(merged_data)} complete result(s) into {output_file}")
    if skipped:
        print(f"Skipped {len(skipped)} partial or temporary file(s): {', '.join(skipped)}")

if __name__ == "__main__":
    process_and_merge_json_files()
