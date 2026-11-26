import json
import os

def process_and_merge_json_files(input_dir="individual", output_file="output.json"):
    merged_data = {}
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            # Read the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Add "correct_short_answer" to all question objects
            if 'answers' in data:
                for q_key, q_data in data['answers'].items():
                    if isinstance(q_data, dict) and 'ShortAnswer' in q_data:
                        q_data['correct_short_answer'] = ''
            
            # Use the filename (without extension) as the key
            file_key = os.path.splitext(filename)[0]
            merged_data[file_key] = data
    
    # Write the merged data to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged data saved to {output_file}")

# Run the script with the specified directory structure
process_and_merge_json_files()