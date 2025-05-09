import json
from configparser import ConfigParser

def convert_to_exact_format(input_json):
    """
    Converts to the exact format specified with individual dropdown fields.
    
    Args:
        input_json (dict): Original JSON data from output_mistral.json
        
    Returns:
        dict: Transformed JSON with exact requested structure
    """
    output = {}
    
    for paper_id, paper_data in input_json.items():
        output[paper_id] = {
            "metadata": paper_data["metadata"],
            "answers": {}
        }
        
        for q_key, q_data in paper_data["answers"].items():
            # Handle string answers that need JSON parsing
            if isinstance(q_data, str):
                try:
                    q_data = json.loads(q_data.replace('\n', ''))
                except json.JSONDecodeError:
                    q_data = {"ShortAnswer": q_data, "Reasoning": "", "Evidence": ""}
            
            # Create the exact requested format
            output[paper_id]["answers"][q_key] = {
                "QuestionText": q_data.get("QuestionText", ""),
                "ShortAnswer": q_data.get("ShortAnswer", ""),
                "Reasoning": q_data.get("Reasoning", ""),
                "Evidence": q_data.get("Evidence", ""),
                "Evaluation": {
                    "Correct_Short_answer": "",
                    # "Correctness": {
                    #   "Is the answer factually correct given the paper?": {
                    #     "dropdown": [
                    #       1,
                    #       2,
                    #       3,
                    #       4,
                    #       5
                    #     ]
                    #   }
                    # },
                    # "Completeness": {
                    #   "Does the answer cover all aspects of the question?": {
                    #     "dropdown": [
                    #       1,
                    #       2,
                    #       3,
                    #       4,
                    #       5
                    #     ]
                    #   }
                    # },
                    # "Relevance": {
                    #   "Is the answer directly relevant to the question?": {
                    #     "dropdown": [
                    #       1,
                    #       2,
                    #       3,
                    #       4,
                    #       5
                    #     ]
                    #   }
                    # },
                    # "Evidence Quality": {
                    #   "Does the answer provide clear supporting evidence from the paper?": {
                    #     "dropdown": [
                    #       1,
                    #       2,
                    #       3,
                    #       4,
                    #       5
                    #     ]
                    #   }
                    # }
            }
    }
    
    return output

# Usage example:
with open('output/output_mistral.json') as f:
    original_data = json.load(f)

exact_format = convert_to_exact_format(original_data)

# Save to file
config = ConfigParser()
config.read('./Config/ConfigLitScr.cfg')
service = config.get(config.get('llm-model', 'service'), 'model_name').replace('/', '_')
form_file = f"form/format_{service}.json"

with open(form_file, 'w') as f:
    json.dump(exact_format, f, indent=2)

print("Converted to exact format. Saved to exact_format.json")