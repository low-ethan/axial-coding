import json
import pandas as pd
import re


# def parse_result(result):
#     match = re.search(r'\{[\s\S]*\}', result)
#     if match:
#         json_str = match.group(0)
#         try:
#             return json.loads(json_str)
#         except json.JSONDecodeError:
#             return None
#     return None


def validate_json_objects(json_objects):
    # Define the required keys
    required_keys = {"Original_Text", "Code", "Topic", "Keywords"}

    # Iterate over each JSON object and validate
    for obj in json_objects:
        # Check if all required keys are present
        if not required_keys.issubset(obj.keys()):
            return None

        # Check if 'Keywords' is a list of strings
        if not isinstance(obj['Keywords'], list) or not all(isinstance(keyword, str) for keyword in obj['Keywords']):
            return None

    # If all objects pass the validation, return the parsed JSON objects
    return json_objects


def json_match(output_text):
    json_pattern = r'\[\{.*?\}\]'
    json_matches = re.findall(json_pattern, output_text, re.DOTALL)
    if json_matches:
        try:
            x = json.loads(json_matches[0])
            print(x)
            return validate_json_objects(x)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None


def main():
    file = '2_gemma_7B_prompt_two_examples.csv'
    df = pd.read_csv(file)

    i = 0

    for index, row in df.iterrows():
        print(row['answer'])
        print(json_match(row['label_json']))
        print()
        i += 1
        if i > 50:
            break



if __name__ == '__main__':
    main()