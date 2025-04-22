import pandas as pd
from itertools import islice
import os
import requests
import json
import random
import re

def parse_result(result):
    list_pattern = r'\[\s*{.*?\}\s*]'
    json_list_match = re.search(list_pattern, result, re.DOTALL)

    if json_list_match:
        json_list_text = json_list_match.group()
        # Dealing with nested " breaking the json
        json_list_text = re.sub('"', '\'', json_list_text)
        obj = re.sub(r'\{\'', '{"', json_list_text)
        obj = re.sub(r'\'}', '"}', obj)
        obj = re.sub(r'\':', '":', obj)
        obj = re.sub(r':\s*\'', ': "', obj)
        obj = re.sub(r'\',\s*\'', '", "', obj)
        obj = re.sub(r'\[\'', '["', obj)
        obj = re.sub(r'\'\]', '"]', obj)
        try:
            json_data = json.loads(obj)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    return None

def batch_iterator(iterator, batch_size):
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def read_system_prompt(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        return ""


def read_completed_ids(output_csv_path):
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df['id'])
    else:
        return set()


def llama3(prompt):
    data = {
        # "model": "llama3:70b",
        # "model": "qwen2.5-coder:32b",
        "model": "gemma3:27b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.2,
            "num_predict": 1000,
            # "stop":["}"],
        },
    }

    headers = {
        "Content-Type": "application/json"
    }
    url = "http://localhost:11434/api/chat"
    response = requests.post(url, headers=headers, json=data)
    # print('r: ', response.json())
    return response.json()["message"]["content"]



def generate_prompt(row, system_prompt):
    user_prompt = f"""Text Requiring Open Codes:
                Input:
                Interviewer: {row['prompt']}
                Answer: {row['answer']}
                JSON Output:
                """
    return system_prompt + '\n' + user_prompt