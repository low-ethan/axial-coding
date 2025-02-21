import fire
import json
import pandas as pd
import re
import os
from llama import Llama
from itertools import islice

def batch_iterator(iterator, batch_size):
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch

def read_system_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def read_completed_ids(output_csv_path):
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df['id'])
    else:
        return set()

def parse_result(result):
    # match = re.search(r'\{[\s\S]*\}', result)
    # if match:
    #     json_str = match.group(0)
    #     try:
    #         return json.loads(json_str)
    #     except json.JSONDecodeError:
    #         return None
    # return None
    #try:
    #    # Attempt to parse the text as JSON
    #    json_data = json.loads(result)
    #    return json_data
    #except json.JSONDecodeError as e:
    #    # Handle JSON decoding errors
    #    print(f"JSON decoding error: {e}")
    #    return None
    #except Exception as e:
    #    # Handle any other exceptions
    #    print(f"An unexpected error occurred: {e}")
    #      return None
    list_pattern = r'\[{.*?\}]'
    json_list_match = re.search(list_pattern, result, re.DOTALL)

    if json_list_match:
        json_list_text = json_list_match.group()
        json_list_text = re.sub('"', '\'', json_list_text)
        obj = re.sub(r'\{\'', '{"', json_list_text)
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

def generate_prompt(row, system_prompt):
    user_prompt = f"""Text Requiring Open Codes:
                Input:
                Interviewer: {row['prompt']}
                Answer: {row['answer']}
                JSON Output:
                """
    return system_prompt + '\n' + user_prompt

def main(
    ckpt_dir,
    tokenizer_path,
    system_prompt_file,
    input_csv_path,
    temperature=-1,
    top_p=0.2,
    max_seq_len=4960,
    max_gen_len=1024,
    max_batch_size=4
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    output_csv_path = f"llama_{system_prompt_file[:-4]}_neg_temp.csv"
    system_prompt = read_system_prompt(system_prompt_file)
    input_df = pd.read_csv(input_csv_path)
    completed_ids = read_completed_ids(output_csv_path)

    input_df = input_df[~input_df['id'].isin(completed_ids)]
    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))

    print('Going into the main loop now!')

    for batch in batch_iterator(input_df.iterrows(), max_batch_size):
        batch_prompts = []
        batch_metadata = []

        for index, row in batch:
            full_prompt = generate_prompt(row, system_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {'row': index, 'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer']}
            batch_metadata.append(this_metadata)

        results = generator.text_completion(
            batch_prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            parsed_json = parse_result(result['generation'])
            if parsed_json is not None:
                metadata['label_json'] = parsed_json
                new_rows.append(metadata)
            else: 
                print("couldn't parse this: ", result['generation'])

        if new_rows:
            print(f"Writing {len(new_rows)} rows to csv")
            print(f"New ids completed: {[row['id'] for row in new_rows]}")
            new_df = pd.DataFrame(new_rows)
            if os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    fire.Fire(main)
