import fire
import json
import pandas as pd
import re
import os
from itertools import islice
from transformers import AutoTokenizer, AutoModelForCausalLM


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

def read_axials(axial_path):
    if os.path.exists(axial_path) and os.path.getsize(axial_path) > 0:
        df = pd.read_csv(axial_path)
        return set(df['axial_name'])
    else:
        return set()

def parse_result(result):
    json_pattern = r'\{\s*"Axial_Category":\s*".*?"\s*\}'
    matches = re.findall(json_pattern, result, re.DOTALL)
    results = []
    for match in matches:
        try:
            x = json.loads(match)
            results.append(x)
        except json.JSONDecodeError as e:
            print(e)
    return results
def generate_prompt(row, system_prompt):
    user_prompt = f"""Codes Requiring Axial Category: {row['clusters']}
                JSON Output:
                """
    return system_prompt + '\n' + user_prompt

def main(
    tokenizer_path,
    system_prompt_file,
    input_csv_path,
    temperature=0.6,
    top_p=0.9,
    max_seq_len=4960,
    max_gen_len=180,
    max_batch_size=4
):

    model_path = tokenizer_path
    tokenizer_path = tokenizer_path

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")

    output_csv_path = f"gemma_7B_{system_prompt_file[:-4]}.csv"
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
        results=[]

        for index, row in batch:
            full_prompt = generate_prompt(row, system_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {'row': index, 'id': row['id'], 'prompt': row['prompt'], 'topic': row['Q3_Health Problem']}
            batch_metadata.append(this_metadata)
            input_ids = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids,
                                     max_new_tokens=max_gen_len,
                                     do_sample=True)
            results.append(tokenizer.decode(outputs[0])[len(full_prompt):])



        # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        # outputs = model.generate(**input_ids)
        # print(tokenizer.decode(outputs[0]))
        #
        # results = generator.text_completion(
        #     batch_prompts,
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        # )

        new_rows = []
        for metadata, result in zip(batch_metadata, results):
            parsed_json = parse_result(result)
            if parsed_json is not None:
                metadata['label_json'] = parsed_json
                new_rows.append(metadata)
            else:
                print("couldn't parse this: ", result)

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
