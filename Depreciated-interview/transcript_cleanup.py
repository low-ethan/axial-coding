import pandas as pd
import os


def reformat_transcript(transcript_file):
    # Read the contents of the file
    with open('orig_doc/transcripts/'+transcript_file, 'r') as file:
        lines = file.readlines()

    # Filter out empty lines and any lines that don't start with a speaker's name
    lines = [line.strip() for line in lines if
             line.strip()]

    # Initialize lists to store the prompt-answer pairs
    prompts = []
    answers = []
    current_prompt = ""
    current_answer = ""

    # Process the lines to create prompt-answer pairs
    for line in lines:
        if line.startswith("Timothy Price:"):
            if current_prompt and current_answer:
                prompts.append(current_prompt)
                answers.append(current_answer)
            current_prompt = line[len("Timothy Price:"):].strip()
            current_answer = ""
        elif line.startswith("Participant"):
            if current_answer:
                current_answer += " " + line[len("Participant 01:"):].strip()
            else:
                current_answer = line[len("Participant 01:"):].strip()
        else:
            print(line)

    # Append the last prompt-answer pair
    if current_prompt and current_answer:
        prompts.append(current_prompt)
        answers.append(current_answer)

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame({'prompt': prompts, 'answer': answers})
    csv_name = transcript_file[:-4] + ".csv"
    df.to_csv('ts_dfs/'+csv_name, index=True)


def compile_to_one(df_list):
    for df in df_list:
        df['id'] = range(1, len(df) + 1)



def main():
    dir_list = os.listdir('orig_doc/transcripts')
    for file in dir_list:
        print(file)
        reformat_transcript(file)


if __name__ == '__main__':
    main()