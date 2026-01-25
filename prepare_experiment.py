"""
This script is designed to prepare tasks for psycholinguistic experiments.
It reads configuration from a JSONL file and words from an Excel file, then
generates tasks aimed to a given API. These tasks are batched and saved as
JSONL files for later execution.

Key Components:
- `load_word_list_from_excel`: Loads a list of words from an Excel file.
- `get_tasks`: Generates tasks for each word using a specified prompt and
  model configuration.
- `create_batches`: Splits tasks into batches and saves them as JSONL files.

Important Considerations:
1. Configuration File: Ensure the `config.yaml` file is correctly set up
   with the necessary parameters `words_dataset_path`, `words_dataset_column`,
   `sentences_dataset_path`, `prompt_path`, `model_name` and `company`.
2. Environment Variables: The `apis.env` file must contain valid API
   credentials.
3. Excel File: The Excel file should be located in the specified experiment
   directory and it should contain a column with the words to be used in
   the experiment.

Usage:
Run the script from the command line with the experiment path and name as
arguments:
    python prepare_experiment.py <EXPERIMENT_PATH> <EXPERIMENT_NAME>

Example:
    python prepare_experiment.py my_experiment_path my_experiment_name

This will create, within my_experiment_path directory, a subdirectory named
"batches" containing the prepared task JSONL files.
"""

import json
import os
import sys
import jsonlines
from datetime import datetime

import pandas as pd

from utils import load_config, openai_login, read_txt

def load_word_and_sentence_lists(words_path: str, column_name: str, sentences_path: str):
    print(f"Loading words and sentences lists from {words_path} column {column_name}...")
    
    wf = None
    if words_path.endswith(".csv"):
       wf = pd.read_csv(words_path, encoding='iso-8859-1')
    elif words_path.endswith(".xlsx"):
       wf = pd.read_excel(words_path)
    else:
       raise ValueError(f"Unsupported file extension: {words_path}")
    word_list = wf[column_name].tolist()
        
    sf = None
    if sentences_path.endswith(".csv"):
       sf = pd.read_csv(sentences_path, encoding='iso-8859-1')
    elif sentences_path.endswith(".xlsx"):
       sf = pd.read_excel(sentences_path)
    else:
        raise ValueError(f"Unsupported file extension: {sentences_path}")
    sentence_list = sf[['Id', 'Sentence']].values.tolist()

    print(f"Successfully loaded {len(word_list)} words and {len(sentence_list)} sentences.")
    return word_list, sentence_list

def get_tasks(experiment_path, word_list, sentence_list, prompt,
              model_version = "gpt-4o-2024-08-06", temperature = 0,
              logprobs = True, top_logprobs = 5,
              word_key = "{Word}", sentence_key = "{Sentence}",
              company = "OpenAI", ft_dir = None) -> list:

    if company == "OpenAI":
        return get_tasks_openai(experiment_path, word_list, sentence_list,
                                prompt, model_version, temperature, logprobs,
                                top_logprobs, word_key, sentence_key)
    # elif company == "Google":
        # return get_tasks_gemini(word_list, experiment_path, prompt, model_version, temperature, logprobs, top_logprobs,prompt_key)
    # elif company == "HuggingFace":
        # return get_tasks_huggingface(word_list, experiment_path, prompt, model_version, temperature, logprobs, top_logprobs,prompt_key, ft_dir)
    # elif company == "Local":
        # return get_tasks_huggingface(word_list, experiment_path, prompt, model_version, temperature, logprobs, top_logprobs,prompt_key, ft_dir)
    else:
        raise ValueError(f"Unknown company: {company}")

def get_tasks_openai(experiment_path, word_list, sentence_list, prompt,
                     model_version, temperature, logprobs, top_logprobs,
                     word_key, sentence_key) -> list:

    tasks = []
    for word in word_list:
        for key, sentence in sentence_list:
            task = {
                   "custom_id": f"{experiment_path}_task_{word}_{key}",
                   "method": "POST",
                   "url": "/v1/chat/completions",
                   "body": {
                           "model": model_version,
                           "temperature": temperature,
                           "logprobs": logprobs,
                           "top_logprobs": top_logprobs,
                           "response_format": {"type": "text"},
                           "messages": [ {"role": "user", "content": prompt.replace(word_key, str(word)).replace(sentence_key, str(sentence))} ],
                           },
                   }

            tasks.append(task)

    return tasks

def create_batches(
    tasks: list, experiment_path: str, run_prefix: str, chunk_size: int = 50000
):
    os.makedirs(f"{experiment_path}/batches", exist_ok = True)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    list_of_tasks = [
        tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)
    ]
    list_of_batch_names = []
    for index, tasks in enumerate(list_of_tasks):
        batch_name = f"batch_{index}_{date_string}.jsonl"
        list_of_batch_names.append(batch_name)
        with jsonlines.open(f"{experiment_path}/batches/{run_prefix}_{batch_name}", "w") as file:
            file.write_all(tasks)

    return list_of_batch_names

######################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            EXPERIMENT_NAME = sys.argv[2]
        else:
            EXPERIMENT_NAME = "original"
    else:
        print(
            "Provide as arguments the experiment path and optionally the experiment name, i.e.: python3 prepare_experiment.py <EXPERIMENT_PATH> <EXPERIMENT_NAME>."
        )
        exit()

    # prepare data
    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="experiments",
        name=EXPERIMENT_NAME,
    )
    
    word_list, sentence_list = load_word_and_sentence_lists(
        words_path = f"{EXPERIMENT_PATH}/data/{config_args['words_dataset_path']}",
        column_name = config_args['words_dataset_column'],
        sentences_path = f"{EXPERIMENT_PATH}/data/{config_args['sentences_dataset_path']}"
    )

    #
    # Prepare batches
    #
    
    prompt = read_txt(
        file_path = f"{EXPERIMENT_PATH}/prompts/{config_args['prompt_path']}"
    )

    tasks = get_tasks(
        experiment_path = EXPERIMENT_PATH,
        word_list = word_list,
        sentence_list = sentence_list,
        prompt = prompt,
        word_key = "{Word}",
        sentence_key = "{Sentence}",
        model_version = config_args["model_name"],
        company = config_args["company"],
        ft_dir = config_args.get("ft_dir", None),
        top_logprobs = config_args.get("top_logprobs", 5)
    )

    list_of_batch_names = create_batches(
        tasks = tasks,
        experiment_path = EXPERIMENT_PATH,
        run_prefix = EXPERIMENT_NAME
    )
