import json
import csv
from pathlib import Path
import numpy as np
import re
import time
import pandas as pd
import sys


""" USAGE:
python generateResults_json_output.py [EXPERIMENT_PATH][mode] [language]

modes: 
- json (output of estimations is a JSON with the word and its prediction.
  It checks if the word in the input matches the word in the output)
- weighted_sum (wheighted sum of the logprobs of the tokens in the word.
  Only valid if single token output)
- number (output of estimations is a number, the estimation of the word)

languages:
- german
"""

def extract_word_input(text):
    match = re.search(f'{open_quotations_constant}(.*?){closing_quotations_constant}', text)
    if match:
        word = match.group(1)
        return word
    return None


def extract_word_output(text):
    match = re.search(f'"{word_constant}"' + r'\s*:\s*"([^"]+)"', text)
    if match:
        word = match.group(1)
        return word
    return None


def extract_number(text):
    match = re.search(f'"{feature_constant}' + r'"\s*:\s*"([0-9]*\.?[0-9]+)"', text)
    if not match:
        match = re.search(f'"{feature_constant}' + r'"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        all_matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+\.\d*|[-+]?\d+', text)
        if all_matches:
            return float(all_matches[-1])
    if match:
        return float(match.group(1))
    return None


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

## OpenAI ##

def openAI_processing(results_content_file, batches_content_file):
    match_key = 'custom_id'
    lookup = {entry[match_key]: entry for entry in batches_content_file if match_key in entry}
    combined_data = []
    index = 0

    for entry in results_content_file:
        entry_result = {}
        index += 1
        weighted_sum = None
        logprob = None
        word_input = None
        feature_value = None

        if match_key in entry and entry[match_key] in lookup:
           combined_entry = {**entry, **lookup[entry[match_key]]}
           
           print(entry, lookup[entry[match_key]])
           
           custom_id = combined_entry['custom_id']
           if mode == "json":             
              word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
              word_output = extract_word_output(combined_entry['response']['body']['choices'][0]['message']['content'])

              feature_value = extract_number(combined_entry['response']['body']['choices'][0]['message']['content'])
              if word_input and word_output:
                 if word_input != word_output:
                    print(f"Warning: custom Id: '{custom_id}. Word input '{word_input}' does not match word output '{word_output}'")
                    #feature_value = '#N/D'

           elif mode == "weighted_sum" or mode == "number":
              word_input = extract_word_input(combined_entry['body']['messages'][0]['content'])
              # Only valid for responses of single token
              if len(combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"]) == 1:
                 top_logprobs_list = combined_entry["response"]["body"]["choices"][0]["logprobs"]["content"][0]['top_logprobs']
                 weighted_sum = 0
                 total_prob = 0
                 # Iterate over the list of top_logprobs that are numbers
                 for top_logprob in top_logprobs_list:
                     try:
                         token_value = int(top_logprob['token'])
                         logprob_value = top_logprob['logprob']
                         weighted_sum += token_value * np.exp(float(logprob_value))
                         total_prob += np.exp(float(logprob_value))
                     except ValueError:
                         pass
                 weighted_sum /= total_prob if total_prob > 0 else 1
                 logprob = combined_entry['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
              feature_value = combined_entry['response']['body']['choices'][0]['message']['content']

           entry_result['word'] = word_input
           # entry_result['custom_id'] = custom_id
           entry_result[feature_column] = feature_value

           if logprob is not None:
              entry_result['logprob'] = logprob
              logprobs = True
           if weighted_sum is not None:
              entry_result['weighted_sum'] = weighted_sum

           combined_data.append(entry_result)

    return combined_data

## Main ##

if __name__ == "__main__":
    if len(sys.argv) > 1:
       EXPERIMENT_PATH = sys.argv[1]

    if len(sys.argv) > 2:
       mode = sys.argv[2]
       if len(sys.argv) > 3:
          country = sys.argv[3]
       else:
          country = ""
    else:
        print(
            "Provide as arguments the experiment path, the mode\
             [json, number, weighted_sum] and optionally the language."
        )
        exit()

    open_quotations_constant = '"'
    closing_quotations_constant = '"'
    # TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
    word_constant = 'Word'
    feature_column = 'familiarity'
    feature_constant = 'AoA'
    # END TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
    logprobs = False
    timestamp = int(time.time())
    output_file = f'{EXPERIMENT_PATH}/output_{timestamp}.xlsx'


    if country == 'german':
        # TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
        word_constant = 'Wort'
        feature_constant = 'Erwerbsalter'
        # END TO CONFIGURE DEPENDING ON YOUR CASE STUDY #
        open_quotations_constant = '„'
        closing_quotations_constant = '”'


    #results_file = f"{EXPERIMENT_PATH}/results/results.jsonl"
    #batches_file = f"{EXPERIMENT_PATH}/batches/batches.jsonl"
    results_file = f"{EXPERIMENT_PATH}/batch_69768be725ec81909eff329e70702946_output.jsonl"
    batches_file = f"{EXPERIMENT_PATH}/questionaries_arousal_valence_v01_sample_batch_0_2026-01-25-22-29.jsonl"


    results_content_file = read_jsonl(results_file)
    batches_content_file = read_jsonl(batches_file)

    if 'custom_id' in batches_content_file[0]:
        combined_data = openAI_processing(results_content_file, batches_content_file)
    # elif 'key' in batches_content_file[0]:
        # combined_data = google_processing(results_content_file, batches_content_file)
    # elif 'id' in batches_content_file[0]:
        # combined_data = huggingface_processing(results_content_file, batches_content_file)
    else:
        print("Unknown batch format, cannot process results.")

    all_fieldnames = list(combined_data[0].keys())

    df = pd.DataFrame(combined_data)
    df.to_excel(output_file, index=False, columns=all_fieldnames)

    print(f"Combined data written to {output_file}")
