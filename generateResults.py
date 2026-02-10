import sys
import json
from   pathlib import Path
import re
import time
import pandas as pd
import collections
import numpy as np

""" USAGE:

python generateResults_json_output.py [EXPERIMENT_PATH]

"""

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

## Main ##

if __name__ == "__main__":
    if len(sys.argv) > 1:
       EXPERIMENT_PATH = sys.argv[1]
    else:
        print(
            "Provide as arguments the experiment path (only)."
        )
        exit()

    timestamp = int(time.time())
    output_file = f'{EXPERIMENT_PATH}/{EXPERIMENT_PATH}_output_{timestamp}.xlsx'

    results_file = f"{EXPERIMENT_PATH}/results/results.jsonl"
    batches_file = f"{EXPERIMENT_PATH}/batches/batches.jsonl"
    words_file = f"{EXPERIMENT_PATH}/data/selected_words_for_multidimensional_arousal_estimates.xlsx"

    #results_file = f"{EXPERIMENT_PATH}/batch_69768be725ec81909eff329e70702946_output.jsonl"
    #batches_file = f"{EXPERIMENT_PATH}/questionaries_arousal_valence_v01_sample_batch_0_2026-01-25-22-29.jsonl"
    #words_file = f"{EXPERIMENT_PATH}/data/selected_words_for_multidimensional_arousal_estimates_sample.xlsx"

    df = pd.read_excel(words_file)
    word_base_info = dict((k, {'valence': v, 'arousal': a, 'dominance': d}) for k, v, a, d in df.values)

    results_content = read_jsonl(results_file)
    batches_content = read_jsonl(batches_file)
    # Comparamos que ambos datasets tengan los mismos 'custom_id' para determinar su correspondencia
    # (si falta algún resultado)
    results_ids = [row['custom_id'] for row in results_content]
    batches_ids = [row['custom_id'] for row in batches_content]
    if collections.Counter(results_ids) == collections.Counter(batches_ids):
        print( '"custom_id" fields in batches file and results file DO match!' )
    else:
        print( f'WARNING: results file (length: {len(results_ids)}) DOES NOT match batches file (length: {len(batches_ids)})' )
        
    excel_rows = list()
    excel_rows_dict = dict()
    for list_item in results_content:
        word, sentence_id = re.match( r'questionaries_arousal_valence_task_([^_]+)_(.+)', list_item['custom_id']).group(1, 2)
        model = list_item['response']['body']['model']
        column_result_name = sentence_id + '_result_' + model 
        column_logprob_name = sentence_id + '_logprob_' + model
        ia_result = int(list_item['response']['body']['choices'][0]['message']['content'])
        
        logprob_list = []
        token_list = []
        for logprob_info in list_item['response']['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'] :
            try:
                token = int(logprob_info['token'])
                logprob = float(logprob_info['logprob'])
                token_list.append( token )
                logprob_list.append( logprob )
            except ValueError:
                pass

        prob_list = np.exp(logprob_list)
        total_prob = np.sum(prob_list)
        if total_prob <= 0: total_prob = 1
        expected_token = np.dot(prob_list, token_list) / total_prob
        
        if word in excel_rows_dict:
           pruned_record = excel_rows[excel_rows_dict[word]]
           pruned_record[column_result_name] = ia_result
           pruned_record[column_logprob_name] = expected_token
        else:
           pruned_record = dict()
           pruned_record['word'] = word
           pruned_record['valence'], pruned_record['arousal'], pruned_record['dominance'] = \
            (word_base_info[word]['valence'], word_base_info[word]['arousal'], word_base_info[word]['dominance'])
           pruned_record[column_result_name] = ia_result
           pruned_record[column_logprob_name] = expected_token
           excel_rows_dict[word] = len(excel_rows)
           excel_rows.append(pruned_record)
    
    pd.DataFrame(excel_rows).to_excel(output_file, index = False) 