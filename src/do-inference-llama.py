# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys

import fire
import json
import math
import pandas as pd

import sembleu_script
import smatch_script

from ast import literal_eval

from collections import namedtuple
NgramInst = namedtuple('NgramInst', 'ngram length')

from typing import Optional

from llama import Llama



"""
torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.0 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-8_exs_2023-08-07.json \
    --report_path ~/reports/llama-massive-13b-chat_8_exs_2023-08-07.json


"""

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def clean_up_output(output_str):
    clean_str = ''
    return clean_str

def convert_to_ngram(obj):

    converted_data = dict()
    length_this_ngram = 0

    len_ngrams = [1, 2]
    for len_ngram in len_ngrams:
        for item in obj:
            item = tuple(item)

            # because each item contains relations
            if len(item)==1:
                count_nodes = 1
            elif len(item)==3:
                count_nodes = 2
            else:
                count_nodes = 3

            if count_nodes==len_ngram:
                if len_ngram not in converted_data:
                    converted_data[len_ngram]=[]
                converted_data[len_ngram].append(item)
                length_this_ngram+=1

    return converted_data, length_this_ngram


def analyze_results(results_list):

    errors, targets, max_ngrams, scores, notes = [],[],[],[],[]

    for item in results_list:

        targets.append(item['target'])
        max_ngrams.append(item['max_ngrams'])

        score = item['score']

        if 'error' in str(score).lower():
            errors.append(1)
            notes.append(score)
            scores.append(0)

        else:
            errors.append(0)
            notes.append(None)
            scores.append(score)

    df = pd.DataFrame(
            {'error': errors,
            'target': targets,
            'max_ngram': max_ngrams,
            'score': scores,
            'note': notes
            })
    
    print("-----------------------")
    print("Results")

    print(f"Temp:{}")

    print(f"# errors: {df['error'].sum()}")
    print(df['note'].value_counts())
    print()

    df_amr = df[df.targets=='raw_amr']
    print("Smatch scores")
    print(df_amr[['score']].describe())
    print()

    df_ngram_1 = df[df.targets=='amr_ngrams' & df.max_ngrams==1]
    print("Sembleu scores, max_ngram==1")
    print(df_ngram_1[['score']].describe())
    print()

    df_ngram_2 = df[df.targets=='amr_ngrams' & df.max_ngrams==2]
    print("Sembleu scores, max_ngram==2")
    print(df_ngram_2[['score']].describe())
    print()
    print("-----------------------")

    




    


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    report_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    num_chunks: int = 4,
    max_gen_len: Optional[int] = None
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size    
    )

    with open(data_path, 'r') as fin:
        dialogs = json.load(fin)

    # for testing amr smatch
    # dialogs = dialogs[-4:]

    theseDialogs = [i['dialog'] for i in dialogs]

    theseDialogInstanceChunks = chunks(dialogs, num_chunks)
    theseDialogChunks = chunks(theseDialogs, num_chunks)

    theseResults = list()

    generation_errors = list()

    smoofunc = getattr(sembleu_script.SmoothingFunction(), 'method3')

    for dialogInstanceChunk, dialogChunk in zip(theseDialogInstanceChunks, theseDialogChunks):

        results = generator.chat_completion(
            dialogChunk,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for d, result in zip(dialogInstanceChunk, results):

            max_ngrams = d['max_ngrams']

            if max_ngrams == 3:
                weights = (0.34, 0.33, 0.34)
            elif max_ngrams == 2:
                weights = (0.5, 0.5)
            else:
                weights = (1.0,)

            dialog = d['dialog']

            print(f"Question: {d['question']}")

            if d['target']=='amr_ngrams':

                rc = result['generation']['content']

                if rc[0]!='[' or rc[-1]!=']':
                    print("Error in generation, cleaning up...")
                    print(rc)
                    rc, sep, tail = rc.partition(']')
                    rc += ']'
                    print(rc)
                    #d['score'] = 'Error in generation'
                
                literal_results = literal_eval(rc)

                thisHyp, length_thisHyp = convert_to_ngram(literal_results)
                thisHypInst = NgramInst(ngram=thisHyp, length=length_thisHyp)

                refDict, length_thisRef = convert_to_ngram(d['amr_ngrams'])
                thisRef = NgramInst(ngram=refDict, length=length_thisRef)

                print(
                    f"> Hypothesis: {thisHypInst}"
                )

                print(
                    f"> Reference: {thisRef}"
                )
                try:
                    sntbleu = round(sembleu_script.sentence_bleu([thisRef], thisHypInst, weights=weights, smoothing_function=smoofunc, auto_reweigh=False), max_ngrams)
                except Exception as e:
                    print("Error in sentence_bleu")
                    sntbleu = 'Error in sentence_bleu'
                print(f"Sembleu: {sntbleu}")
                d['score']=sntbleu

            else:

                thisHyp = result['generation']['content']
                thisRef = d['raw_amr']

                print(
                    f"> Hypothesis: {thisHyp}"
                )

                print(
                    f"> Reference: {thisRef}"
                )
                try:
                    smatch_score = smatch_script.main([thisRef], [thisHyp])
                except Exception as e:
                    print("Error in smatch scoring")
                    smatch_score = 'Error in smatch'
                print(f">Smatch:{smatch_score}")
                d['score']=smatch_score

            print("\n==================================\n")

            d['content'] = result['generation']['content']
            d['temperature'] = temperature

            theseResults.append(d)

    #this_report_path = report_path.replace('{temp}', f'temp={temperature}')
    with open(report_path, 'w') as fout:
        json.dump(theseResults, fout, indent=4)

    # quick view of results
    analyze_results(theseResults)


if __name__ == "__main__":
    fire.Fire(main)