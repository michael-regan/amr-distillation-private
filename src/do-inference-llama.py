# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys

import datetime
import fire
import json
import math
import numpy as np
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
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-8_exs_test_2023-08-08.json \
    --report_path ~/reports/llama-massive-13b-chat_8_exs_test_2023-08-08.json


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


def analyze_results(results_list, model_name):

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
            notes.append('n/a')
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{now}")

    print(f"Model: {model_name}\tTemperature: {results_list[0]['temperature']}")
    print()

    print(f"# errors: {df['error'].sum()}")
    print(f"% errors: {df['error'].sum()/len(df):.2f}")
    print(df[['note']].describe())
    print()

    df_amr = df[df.target=='raw_amr']
    #df_amr_scores = df[df.target=='raw_amr' & df.error==0]
    print("Smatch scores, ignoring errors")

    precision, recall, f1 = [],[],[]
    cnt_smatch_errors = 0
    for idx, row in df_amr.iterrows():
        if row.error==0 and type(row.score)==dict:
            #print(row.score, type(row.score))
            obj = row.score
            precision.append(obj['precision'])
            recall.append(obj['recall'])
            f1.append(obj['f1'])
        else:
            cnt_smatch_errors+=1
    print(f"Mean precision:\t{np.mean(precision):.2f}")
    print(f"Mean recall:\t{np.mean(recall):.2f}")
    print(f"Mean f1-score:\t{np.mean(f1):.2f}")
    print(f"Count smatch errors:\t{cnt_smatch_errors}")
    print()

    df_ngram_1 = df[(df.target=='amr_ngrams') & (df.max_ngram==1)]
    df_ngram_1_scores = df[(df.target=='amr_ngrams') & (df.error==0)]
    print("Sembleu scores, max_ngram==1")
    print(df_ngram_1[['score']].describe())
    print()
    print("Sembleu scores, max_ngram==1, ignoring errors")
    print(df_ngram_1_scores[['score']].describe())
    print()

    df_ngram_2 = df[(df.target=='amr_ngrams') & (df.max_ngram==2)]
    df_ngram_2_scores = df[(df.target=='amr_ngrams') & (df.error==0)]
    print("Sembleu scores, max_ngram==2")
    print(df_ngram_2[['score']].describe())
    print()
    print("Sembleu scores, max_ngram==2, ignoring errors")
    print(df_ngram_2_scores[['score']].describe())
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
                print(f"> Smatch: {smatch_score}")
                d['score']=smatch_score

            print("\n==================================\n")

            d['content'] = result['generation']['content']
            d['temperature'] = temperature

            theseResults.append(d)

    print(f"Write report to: {report_path}")
    with open(report_path, 'w') as fout:
        json.dump(theseResults, fout, indent=4)

    # quick view of results

    ckpt_name = ckpt_dir.split('/')[-1]
    analyze_results(theseResults, ckpt_name)


if __name__ == "__main__":
    fire.Fire(main)