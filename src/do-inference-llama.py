# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys
sys.path.insert(0, "/home/michaelr/packages")
import sembleu

from sembleu import src
from sembleu.src import bleu_score
from sembleu.src.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, NgramInst

# from sembleu.src import amr_graph
# from sembleu.src.amr_graph import AMRGraph

from typing import Optional

import fire
import json

from llama import Llama



"""
torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir llama-2-13b-chat \
    --tokenizer_path tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --temperature 0.0 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts_2023-08-07.json \
    --report_path ~/reports/llama-massive-13b-chat-2023-08-07.json


"""

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    report_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
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

    theseDialogs = [i['dialog'] for i in dialogs]

    theseDialogInstanceChunks = chunks(dialogs, 4)
    theseDialogChunks = chunks(theseDialogs, 4)

    theseResults = list()

    smoofunc = getattr(SmoothingFunction(), 'method3')

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

            # for msg in dialog:
            #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")

            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )

            # print("ground truth")
            # print()
            # print(d['amr_triples'])
            # print()

            thisHyp = result['generation']['content']
            thisRef = d['amr_ngrams']

            # sntbleu = round(sentence_bleu([thisRef], thisHyp, weights=weights, smoothing_function=smoofunc, auto_reweigh=True), max_ngrams)
            # print(f"Sembleu: {sntbleu}")

            cnt_match_ngrams = 0
            for ngram in thisHyp:
                if thisHyp in thisRef:
                    cnt_match_ngrams+=1
            print(f"Sembleu: {cnt_match_ngrams/len(thisRef)}")
            

            #thisContent = thisContent.replace('Abstract Meaning Representation (AMR)', '').strip()
            #thisContent = thisContent.replace('list of semantic frames', '').strip()

            print("\n==================================\n")

            d['content'] = thisHyp

            theseResults.append(d)

    with open(report_path, 'w') as fout:
        json.dump(theseResults, fout, indent=4)


if __name__ == "__main__":
    fire.Fire(main)