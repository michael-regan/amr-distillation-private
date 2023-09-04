# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# Note: 2023-09-04 -- copy of do-inference-llama.py 
# no changes made

import sys

import datetime
import fire
import json
import jsonlines
import math
import numpy as np
import pandas as pd

from ast import literal_eval

from typing import Optional

from llama import Llama



"""
torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 2048 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/forestry-chat-messages-2023-09-04.json \
    --report_path ~/reports/llama-13b-forestry-chat-messages-2023-09-04.json

    
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
    num_chunks: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size    
    )

    with open(data_path, 'r') as fin:
        dialogs = json.load(fin)

    theseDialogs = [i['dialog'] for i in dialogs]

    theseDialogInstanceChunks = chunks(dialogs, num_chunks)
    theseDialogChunks = chunks(theseDialogs, num_chunks)

    theseResults = list()

    for dialogInstanceChunk, dialogChunk in zip(theseDialogInstanceChunks, theseDialogChunks):

        results = generator.chat_completion(
            dialogChunk,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for d, result in zip(dialogInstanceChunk, results):

            dialog = d['dialog']

            print(result)
            print()

    # print(f"Write report to: {report_path}")
    # with open(report_path, 'w') as fout:
    #     json.dump(theseResults, fout, indent=4)


if __name__ == "__main__":
    fire.Fire(main)