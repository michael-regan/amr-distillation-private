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
import random

from ast import literal_eval

from typing import Optional

from llama import Llama



"""
torchrun --nproc_per_node 1 src/do-inference-generic.py \
    --ckpt_dir ~/models/llama/llama-2-7b-chat \
    --tokenizer_path ~/models/llama/tokenizer.model \
    --max_seq_len 2048 \
    --max_batch_size 2 \
    --num_chunks 1 \
    --temperature 0.7 \
    --data_path ~/portfolio/amr-distillation-private/data/forestry-chat-messages-2023-09-04.json \
    --report_path ~/reports/llama-7b-forestry-chat-messages-2023-09-05.json

    
"""

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def add_helpful_harmful(initial_generations):

    content_for_next_iteration = list()

    for gen in initial_generations:

        print('XXX')


        new_instruction_help = 'Using evidence from the report, list the factors that are helping the forest, like in the example.'
        new_instruction_harm= 'Using evidence from the report, list the factors that are harming the forest, like in the example.'

        new_system_help = {'role': 'system', 'content': new_instruction_help}
        new_system_harm = {'role': 'system', 'content': new_instruction_harm}

        user_example_help_1 = {'role': 'user', 'content': 'Example 1\nThe forest gets a lot of sunshine and rain. Invasive species have been harming the forest.\nHelpful factors\n'}
        user_example_harm_1 = {'role': 'user', 'content': 'Example 1\nThe forest gets a lot of sunshine and rain. Invasive species have been harming the forest.\nHarmful factors\n'}

        assistant_example_help_1 = {'role': 'assistant', 'content': "['sunshine', 'rain']"}
        assistant_example_harm_1 = {'role': 'assistant', 'content': "['invasive species']"}

        user_example_help_2 = {'role': 'user', 'content': 'Example report\nIncreased wildfire activity has intensified disturbance across wide areas of the forest and has raised concern among scientists and land managers about the resilience of disturbed landscapes.\nHelpful factors\n'}
        user_example_harm_2 = {'role': 'user', 'content': 'Example report\nIncreased wildfire activity has intensified disturbance across wide areas of the forest and has raised concern among scientists and land managers about the resilience of disturbed landscapes.\nHarmful factors\n'}

        assistant_example_help_2 = {'role': 'assistant', 'content': "[None]"}
        assistant_example_harm_2 = {'role': 'assistant', 'content': "['increased wildfire activity']"}

        gen_content = gen['generation']['content'].strip()

        new_content_help = f"Report\n{gen_content}\nHelpful factors\n"
        new_content_harm = f"Report\n{gen_content}\nHarmful factors\n"

        new_user_help = {'role': 'user', 'content': new_content_help}
        new_user_harm = {'role': 'user', 'content': new_content_harm}

        prompt_help = [new_system_help, user_example_help_1, assistant_example_help_1, user_example_help_2, assistant_example_help_2, new_user_help]
        prompt_harm = [new_system_harm, user_example_harm_1, assistant_example_harm_1, user_example_harm_2, assistant_example_harm_2, new_user_harm]

        content_for_next_iteration.append(prompt_help)
        content_for_next_iteration.append(prompt_harm)

    return content_for_next_iteration



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
    model_parallel_size: Optional[int] = None,
    
):


    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size
    )

    final_generations = list()

    with open(data_path, 'r') as fin:
        dialogs = json.load(fin)

    #$theseDialogInstanceChunks = chunks(dialogs, num_chunks)
    theseDialogChunks = chunks(dialogs, num_chunks)

    def inference(dialogChunk, generator, max_gen_len, temperature, top_p):

        compiled_results = list()

        #for chatInstance in dialogChunk:
        results = generator.chat_completion(
            dialogChunk,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for d, result in zip(dialogChunk, results):
            print('-----------'*4)
            print()
            print(d)
            print()
            print(result)
            print()
            compiled_results.append(result)

        return compiled_results
    
    # initial generation of reports
    print("Initial generation of forestry reports")
    
    for dialogChunk in theseDialogChunks:

        temperature = random.random()
        top_ps = [0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95]
        random_top_p = random.sample(top_ps, 1)[0]
        max_gen_len = 1024

        print(f"Random temperature | top_p: {temperature} | {random_top_p}")
        print()

        initial_generations = inference(dialogChunk, generator, max_gen_len, temperature, random_top_p)

        content_for_next_iteration = add_helpful_harmful(initial_generations)

        temperature = 0.95
        max_gen_len = 128
        top_p = 0.95

        print("****Helpful and harmful****")
        print()
        final_results = inference(content_for_next_iteration, generator, max_gen_len, temperature, top_p)

        final_generations.extend(final_results)

    print(f"Writing report to: {report_path}")
    with open(report_path, 'w') as fout:
        json.dump(final_generations, fout, indent=4)


if __name__ == "__main__":
    fire.Fire(main)