# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import json

from llama import Llama



"""
torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir llama-2-13b-chat \
    --tokenizer_path tokenizer.model \
    --max_seq_len 1024 \
    --max_batch_size 40 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts.json \
    --report_path ~/reports/llama-massive-as-triples-2023-08-03.json


"""



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    report_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open(data_path, 'r') as fin:
        dialogs = json.load(fin)

    theseDialogs = [i['dialog'] for i in dialogs]

    results = generator.chat_completion(
        theseDialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    results = list()

    for d, result in zip(dialogs, results):

        dialog = d['dialog']

        print(f"Utt: {d['utt']}")

        # for msg in dialog:
        #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")

        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )

        print("ground truth")
        print()
        print(d['amr_triples'])
        print()

        thisContent = result['generation']['content']

        print("\n==================================\n")

        d['content'] = thisContent
        results.append(d)

    with open(report_path, 'w') as fout:
        json.dump(results, fout, indent=4)


if __name__ == "__main__":
    fire.Fire(main)