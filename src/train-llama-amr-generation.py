import fire
import llama
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Optional


"""
python train-llama-amr-generation.py \
    --ckpt_dir ../../../models/llama/llama-2-7b \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 2048 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-12_exs_test_2023-08-08.json \
    --report_path ~/reports/llama-massive-13b-chat_12_exs_test_2023-08-07.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl


"""

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    report_path: str,
    compiled_results_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    num_chunks: int = 4,
    max_gen_len: Optional[int] = None
):

    #model_id="./models_hf/7B"

    tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir)

    model = LlamaForCausalLM.from_pretrained(ckpt_dir, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)


if __name__ == "__main__":
    fire.Fire(main)