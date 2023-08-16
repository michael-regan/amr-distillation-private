#!/bin/bash

torchrun --nproc_per_node 1 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-7b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-qald9-prompts-8_exs_all_lgs_2023-08-15.json \
    --report_path ~/reports/llama-qald9-7b-chat_8_exs_2023-08-15.json \
    --compiled_results_path ~/reports/llama-qald9-compiled-results_2023-08-15.jsonl

torchrun --nproc_per_node 1 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-7b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-qald9-prompts-12_exs_all_lgs_2023-08-15.json \
    --report_path ~/reports/llama-qald9-7b-chat_12_exs_2023-08-15.json \
    --compiled_results_path ~/reports/llama-qald9-compiled-results_2023-08-15.jsonl

torchrun --nproc_per_node 1 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-7b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-qald9-prompts-16_exs_all_lgs_2023-08-15.json \
    --report_path ~/reports/llama-qald9-7b-chat_16_exs_2023-08-15.json \
    --compiled_results_path ~/reports/llama-qald9-compiled-results_2023-08-15.jsonl