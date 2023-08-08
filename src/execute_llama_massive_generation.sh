#!/bin/bash

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-8_exs_2023-08-07.json \
    --report_path ~/reports/llama-massive-13b-chat_8_exs_2023-08-07.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-12_exs_2023-08-07.json \
    --report_path ~/reports/llama-massive-13b-chat_12_exs_2023-08-07.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-16_exs_2023-08-07.json \
    --report_path ~/reports/llama-massive-13b-chat_16_exs_2023-08-07.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-8_exs_spanish_2023-08-08.json \
    --report_path ~/reports/llama-massive-13b-chat_8_exs_spanish_2023-08-08.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-12_exs_spanish_2023-08-08.json \
    --report_path ~/reports/llama-massive-13b-chat_12_exs_spanish_2023-08-08.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl

torchrun --nproc_per_node 2 do-inference-llama.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-massive-prompts-16_exs_spanish_2023-08-08.json \
    --report_path ~/reports/llama-massive-13b-chat_16_exs_spanish_2023-08-08.json \
    --compiled_results_path ~/reports/llama-massive-compiled-results_2023-08-08.jsonl