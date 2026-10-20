#!/bin/bash

model="${1:-qwen}"
input_file="${2:-./output/${model}/chair/bon.json}"
output_file="./output/${model}/scored.json"

python rerank.py \
    --model "$model" \
    --input_file "$input_file" \
    --output_file "$output_file"
