#!/bin/bash

model="${1:-qwen}"
coco_path="./data/coco"
output_path="./output/${model}/chair"

python generate.py \
    --model "$model" \
    --data_path "$coco_path/val2014" \
    --anno_path "$coco_path/annotations" \
    --output_file "$output_path/bon.json" \
    --num_eval_samples 500 \
    --N 5 \
    --max_new_tokens 512 &&
python evaluate.py \
    --input_file "$output_path/bon.json" \
    --anno_path "$coco_path/annotations" \
    --output_file "$output_path/result.json"
