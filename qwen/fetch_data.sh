#!/bin/bash
# Download the evaluation data into qwen/data/ (MATH-500 from HuggingFaceH4, AIME 2025 from MathArena).
set -e; cd "$(dirname "$0")/data"
curl -sfL https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl -o math500.jsonl
curl -sfL https://huggingface.co/api/datasets/MathArena/aime_2025/parquet/default/train/0.parquet -o aime25.parquet
python3 -c "
import json, pyarrow.parquet as pq
json.dump(pq.read_table('aime25.parquet').to_pylist(), open('aime25.json', 'w'), indent=1, default=str)"
rm aime25.parquet; wc -l math500.jsonl; echo 'aime25.json written'
