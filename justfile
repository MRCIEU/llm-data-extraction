# ==== Variables ====
# py files flatten and delimited by space
PY_FILES := `fd -e py --exclude "*legacy*" --exclude "*tom*" --exclude "*yiutils*"  | tr '\n' ' '`

# list recipes
default:
    @just --list --unsorted

# ==== codebase ====

# sanity-check
[group('codebase')]
sanity:
    #!/bin/bash
    echo {{PY_FILES}}
    micromamba env list | grep data-extraction
    pip list | grep local_funcs
    pip list | grep yiutils
    python scripts/python/check-health.py

# format codebase, need to have ruff in path
[group('codebase')]
format:
    echo "{{PY_FILES}}" | xargs ruff format

# check codebase using ruff check, need to have ruff in path
[group('codebase')]
ruff-check:
    echo "{{PY_FILES}}" | xargs ruff check --fix

[group('codebase')]
ty-check:
    echo "{{PY_FILES}}" | xargs ty check

# test
[group('codebase')]
test:
    python -m pytest

# vscode server
[group('codebase'), group('isb')]
vscode-isb:
    sbatch scripts/isb/vscode.sbatch

# vscode server, 2gpus
[group('codebase'), group('isb')]
vscode-isb-2gpus:
    sbatch scripts/isb/vscode-2gpus.sbatch

# ==== data preprocessing ====

# data preprocessing: mr-pubmed-data.json
[group('data-prep')]
data-prep-mr-pubmed:
    python scripts/python/mr-pubmed-data-prep.py

# ==== local llm batch processing ====

# Perform data extraction, isb, llama3, pilot
[group('isb'), group('devel')]
devel-isb-extract-data-llama3-pilot:
    sbatch scripts/isb/extract-data-llama3-pilot.sbatch

# Perform data extraction, isb, llama3
[group('isb'), group('devel')]
devel-isb-extract-data-llama3:
    sbatch scripts/isb/extract-data-llama3.sbatch

# Perform data extraction, isb, deepseek, pilot
[group('isb'), group('devel')]
devel-isb-extract-data-ds-pilot:
    sbatch scripts/isb/extract-data-ds-pilot.sbatch

# Perform data extraction, isb, deepseek
[group('isb'), group('devel')]
devel-isb-extract-data-ds:
    sbatch scripts/isb/extract-data-ds.sbatch

# Perform data extraction, isb, deepseek-prover, pilot
[group('isb'), group('devel')]
devel-isb-extract-data-ds-prover-pilot:
    sbatch scripts/isb/extract-data-ds-prover-pilot.sbatch

# Perform data extraction, isb, deepseek-prover
[group('isb'), group('devel')]
devel-isb-extract-data-ds-prover:
    sbatch scripts/isb/extract-data-ds-prover.sbatch

# Perform data extraction, isb, llama3.2, pilot
[group('isb'), group('devel')]
devel-isb-extract-data-llama3-2-pilot:
    sbatch scripts/isb/extract-data-llama3-2-pilot.sbatch

# Perform data extraction, isb, llama3.2
[group('isb'), group('devel')]
devel-isb-extract-data-llama3-2:
    sbatch scripts/isb/extract-data-llama3-2.sbatch


# ==== openai model batch processing ====
[group('openai'), group('devel')]
devel-openai-extract-data-o4-mini-pilot:
    python scripts/python/extract-data-openai.py \
        --model o4-mini \
        --pilot \
        --path_data data/intermediate/mr-pubmed-data/mr-pubmed-data-sample.json \
        --output_dir data/intermediate/openai-batch-results/

[group('openai'), group('devel')]
devel-openai-extract-data-gpt-4o-pilot:
    python scripts/python/extract-data-openai.py \
        --model gpt-4o \
        --pilot \
        --path_data data/intermediate/mr-pubmed-data/mr-pubmed-data-sample.json \
        --output_dir data/intermediate/openai-batch-results/

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-o4-mini-lite:
    sbatch scripts/bc4/extract-data-o4-mini-lite.sbatch

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-gpt-4o-lite:
    sbatch scripts/bc4/extract-data-gpt-4o-lite.sbatch

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-o4-mini:
    sbatch scripts/bc4/extract-data-o4-mini.sbatch

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-gpt-4-1:
    sbatch scripts/bc4/extract-data-gpt-4-1.sbatch

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-gpt-4-1-full:
    sbatch scripts/bc4/extract-data-gpt-4-1-full.sbatch

[group('openai'), group('devel'), group('bc4')]
devel-openai-extract-data-gpt-5-full:
    sbatch scripts/bc4/extract-data-gpt-5-full.sbatch

# ==== post-batch processing ====

# Aggregate LLM batch results
[group('data-processing')]
aggregate-llm-batch-results:
    python scripts/python/aggregate-llm-batch-results.py --all

# Process LLM batch results
[group('data-processing')]
process-llm-batch-results:
    python scripts/python/process-llm-aggregated-results.py

# analysis sample: trial
[group('data-processing')]
analysis-sample-trial:
    python scripts/python/make-analysis-sample.py --size 20 --seed 42
    python scripts/python/render-analysis-sample.py --file sample-42-20.json

# analysis sample: formal
[group('data-processing')]
analysis-sample-formal:
    python scripts/python/make-analysis-sample.py --size 100 --seed 42
    python scripts/python/render-analysis-sample.py --file sample-42-100.json
