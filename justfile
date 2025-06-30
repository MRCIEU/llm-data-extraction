# Variables
archive_host := "epi-franklin2"
archive_path := "/projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction/data/"
data_archive := shell('echo ' + archive_host + ':' + archive_path)

# list recipes
default:
    @just --list --unsorted

# ==== codebase ====

# sanity-check
[group('codebase')]
check-health:
    #!/bin/bash
    echo "Data archive data_archive: {{data_archive}}"
    micromamba env list | grep data-extraction
    pip list | grep local_funcs
    pip list | grep yiutils

# test
[group('codebase')]
test:
    python -m pytest

# vscode server
[group('codebase')]
[group('isb')]
vscode-isb:
    sbatch scripts/isb/vscode.sbatch

# vscode server, 2gpus
[group('codebase')]
[group('isb')]
vscode-isb-2gpus:
    sbatch scripts/isb/vscode-2gpus.sbatch

# ==== data ====

# data push, dry run
[group('data')]
data-push-dry:
    rsync -aLvzP --delete -n ./data {{data_archive}}

# data push
[group('data')]
data-push:
    rsync -aLvzP --delete ./data {{data_archive}}

# data pull, dry run
[group('data')]
data-pull-dry:
    rsync -aLvzP --delete -n {{data_archive}} ./data/

# data pull
[group('data')]
data-pull:
    rsync -aLvzP --delete {{data_archive}} ./data/

# ==== docs ====

# docs all
[group('docs')]
docs-all: docs-data-archive docs-filetree

# docs about filetree
[group('docs')]
docs-filetree:
    #!/bin/bash
    OUTFILE="./docs/filetree.txt"
    eza -T --git-ignore ./ > $OUTFILE

# docs data archive
[group('docs')]
docs-data-archive:
    #!/bin/bash
    OUTFILE="./docs/data_archive.txt"
    ssh {{archive_host}} "eza -T -L 3 {{archive_path}}" > $OUTFILE

# ==== data preprocessing ====

# data preprocessing: mr-pubmed-data.json
[group('data-prep')]
data-prep-mr-pubmed:
    python scripts/python/mr-pubmed-data-prep.py

# ==== local llm batch processing ====

# Perform data extraction, isb, llama3, pilot
[group('isb')]
[group('devel')]
devel-isb-extract-data-llama3-pilot:
    sbatch scripts/isb/extract-data-llama3-pilot.sbatch

# Perform data extraction, isb, llama3
[group('isb')]
[group('devel')]
devel-isb-extract-data-llama3:
    sbatch scripts/isb/extract-data-llama3.sbatch

# Perform data extraction, isb, deepseek, pilot
[group('isb')]
[group('devel')]
devel-isb-extract-data-ds-pilot:
    sbatch scripts/isb/extract-data-ds-pilot.sbatch

# Perform data extraction, isb, deepseek
[group('isb')]
[group('devel')]
devel-isb-extract-data-ds:
    sbatch scripts/isb/extract-data-ds.sbatch

# Perform data extraction, isb, deepseek-prover, pilot
[group('isb')]
[group('devel')]
devel-isb-extract-data-ds-prover-pilot:
    sbatch scripts/isb/extract-data-ds-prover-pilot.sbatch

# Perform data extraction, isb, deepseek-prover
[group('isb')]
[group('devel')]
devel-isb-extract-data-ds-prover:
    sbatch scripts/isb/extract-data-ds-prover.sbatch

# Perform data extraction, isb, llama3.2, pilot
[group('isb')]
[group('devel')]
devel-isb-extract-data-llama3-2-pilot:
    sbatch scripts/isb/extract-data-llama3-2-pilot.sbatch

# Perform data extraction, isb, llama3.2
[group('isb')]
[group('devel')]
devel-isb-extract-data-llama3-2:
    sbatch scripts/isb/extract-data-llama3-2.sbatch


# ==== openai model batch processing ====
[group('openai')]
[group('devel')]
[group('local')]
devel-openai-extract-data-lite:
    python scripts/python/extract-data-openai.py \
        --models o3-mini \
        --path_data data/intermediate/mr-pubmed-data/mr-pubmed-data-sample-lite.json \
        --output_dir data/intermediate/openai-batch-results/

# TODO: do it on epi-franklin2
[group('openai')]
[group('devel')]
devel-openai-extract-data:
    python scripts/python/extract-data-openai.py \
        --models o3-mini gpt-4o \
        --pilot \
        --input data/intermediate/mr-pubmed-data/mr-pubmed-data-sample.json \
        --output data/intermediate/openai-batch-results/

# ==== post-batch processing ====

# Aggregate LLM batch results
[group('data-processing')]
aggregate-llm-batch-results:
    python scripts/python/aggregate-llm-batch-results.py

# Process LLM batch results
[group('data-processing')]
process-llm-batch-results:
    python scripts/python/process-llm-aggregated-results.py

# analysis sample: trial
[group('data-processing')]
analysis-sample-trial:
    python scripts/python/make-analysis-sample.py --size 20 --seed 42
    python scripts/python/render-analysis-sample.py --file sample-42-20.json
