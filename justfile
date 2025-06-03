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

# ==== development ====

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
