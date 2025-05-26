# Variables
archive_host := "epi-franklin2"
archive_path := "/projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction/"
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

# data sync, dry run
[group('data')]
data-sync-dry:
    rsync -aLvzP --delete -n ./data {{data_archive}}

# data sync
[group('data')]
data-sync:
    rsync -aLvzP --delete ./data {{data_archive}}

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
    python scripts/local/mr-pubmed-data-prep.py

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
