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
