set dotenv-load

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
    echo "\$ACCOUNT_CODE: ${ACCOUNT_CODE}"
    echo "\$PROJECT_OPENAI_API_KEY: ${PROJECT_OPENAI_API_KEY}"
    echo "\$HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN}"
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

# Focused assessment-score regression checks (no extraction/network calls)
[group('codebase')]
assessment-test:
    MPLBACKEND=Agg python -m pytest -q tests/test_assessment_score_validation.py

# Check changed assessment sources without applying lint fixes
[group('codebase')]
assessment-lint:
    ruff check scripts/python/analysis/process-assessment-data.py scripts/python/analysis/create-manuscript-assets.py tests/test_assessment_score_validation.py

# vscode server
[group('codebase'), group('isb')]
vscode-isb:
    sbatch scripts/isb/vscode.sbatch

# vscode server, 2gpus
[group('codebase'), group('isb')]
vscode-isb-2gpus:
    sbatch scripts/isb/vscode-2gpus.sbatch
