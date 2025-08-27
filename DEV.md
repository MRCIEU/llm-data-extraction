# Development

Overarching principles:

- Unless otherwise specified, run commands at the repo root, with the conda environment activated

## Setting up

### Clone

git clone; git submodule update --init --recursive

### conda env

bootstrap the appropriate conda env

### local packages

inside conda env and at project root, install local packages as the following

```bash
python -m pip install -e src/local_funcs

python -m pip install -e src/yiutils
```

### env variables

Add .env

### (Final.) sanity check

```bash
just sanity
```

______________________________________________________________________

## Technical information

### RDSF archive

```text
/projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction
```

### Project structure and key files

- `data/`: check DATA.md
- `docs/`: docs
- `envs/`: conda envs
- `src/`: local packages
  - `src/local_funcs`: `local_funcs` package
  - `src/yiutils`: `yiutils` package
- `justfile`: task runner for codebase maintenance and utilities
- `justfile-batch`: task runner for batch experiments
- `justfile-processing`: task runner for pre- and post-processing

### conda environment

- `envs/environment.yml` / `data-extraction`: standard gpu env
- `envs/environment-bp1.yml`: / `data-extraction-bp1`: bp1 gpu env
- `envs/environment-non-gpu.yml`: / `data-extraction-non-gpu`: non gpu env for processing

### environment variable specification

Put .env in repo root

- `HUGGINGFACE_TOKEN`: huggingface token for llama3
- `PROJECT_OPENAI_API_KEY`: OpenAI API key
- `ACCOUNT_CODE`: HPC account code

______________________________________________________________________
