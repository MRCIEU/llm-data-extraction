# Dev info

# Meta info

- location of RDSF archive:
  /projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction

# Setting up

git clone; git submodule update --init --recursive

bootstrap conda env

inside conda env and at root, install local packages

- local_funcs (src/local_funcs/src/local_funcs): python -m pip install -e src/local_funcs
- yiutils (src/yiutils/src/yiutils): python -m pip install -e src/yiutils

Add .env

# Technical information

## conda environment

conda envs
- ~docs/environment.yml~ / ~data-extraction~: standard gpu env
- ~docs/environment-bp1.yml~ / ~data-extraction-bp1~: bp1 gpu env
- ~docs/environment-non-gpu.yml~ / ~data-extraction-non-gpu~: non gpu env

## environment variables

.env in repo root

- ~HUGGINGFACE_TOKEN~: huggingface token for llama3
- ~OPENAI_API_KEY~: OpenAI API key

* analysis

Overarching principles:
- Unless otherwise specified, run things at repo root

## Key logics

### Data preparation

### LLM

- isb
  - scripts/isb/extract-data-pilot.sbatch: pilot run as an entrypoint
  - scripts/isb/extract-data.sbatch: Proper array job for producing data extraction using llama3-8B-Instruct
  - scripts/isb/TODO1: perform extraction on TODO1 data
- bp1
  - `LATER`
- local
  - `LATER`
