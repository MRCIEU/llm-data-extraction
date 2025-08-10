# Data Extraction using LLMs

---

# Setting up

## Meta info

- location of RDSF archive:
  /projects/MRC-IEU/research/projects/ieu3/p3/015/working/data/llm-data-extraction

## Clone
git clone; git submodule update --init --recursive

## conda env
bootstrap conda env

## local packages
inside conda env and at root, install local packages

- local_funcs (src/local_funcs/src/local_funcs): python -m pip install -e src/local_funcs
- yiutils (src/yiutils/src/yiutils): python -m pip install -e src/yiutils

sanity check: run `pip list | less` and check installation paths

## env variables
Add .env

---

# Technical information

## conda environment

conda envs
- docs/environment.yml / data-extraction: standard gpu env
- docs/environment-bp1.yml / data-extraction-bp1: bp1 gpu env
- docs/environment-non-gpu.yml / data-extraction-non-gpu: non gpu env for processing

## environment variables

.env in repo root

- `HUGGINGFACE_TOKEN`: huggingface token for llama3
- `PROJECT_OPENAI_API_KEY`: OpenAI API key
- `ACCOUNT_CODE`: HPC account code

---

# Other information

- For details about analysis refer to `./ANALYSIS.md`
- For details about data refer to `./DATA.md`
