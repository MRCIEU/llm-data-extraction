# Data Extraction using LLMs

## Init

git clone; git submodule update --init --recursive

bootstrap conda env

inside conda env and at root, install local packages

- local_funcs: python -m pip install src/local_funcs
- yiutils: python -m pip install yiutils/

## upstream RDSF mount

TODO

## envs

.env in repo root

conda envs
- environment.yml / data-extraction: standard gpu env
- environment-bp1.yml: bp1 gpu env
- environment-non-gpu.yml: non gpu env
