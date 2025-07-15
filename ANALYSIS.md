# Analysis steps

Overarching principles:
- Unless otherwise specified, run things at repo root

---
# data preprocessing / preparation stage

## main preprocessing

convert Gib's raw data into preprocessed data for extraction processing

```
just data-prep-mr-pubmed
```

- raw data: `data/raw/mr-pubmed-abstracts`
- output: `data/intermediate/mr-pubmed-data`

## diagnostics, etc

`notebooks/analysis-data-prep/mr-data.ipynb`

# Extraction processing

## initial exploration

`notebooks/models`

## data extraction with local LLM

Run on isambard-ai in slurm job arrays

Output logic
- Output is initially stored in `output/isb-ai-{SLURM_ARRAY_JOB_ID}`
- then moved to `data/intermediate/llm-results/isb-ai-{SLURM_ARRAY_JOB_ID}`

### llama3

pilot

> just devel-isb-extract-data-llama3-pilot

full

> just devel-isb-extract-data-llama3

### llama3.2

pilot

> just devel-isb-extract-data-llama3-2-pilot

full

> just devel-isb-extract-data-llama3-2

### deepseek-r1

pilot

> just devel-isb-extract-data-ds-pilot

full

> just devel-isb-extract-data-ds

## data extraction with openai models

TODO: update this

# Post processing

## aggregate and process

> just aggregate-llm-batch-results

aggregate model-specific batch raw results into aggregated raw results
- input: `data / intermediate / llm-results / <EXPERIMENT-ID> / results / *.json`
- output: `data / intermediate / llm-results-aggregated / <MODEL-NAME> / raw_results.json`

> just process-llm-batch-results

Process raw results with global and model-specific processing

- input: `data / intermediate / <MODEL-NAME> / raw_results.json`.

- output: `data / intermediate / <MODEL-NAME> / processed_results.json`

## Diagnostics

`notebooks/analysis-extraction/diagnostics-data-processing.ipynb`

## produce analysis samples

> just analysis-sample-trial

generate analysis sample for size 20 and seed 42

TODO: full analysis just steps
