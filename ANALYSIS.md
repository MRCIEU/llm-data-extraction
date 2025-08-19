# Analysis steps

## Data preprocessing / preparation

### Main preprocessing

Convert Gib's raw data into preprocessed data for extraction processing.

```bash
just -f justfile-processing data-prep-mr-pubmed
```

- Raw data: `data/raw/mr-pubmed-abstracts`
- Output: `data/intermediate/mr-pubmed-data`

### Diagnostics (post-processing)

Notebook: `notebooks/analysis-data-prep/mr-data.ipynb`

______________________________________________________________________

## Extraction processing

### Initial exploration

Notebooks: `notebooks/models`

### Data extraction with local LLM (ISB Slurm)

Run on isambard-ai via Slurm job arrays. Use the batch justfile to submit jobs:

- LLaMA 3 (pilot):

  ```bash
  just -f justfile-batch isb-extract-data-llama3-pilot
  ```

- LLaMA 3 (full):

  ```bash
  just -f justfile-batch isb-extract-data-llama3
  ```

- LLaMA 3.2 (pilot):

  ```bash
  just -f justfile-batch isb-extract-data-llama3-2-pilot
  ```

- LLaMA 3.2 (full):

  ```bash
  just -f justfile-batch isb-extract-data-llama3-2
  ```

- DeepSeek-R1 (pilot):

  ```bash
  just -f justfile-batch isb-extract-data-ds-pilot
  ```

- DeepSeek-R1 (full):

  ```bash
  just -f justfile-batch isb-extract-data-ds
  ```

- DeepSeek Prover (pilot):

  ```bash
  just -f justfile-batch isb-extract-data-ds-prover-pilot
  ```

- DeepSeek Prover (full):

  ```bash
  just -f justfile-batch isb-extract-data-ds-prover
  ```

Output logic:

- Initial outputs: `output/isb-ai-{SLURM_ARRAY_JOB_ID}`
- Then moved to: `data/intermediate/llm-results/isb-ai-{SLURM_ARRAY_JOB_ID}`

### Data extraction with OpenAI models

Two ways to run:

1. Pilot runs (small sample, no Slurm):

   - o4-mini pilot:

     ```bash
     just -f justfile-batch openai-extract-data-o4-mini-pilot
     ```

   - gpt-4o pilot:

     ```bash
     just -f justfile-batch openai-extract-data-gpt-4o-pilot
     ```

   These write pilot outputs under: `data/intermediate/openai-batch-results/`

1. BC4 cluster (Slurm):

   - o4-mini (lite/full):

     ```bash
     just -f justfile-batch openai-extract-data-o4-mini-lite
     just -f justfile-batch openai-extract-data-o4-mini
     ```

   - gpt-4-1 (lite/full):

     ```bash
     just -f justfile-batch openai-extract-data-gpt-4-1
     just -f justfile-batch openai-extract-data-gpt-4-1-full
     ```

   - gpt-5 (lite/full):

     ```bash
     just -f justfile-batch openai-extract-data-gpt-5-lite
     just -f justfile-batch openai-extract-data-gpt-5-full
     ```

BC4 job outputs are organized under: `data/intermediate/llm-results/<BC4-JOB-ID>/results/<model>`

______________________________________________________________________

## Post-processing

### Aggregate and process

Aggregate model-specific batch raw results into aggregated raw results:

```bash
just -f justfile-processing aggregate-llm-batch-results
```

- Input: `data/intermediate/llm-results/<EXPERIMENT-ID>/results/*.json`
- Output: `data/intermediate/llm-results-aggregated/<MODEL-NAME>/raw_results.json`

Process aggregated results (global + model-specific processing and schema validation):

```bash
just -f justfile-processing process-llm-batch-results
```

- Input: `data/intermediate/llm-results-aggregated/<MODEL-NAME>/raw_results.json`
- Output: `data/intermediate/llm-results-aggregated/<MODEL-NAME>/processed_results.json` plus
  `processed_results_valid.json` and `processed_results_invalid.json` (schema validation splits)

### Diagnostics

Notebook: `notebooks/analysis-extraction/diagnostics-data-processing.ipynb`

______________________________________________________________________

## Analysis

### Produce analysis samples

Trial (size 20, seed 42):

```bash
just -f justfile-processing analysis-sample-trial
```

Formal (size 100, seed 42):

```bash
just -f justfile-processing analysis-sample-formal
```
