# Analysis steps

last updated: 2025-10-21

## Overview

This document describes the analysis workflow for LLM-based data extraction.
The workflow consists of three main phases:

1. Data preprocessing and preparation
2. Extraction processing (via local LLMs or OpenAI models)
3. Post-processing and analysis

Two parallel workflows exist:

1. Standard workflow: Full dataset with chunked processing
2. Special sample (SP) workflow: Targeted subset for detailed review

```mermaid
flowchart TD
    A[Raw MR PubMed Data] --> B[Preprocessing]
    B --> C[mr-pubmed-data.json]

    C --> D[Standard Extraction<br/>ISB/BC4]

    C --> L[Analysis Sampling<br/>100 PMIDs]
    L --> E[Special Sample Prep]
    E --> F[special-sample.json]
    F --> G[SP Extraction<br/>Local OpenAI]

    D --> H[Standard Aggregation]
    G --> I[SP Aggregation]

    H --> J[Standard Processing<br/>& Validation]
    I --> K[SP Processing<br/>& Validation]

    J --> M[Analysis Sampling<br/>trial/formal]
    K --> N[SP Analysis<br/>all results]

    style D fill:#e1f5ff
    style G fill:#ffe1e1
    style H fill:#e1f5ff
    style I fill:#ffe1e1
    style J fill:#e1f5ff
    style K fill:#ffe1e1
    style M fill:#e1f5ff
    style N fill:#ffe1e1
```

Legend:

- Blue: Standard workflow
- Red: Special sample (SP) workflow

______________________________________________________________________

## Data preprocessing / preparation

### Main preprocessing

Convert Gib's raw data into preprocessed data for extraction processing.

```bash
just -f justfile-processing data-prep-mr-pubmed
```

Script: `scripts/python/preprocessing/mr-pubmed-data-prep.py`

- No command-line flags
- Raw data: `data/raw/mr-pubmed-abstracts`
- Output: `data/intermediate/mr-pubmed-data`

Processing details:

- Generates `mr-pubmed-data.json` (full dataset)
- Generates `mr-pubmed-data-sample.json` (sample for pilots)
- Generates `mr-pubmed-data-sample-lite.json` (smaller sample for quick checks)
- Creates dated backups in `backups/` subdirectory

### Special sample preparation

Create a special sample for review analysis by filtering the full dataset to include only PMIDs from the formal analysis sample.

```bash
just -f justfile-processing prep-special-sample
```

Script: `scripts/python/preprocessing/prep-special-sample.py`

- No command-line flags
- Input: `data/intermediate/analysis-sample/sample-42-100.json`
- Base data: `data/intermediate/mr-pubmed-data/mr-pubmed-data.json`
- Output: `data/intermediate/mr-pubmed-data/special-sample.json`

This special sample is used for targeted re-extraction with OpenAI models.

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

#### Standard batch extraction (chunked processing)

Script: `scripts/python/batch/extract-data-openai.py`

Available flags:

- `--output_dir`: Directory to save output JSON (default: `output` in project root)
- `--path_data`: Path to MR PubMed abstracts data (default: `data/intermediate/mr-pubmed-data/mr-pubmed-data-sample.json`)
- `--array-id`: Array ID for chunking (default: 0)
- `--array-length`: Number of chunks to split data into (default: 30)
- `--pilot`: Enable pilot mode (processes 5 documents)
- `--model`: Model to use (required, choices: o4-mini, gpt-4o, gpt-4-1, gpt-5)
- `--dry-run`: Print config and schema data then exit without processing

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

2. BC4 cluster (Slurm):

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

#### Special sample extraction (SP workflow)

Script: `scripts/python/batch/extract-data-openai-sp.py`

Available flags:

- `--output_dir`: Directory to save output JSON (default: `output` in project root)
- `--path_data`: Path to MR PubMed abstracts data (default: `data/intermediate/mr-pubmed-data/special-sample.json`)
- `--pilot`: Enable pilot mode (processes 5 documents)
- `--model`: Model to use (required, choices: o4-mini, gpt-4o, gpt-4-1, gpt-5, gpt-5-mini)
- `--dry-run`: Print config and schema data then exit without processing

The SP workflow processes the entire special sample in one go without chunking, using legacy prompts without schema in the extraction phase.

1. Pilot run (local, gpt-5-mini, 5 documents):

   ```bash
   just -f justfile-batch openai-sp-extract-pilot
   ```

2. Full batch run (local, all models):

   ```bash
   just -f justfile-batch openai-sp-extract-batch
   ```

   This processes the special sample with: gpt-5, gpt-5-mini, o4-mini, gpt-4o, gpt-4-1

Output structure:

- Input: `data/intermediate/mr-pubmed-data/special-sample.json`
- Output directory: `data/intermediate/openai-sp-batch-results/<model>/`
- Output file per model: `mr_extract_openai_sp.json` or `mr_extract_openai_sp_pilot.json`

Key differences from standard extraction:

- Processes all data in one go without chunking
- Uses legacy prompt templates (make_message_metadata, make_message_results)
- Does not use schema during extraction
- Stores both metadata and results completions in single JSON file

Processing flow:

```mermaid
flowchart LR
    A[special-sample.json] --> B[extract-data-openai-sp.py]
    B --> C[mr_extract_openai_sp.json]
    C --> D[aggregate-llm-batch-results-sp.py]
    D --> E[raw_results.json]
    E --> F[process-llm-aggregated-results-sp.py]
    F --> G[processed_results.json]
    F --> H[processed_results_valid.json]
    F --> I[processed_results_invalid.json]
```

______________________________________________________________________

## Post-processing

### Standard batch post-processing

#### Aggregate results

Script: `scripts/python/postprocessing/aggregate-llm-batch-results.py`

Available flags:

- `--models`: List of models to process (space separated, choices: deepseek-r1-distilled, llama3, llama3-2, o4-mini, gpt-4-1)
- `--all`: Process all models

Aggregate model-specific batch raw results into aggregated raw results:

```bash
just -f justfile-processing aggregate-llm-batch-results
```

- Input: `data/intermediate/llm-results/<EXPERIMENT-ID>/results/*.json`
- Output: `data/intermediate/llm-results-aggregated/<MODEL-NAME>/raw_results.json`

Supported models and their experiment IDs (hardcoded in script):

- deepseek-r1-distilled: isb-ai-117256
- llama3-2: isb-ai-117535
- llama3: isb-ai-116732
- o4-mini: bc4-12398167
- gpt-4-1: bc4-12459870

Data flow:

```mermaid
flowchart TD
    A[llm-results/isb-ai-*/results/*.json] --> B[aggregate-llm-batch-results]
    C[llm-results/bc4-*/results/<model>/*.json] --> B
    B --> D[llm-results-aggregated/<model>/raw_results.json]
```

#### Process and validate results

Script: `scripts/python/postprocessing/process-llm-aggregated-results.py`

Available flags:

- No command-line flags (processes all available models)

Process aggregated results (global + model-specific processing and schema validation):

```bash
just -f justfile-processing process-llm-batch-results
```

- Input: `data/intermediate/llm-results-aggregated/<MODEL-NAME>/raw_results.json`
- Output:
  - `data/intermediate/llm-results-aggregated/<MODEL-NAME>/processed_results.json`
  - `data/intermediate/llm-results-aggregated/<MODEL-NAME>/processed_results_valid.json`
  - `data/intermediate/llm-results-aggregated/<MODEL-NAME>/processed_results_invalid.json`
  - `data/intermediate/llm-results-aggregated/logs/<MODEL-NAME>_schema_validation_errors.log`

Processing steps:

1. Parse JSON from LLM completions
2. Extract and normalize metadata and results
3. Apply schema remapping for common key variations
4. Validate against JSON schemas
5. Split into valid and invalid subsets

Key remappings applied:

- "95% confidence interval" → "95% CI"
- `"95%_CI"` → "95% CI"
- "standard error" → "SE"
- "odds_ratio" → "odds ratio"
- "hazard_ratio" → "hazard ratio"
- "Direction" → "direction"

```mermaid
flowchart TD
    A[raw_results.json] --> B[Parse JSON]
    B --> C[Extract metadata/results]
    C --> D[Normalize keys]
    D --> E[Schema validation]
    E --> F[processed_results.json]
    E --> G[processed_results_valid.json]
    E --> H[processed_results_invalid.json]
    E --> I[validation errors log]
```

### Special sample (SP) post-processing

#### Aggregate SP results

Script: `scripts/python/postprocessing/aggregate-llm-batch-results-sp.py`

Available flags:

- `--models`: List of models to process (space separated, choices: o4-mini, gpt-4o, gpt-4-1, gpt-5, gpt-5-mini)
- `--all`: Process all models

Aggregate OpenAI SP batch results:

```bash
just -f justfile-processing aggregate-llm-batch-results-sp
```

- Input: `data/intermediate/openai-sp-batch-results/<MODEL-NAME>/mr_extract_openai_sp.json`
- Output: `data/intermediate/llm-results-aggregated-sp/<MODEL-NAME>/raw_results.json`

Supported models: o4-mini, gpt-4o, gpt-4-1, gpt-5, gpt-5-mini

#### Process and validate SP results

Script: `scripts/python/postprocessing/process-llm-aggregated-results-sp.py`

Available flags:

- No command-line flags (processes all available models)

Process and validate SP results:

```bash
just -f justfile-processing process-llm-batch-results-sp
```

- Input: `data/intermediate/llm-results-aggregated-sp/<MODEL-NAME>/raw_results.json`
- Output:
  - `data/intermediate/llm-results-aggregated-sp/<MODEL-NAME>/processed_results.json`
  - `data/intermediate/llm-results-aggregated-sp/<MODEL-NAME>/processed_results_valid.json`
  - `data/intermediate/llm-results-aggregated-sp/<MODEL-NAME>/processed_results_invalid.json`
  - `data/intermediate/llm-results-aggregated-sp/logs/<MODEL-NAME>_schema_validation_errors.log`

Processing steps:

1. Parse JSON from completion_metadata and completion_results fields
2. Process metadata (extract nested structures, remove metainformation)
3. Process results (extract nested structures, remap common key variations)
4. Validate against JSON schemas
5. Split into valid and invalid subsets

Key remappings applied (same as standard workflow):

- "95% confidence interval" → "95% CI"
- `"95%_CI"` → "95% CI"
- "standard error" → "SE"
- "odds_ratio" → "odds ratio"
- "hazard_ratio" → "hazard ratio"
- "Direction" → "direction"

### Diagnostics

Optional diagnostic notebook

`notebooks/analysis-extraction/diagnostics-data-processing.ipynb`

______________________________________________________________________

## Analysis

### Standard analysis samples

#### Sample generation

Script: `scripts/python/analysis/make-analysis-sample.py`

Available flags:

- `--size`: Number of samples to produce (default: 20)
- `--seed`: Random seed (default: 42)

Supported models (hardcoded): llama3, llama3-2, deepseek-r1-distilled, o4-mini, gpt-4-1

Trial sample (size 20, seed 42):

```bash
just -f justfile-processing analysis-sample-trial
```

Formal sample (size 100, seed 42):

```bash
just -f justfile-processing analysis-sample-formal
```

These samples select PMIDs that are commonly available across all models and randomly sample from them.

#### HTML rendering

Script: `scripts/python/analysis/render-analysis-sample.py`

Available flags:

- `-f, --file`: Path to input JSON file, relative to `data/intermediate/analysis-sample/` (required)

Output:

- JSON: `data/intermediate/analysis-sample/sample-42-{20,100}.json`
- HTML: `data/intermediate/analysis-sample/sample-42-{20,100}.html`

The HTML rendering produces an interactive visualization with tabbed model results for review.

### Special sample (SP) analysis

#### Sample generation

Script: `scripts/python/analysis/make-analysis-sample-sp.py`

Available flags:

- No command-line flags

Supported models (hardcoded): o4-mini, gpt-4o, gpt-4-1, gpt-5, gpt-5-mini

Process all SP results without sampling:

```bash
just -f justfile-processing analysis-sample-sp
```

This aggregates all processed valid results from the SP workflow across available models for the special sample PMIDs.

#### HTML rendering

Script: `scripts/python/analysis/render-analysis-sample-sp.py`

Available flags:

- `-f, --file`: Path to input JSON file, relative to `data/intermediate/analysis-sample-sp/` (required)

Output:

- JSON: `data/intermediate/analysis-sample-sp/all-results.json`
- HTML: `data/intermediate/analysis-sample-sp/all-results.html`

The HTML rendering produces an interactive visualization with tabbed model results for review.

Analysis data structure:

```json
[
  {
    "pubmed_data": {
      "pmid": "...",
      "ab": "...",
      ...
    },
    "model_results": {
      "o4-mini": {
        "pmid": "...",
        "metadata": {...},
        "results": [...]
      },
      "gpt-4o": {...},
      ...
    }
  }
]
```
