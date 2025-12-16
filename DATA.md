# Data organization

Last updated: 2025-10-21

This document describes how data is organized and produced in this repository based on the current filesystem and workflow recipes.
It covers both the standard batch extraction workflow and the special sample (SP) workflow for targeted re-extraction.

## Overview

Data flows through these stages:

1. Raw inputs: MR PubMed abstracts and reference data
2. Preprocessing: curate and structure MR abstracts for extraction
3. Extraction: batch LLM runs (ISB for local models, BC4 for OpenAI models)
4. Aggregation: combine raw JSON outputs per model
5. Processing: parse, validate against schema, and split valid/invalid
6. Analysis: produce cross-model samples for inspection and downstream analysis
7. Manuscript assets: generate publication-ready figures and tables

```mermaid
flowchart LR
    A[Raw<br/>PubMed] --> B[Preprocessing]
    B --> C[Intermediate<br/>Prepared Data]
    C --> D[Extraction<br/>ISB/BC4]
    D --> E[Intermediate<br/>Raw Results]
    E --> F[Aggregation]
    F --> G[Intermediate<br/>Aggregated]
    G --> H[Processing &<br/>Validation]
    H --> I[Intermediate<br/>Valid/Invalid]
    I --> J[Analysis<br/>Sampling]
    J --> K[Assessment<br/>Review]
    K --> L[Artifacts<br/>Manuscript]
    
    style A fill:#e8e8e8
    style C fill:#cce5ff
    style E fill:#cce5ff
    style G fill:#cce5ff
    style I fill:#cce5ff
    style L fill:#d4f4dd
```

### Standard workflow

High-level flow:

```text
data/raw/mr-pubmed-abstracts
  ↓ preprocessing
data/intermediate/mr-pubmed-data
  ↓ batch extraction (chunked)
data/intermediate/llm-results/<JOB-ID>/results
  ↓ aggregation
data/intermediate/llm-results-aggregated/<MODEL>
  ↓ analysis sampling
data/intermediate/analysis-sample
```

### Special sample (SP) workflow

The SP workflow processes a targeted subset for detailed review:

```text
data/intermediate/analysis-sample/sample-42-100.json
  ↓ filter PMIDs
data/intermediate/mr-pubmed-data/special-sample.json
  ↓ SP extraction (full dataset, no chunking)
data/intermediate/openai-sp-batch-results/<MODEL>
  ↓ SP aggregation
data/intermediate/llm-results-aggregated-sp/<MODEL>
  ↓ SP analysis (all results, no sampling)
data/intermediate/analysis-sample-sp
```

Notes:

- Some runs briefly write under `output/` and then move into `data/intermediate/llm-results/<JOB-ID>`.
- JSON Schemas for processed outputs are in `data/assets/data-schema/`.
- The SP workflow uses legacy prompts without schema during extraction but applies schema validation during processing.

______________________________________________________________________

## Raw data (data/raw)

### mr-pubmed-abstracts/

Source: [MRCIEU/mr-pubmed-abstracts](https://github.com/MRCIEU/mr-pubmed-abstracts)

Contains PubMed abstracts and reference tables used during preprocessing.

______________________________________________________________________

## Intermediate (data/intermediate)

### mr-pubmed-data/

Preprocessed MR abstracts ready for LLM extraction.

Produced by the preprocessing recipe (see `ANALYSIS.md`; recipe: `data-prep-mr-pubmed`).

Files:

- `mr-pubmed-data.json` — Full preprocessed dataset
- `mr-pubmed-data-sample.json` — Sample for pilots
- `mr-pubmed-data-sample-lite.json` — Smaller sample for quick checks
- `special-sample.json` — Filtered dataset containing only PMIDs from sample-42-100.json (for SP workflow)
- `backups/<YYYY-MM-DD>/mr-pubmed-data.json` — Dated backups of full dataset

### llm-results/{JOB_ID}/

Raw batch extraction outputs grouped by scheduler job ID.

- ISB jobs for local models: IDs like `isb-ai-111542`, `isb-ai-116732`, `isb-ai-117256`, `isb-ai-117535`, `isb-ai-117536`
- BC4 jobs for OpenAI models: IDs like `bc4-12389832`, `bc4-12390298`, `bc4-12390309`, `bc4-12391182`, `bc4-12391186`, `bc4-12398167`, `bc4-12411151`, `bc4-12414116`

Inside each job directory:

- `logs/` — Slurm logs and per-array outputs
- `README` or `README.md` — Job metadata
- `results/` — Raw JSON results; for OpenAI jobs, nested by model name (e.g., `results/o4-mini/…`)

### llm-results-aggregated/{MODEL}/

Aggregated and processed outputs per model from standard batch extraction.

Produced by post-processing recipes (see `ANALYSIS.md`; `aggregate-llm-batch-results` then `process-llm-batch-results`).

Per model, you will find:

- `raw_results.json` — Aggregated raw array outputs across job IDs
- `processed_results.json` — Parsed, normalized results
- `processed_results_valid.json` — Subset passing schema validation
- `processed_results_invalid.json` — Subset failing schema validation

Available model directories:

- `deepseek-r1-distilled/`
- `gpt-4-1/`
- `gpt-4o/`
- `llama3/`
- `llama3-2/`
- `o4-mini/`

Validation logs:

- `logs/*_schema_validation_errors.log` — Per-model schema error summaries

Backups of earlier aggregations are under `intermediate/_backup/llm-results-aggregated-<DATE>/`.

### llm-results-aggregated-sp/{MODEL}/

Aggregated and processed outputs per model from special sample (SP) extraction.

Produced by SP post-processing recipes (see `ANALYSIS.md`; `aggregate-llm-batch-results-sp` then `process-llm-batch-results-sp`).

Per model, you will find:

- `raw_results.json` — Aggregated SP extraction results
- `processed_results.json` — Parsed, normalized results
- `processed_results_valid.json` — Subset passing schema validation
- `processed_results_invalid.json` — Subset failing schema validation

Available model directories (when processed):

- `o4-mini/`
- `gpt-4o/`
- `gpt-4-1/`
- `gpt-5/`
- `gpt-5-mini/`

Validation logs:

- `logs/*_schema_validation_errors.log` — Per-model schema error summaries

### analysis-sample/

Model-comparison samples for analysis and visual inspection from standard batch extraction.

Produced by `analysis-sample-trial` (e.g., 20 items) and `analysis-sample-formal` (e.g., 100 items).

Files:

- `sample-42-20.json/.html` — Trial sample (seed 42, 20 PMIDs)
- `sample-42-100.json/.html` — Formal sample (seed 42, 100 PMIDs)

### analysis-sample-sp/

Model-comparison data for the special sample (SP) workflow.

Produced by `analysis-sample-sp` recipe.

Files:

- `all-results.json/.html` — All SP results without sampling (includes all PMIDs from special-sample.json that have valid results across available models)

### openai-batch-results/

Outputs from small pilot runs executed locally (primarily OpenAI models before cluster jobs), for example:

- `o4-mini/mr_extract_openai_array_0_pilot.json`

### openai-sp-batch-results/

Outputs from special sample (SP) extraction using OpenAI models.

Produced by `openai-sp-extract-pilot` and `openai-sp-extract-batch` recipes.

Structure:

- `<MODEL-NAME>/mr_extract_openai_sp.json` — Full SP extraction results
- `<MODEL-NAME>/mr_extract_openai_sp_pilot.json` — Pilot SP extraction results (5 documents)

Available model directories (when processed):

- `o4-mini/`
- `gpt-4o/`
- `gpt-4-1/`
- `gpt-5/`
- `gpt-5-mini/`

______________________________________________________________________

## Assets (data/assets)

### assessment-reviews/

Raw assessment data from independent reviewers evaluating LLM extraction performance.

Files:

- `assessment-reviewer-1.xlsx` — Reviewer 1 assessments (multi-sheet Excel file, one sheet per model)
- `assessment-reviewer-2.csv` — Reviewer 2 assessments (single CSV file with all models)
- `reviewer-1-sheets.txt` — List of sheet names in reviewer 1 Excel file
- `reviewer-1-columns.txt` — Column names for reviewer 1 data

These raw assessment files are processed by `process-assessment-data.py` to produce consolidated results.

### data-schema/

Schemas and example data for validating processed results.

- `processed_results/metadata.schema.json`
- `processed_results/results.schema.json`
- `example-data/` — Example payloads and schemas (`metadata.json`, `results.json`, etc.)

______________________________________________________________________

## Artifacts (data/artifacts)

Reserved for publication-ready exports and derived artifacts.

### assessment-results/

Processed assessment data from independent reviewer evaluations of LLM extraction performance.

Produced by: `scripts/python/analysis/process-assessment-data.py` (see `ANALYSIS.md`)

Files:

- `assessment-results-numeric.csv` — Numeric assessment scores with pmid and model identifiers
- `assessment-results-strings.csv` — String assessment data including free-text comments and question responses (Q-*-b-3, Q-*-c-3 columns)

______________________________________________________________________

## Operational notes

- Extraction orchestration: See DEV.md for `justfile-batch` task runner usage
- Post-processing workflows: See DEV.md for `justfile-processing` task runner usage
- Complete workflow details: See ANALYSIS.md for step-by-step procedures
- Transient outputs may appear under `output/` during runs before being moved to `data/intermediate/llm-results/<JOB-ID>`
- Manuscript assets generation: See docs/manuscript-assets.md for detailed documentation

### Workflow comparison

| Feature             | Standard Workflow                 | SP Workflow                          |
| ------------------- | --------------------------------- | ------------------------------------ |
| Input data          | Full mr-pubmed-data.json          | special-sample.json (filtered)       |
| Processing          | Chunked (array jobs)              | Full dataset, no chunking            |
| Prompt strategy     | Schema-based prompts              | Legacy prompts (no schema)           |
| Extraction location | ISB (local models), BC4 (OpenAI)  | Local (OpenAI only)                  |
| Raw output location | `llm-results/<JOB-ID>/`           | `openai-sp-batch-results/<MODEL>/`   |
| Aggregated location | `llm-results-aggregated/<MODEL>/` | `llm-results-aggregated-sp/<MODEL>/` |
| Analysis output     | `analysis-sample/` (sampled)      | `analysis-sample-sp/` (all results)  |
| Purpose             | Broad model comparison            | Detailed review of specific PMIDs    |

______________________________________________________________________

## Tree snapshot

Regenerate a tree of the current data directory when needed:

```text
eza -T ./data
```

Full tree snapshot at the time of writing:

```text
./data
├── artifacts
├── assets
│   └── data-schema
│       ├── example-data
│       │   ├── metadata.json
│       │   ├── metadata.schema.json
│       │   ├── results.json
│       │   └── results.schema.json
│       └── processed_results
│           ├── metadata.schema.json
│           └── results.schema.json
├── intermediate
│   ├── _backup
│   │   ├── llm-results-aggregated-2025-06-17
│   │   │   ├── deepseek-r1-distilled
│   │   │   │   ├── deepseek-r1-sample.html
│   │   │   │   ├── processed_results.json
│   │   │   │   ├── processed_results_filtered.json
│   │   │   │   ├── processed_results_sample.json
│   │   │   │   └── raw_results.json
│   │   │   ├── llama3
│   │   │   │   ├── llama3-sample.html
│   │   │   │   ├── processed_results.json
│   │   │   │   ├── processed_results_sample.json
│   │   │   │   └── raw_results.json
│   │   │   └── llama3-2
│   │   │       ├── llama3-2-sample.html
│   │   │       ├── processed_results.json
│   │   │       ├── processed_results_filtered.json
│   │   │       ├── processed_results_sample.json
│   │   │       └── raw_results.json
│   │   └── llm-results-aggregated-2025-06-18
│   │       ├── deepseek-r1-distilled
│   │       │   ├── processed_results.json
│   │       │   └── raw_results.json
│   │       ├── llama3
│   │       │   ├── processed_results.json
│   │       │   └── raw_results.json
│   │       └── llama3-2
│   │           ├── processed_results.json
│   │           └── raw_results.json
│   ├── analysis-sample
│   │   ├── sample-42-20.html
│   │   ├── sample-42-20.json
│   │   ├── sample-42-100.html
│   │   └── sample-42-100.json
│   ├── llm-results
│   │   ├── bc4-12389832
│   │   │   ├── logs
│   │   │   │   ├── script-12389832.out
│   │   │   │   └── slurm-12389832_0.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── o4-mini
│   │   │           └── mr_extract_openai_array_0.json
│   │   ├── bc4-12390298
│   │   │   ├── logs
│   │   │   │   ├── script-12390298.out
│   │   │   │   ├── slurm-12390298_0.out
│   │   │   │   ├── slurm-12390298_1.out
│   │   │   │   ├── slurm-12390298_2.out
│   │   │   │   ├── slurm-12390298_3.out
│   │   │   │   ├── slurm-12390298_4.out
│   │   │   │   ├── slurm-12390298_5.out
│   │   │   │   ├── slurm-12390298_6.out
│   │   │   │   ├── slurm-12390298_7.out
│   │   │   │   ├── slurm-12390298_8.out
│   │   │   │   ├── slurm-12390298_9.out
│   │   │   │   ├── slurm-12390298_10.out
│   │   │   │   ├── slurm-12390298_11.out
│   │   │   │   ├── slurm-12390298_12.out
│   │   │   │   ├── slurm-12390298_13.out
│   │   │   │   ├── slurm-12390298_14.out
│   │   │   │   ├── slurm-12390298_15.out
│   │   │   │   ├── slurm-12390298_16.out
│   │   │   │   ├── slurm-12390298_17.out
│   │   │   │   ├── slurm-12390298_18.out
│   │   │   │   ├── slurm-12390298_19.out
│   │   │   │   ├── slurm-12390298_20.out
│   │   │   │   ├── slurm-12390298_21.out
│   │   │   │   ├── slurm-12390298_22.out
│   │   │   │   ├── slurm-12390298_23.out
│   │   │   │   ├── slurm-12390298_24.out
│   │   │   │   ├── slurm-12390298_25.out
│   │   │   │   ├── slurm-12390298_26.out
│   │   │   │   ├── slurm-12390298_27.out
│   │   │   │   ├── slurm-12390298_28.out
│   │   │   │   └── slurm-12390298_29.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── o4-mini
│   │   │           ├── mr_extract_openai_array_0.json
│   │   │           ├── mr_extract_openai_array_1.json
│   │   │           ├── mr_extract_openai_array_2.json
│   │   │           ├── mr_extract_openai_array_3.json
│   │   │           ├── mr_extract_openai_array_4.json
│   │   │           ├── mr_extract_openai_array_5.json
│   │   │           ├── mr_extract_openai_array_6.json
│   │   │           ├── mr_extract_openai_array_7.json
│   │   │           ├── mr_extract_openai_array_8.json
│   │   │           ├── mr_extract_openai_array_9.json
│   │   │           ├── mr_extract_openai_array_10.json
│   │   │           ├── mr_extract_openai_array_11.json
│   │   │           ├── mr_extract_openai_array_12.json
│   │   │           ├── mr_extract_openai_array_13.json
│   │   │           ├── mr_extract_openai_array_14.json
│   │   │           ├── mr_extract_openai_array_15.json
│   │   │           ├── mr_extract_openai_array_16.json
│   │   │           ├── mr_extract_openai_array_17.json
│   │   │           ├── mr_extract_openai_array_18.json
│   │   │           ├── mr_extract_openai_array_19.json
│   │   │           ├── mr_extract_openai_array_20.json
│   │   │           ├── mr_extract_openai_array_21.json
│   │   │           ├── mr_extract_openai_array_22.json
│   │   │           ├── mr_extract_openai_array_23.json
│   │   │           ├── mr_extract_openai_array_24.json
│   │   │           ├── mr_extract_openai_array_25.json
│   │   │           ├── mr_extract_openai_array_26.json
│   │   │           ├── mr_extract_openai_array_27.json
│   │   │           ├── mr_extract_openai_array_28.json
│   │   │           └── mr_extract_openai_array_29.json
│   │   ├── bc4-12390309
│   │   │   ├── logs
│   │   │   │   ├── script-12390309.out
│   │   │   │   ├── slurm-12390309_0.out
│   │   │   │   ├── slurm-12390309_1.out
│   │   │   │   ├── slurm-12390309_2.out
│   │   │   │   ├── slurm-12390309_3.out
│   │   │   │   ├── slurm-12390309_4.out
│   │   │   │   ├── slurm-12390309_5.out
│   │   │   │   ├── slurm-12390309_6.out
│   │   │   │   ├── slurm-12390309_7.out
│   │   │   │   ├── slurm-12390309_8.out
│   │   │   │   ├── slurm-12390309_9.out
│   │   │   │   ├── slurm-12390309_10.out
│   │   │   │   ├── slurm-12390309_11.out
│   │   │   │   ├── slurm-12390309_12.out
│   │   │   │   ├── slurm-12390309_13.out
│   │   │   │   ├── slurm-12390309_14.out
│   │   │   │   ├── slurm-12390309_15.out
│   │   │   │   ├── slurm-12390309_16.out
│   │   │   │   ├── slurm-12390309_17.out
│   │   │   │   ├── slurm-12390309_18.out
│   │   │   │   ├── slurm-12390309_19.out
│   │   │   │   ├── slurm-12390309_20.out
│   │   │   │   ├── slurm-12390309_21.out
│   │   │   │   ├── slurm-12390309_22.out
│   │   │   │   ├── slurm-12390309_23.out
│   │   │   │   ├── slurm-12390309_24.out
│   │   │   │   ├── slurm-12390309_25.out
│   │   │   │   ├── slurm-12390309_26.out
│   │   │   │   ├── slurm-12390309_27.out
│   │   │   │   ├── slurm-12390309_28.out
│   │   │   │   └── slurm-12390309_29.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── gpt-4o
│   │   │           ├── mr_extract_openai_array_0.json
│   │   │           ├── mr_extract_openai_array_1.json
│   │   │           ├── mr_extract_openai_array_2.json
│   │   │           ├── mr_extract_openai_array_3.json
│   │   │           ├── mr_extract_openai_array_4.json
│   │   │           ├── mr_extract_openai_array_5.json
│   │   │           ├── mr_extract_openai_array_6.json
│   │   │           ├── mr_extract_openai_array_7.json
│   │   │           ├── mr_extract_openai_array_8.json
│   │   │           ├── mr_extract_openai_array_9.json
│   │   │           ├── mr_extract_openai_array_10.json
│   │   │           ├── mr_extract_openai_array_11.json
│   │   │           ├── mr_extract_openai_array_12.json
│   │   │           ├── mr_extract_openai_array_13.json
│   │   │           ├── mr_extract_openai_array_14.json
│   │   │           ├── mr_extract_openai_array_15.json
│   │   │           ├── mr_extract_openai_array_16.json
│   │   │           ├── mr_extract_openai_array_17.json
│   │   │           ├── mr_extract_openai_array_18.json
│   │   │           ├── mr_extract_openai_array_19.json
│   │   │           ├── mr_extract_openai_array_20.json
│   │   │           ├── mr_extract_openai_array_21.json
│   │   │           ├── mr_extract_openai_array_22.json
│   │   │           ├── mr_extract_openai_array_23.json
│   │   │           ├── mr_extract_openai_array_24.json
│   │   │           ├── mr_extract_openai_array_25.json
│   │   │           ├── mr_extract_openai_array_26.json
│   │   │           ├── mr_extract_openai_array_27.json
│   │   │           ├── mr_extract_openai_array_28.json
│   │   │           └── mr_extract_openai_array_29.json
│   │   ├── bc4-12391182
│   │   │   ├── logs
│   │   │   │   ├── script-12391182.out
│   │   │   │   └── slurm-12391182_0.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── gpt-4o
│   │   │           └── mr_extract_openai_array_0.json
│   │   ├── bc4-12391186
│   │   │   ├── logs
│   │   │   │   ├── script-12391186.out
│   │   │   │   ├── slurm-12391186_0.out
│   │   │   │   ├── slurm-12391186_1.out
│   │   │   │   ├── slurm-12391186_2.out
│   │   │   │   ├── slurm-12391186_3.out
│   │   │   │   ├── slurm-12391186_4.out
│   │   │   │   ├── slurm-12391186_5.out
│   │   │   │   ├── slurm-12391186_6.out
│   │   │   │   ├── slurm-12391186_7.out
│   │   │   │   ├── slurm-12391186_8.out
│   │   │   │   ├── slurm-12391186_9.out
│   │   │   │   ├── slurm-12391186_10.out
│   │   │   │   ├── slurm-12391186_11.out
│   │   │   │   ├── slurm-12391186_12.out
│   │   │   │   ├── slurm-12391186_13.out
│   │   │   │   ├── slurm-12391186_14.out
│   │   │   │   ├── slurm-12391186_15.out
│   │   │   │   ├── slurm-12391186_16.out
│   │   │   │   ├── slurm-12391186_17.out
│   │   │   │   ├── slurm-12391186_18.out
│   │   │   │   ├── slurm-12391186_19.out
│   │   │   │   ├── slurm-12391186_20.out
│   │   │   │   ├── slurm-12391186_21.out
│   │   │   │   ├── slurm-12391186_22.out
│   │   │   │   ├── slurm-12391186_23.out
│   │   │   │   ├── slurm-12391186_24.out
│   │   │   │   ├── slurm-12391186_25.out
│   │   │   │   ├── slurm-12391186_26.out
│   │   │   │   ├── slurm-12391186_27.out
│   │   │   │   ├── slurm-12391186_28.out
│   │   │   │   └── slurm-12391186_29.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── gpt-4o
│   │   │           ├── mr_extract_openai_array_0.json
│   │   │           ├── mr_extract_openai_array_1.json
│   │   │           ├── mr_extract_openai_array_2.json
│   │   │           ├── mr_extract_openai_array_3.json
│   │   │           ├── mr_extract_openai_array_4.json
│   │   │           ├── mr_extract_openai_array_5.json
│   │   │           ├── mr_extract_openai_array_6.json
│   │   │           ├── mr_extract_openai_array_7.json
│   │   │           ├── mr_extract_openai_array_8.json
│   │   │           ├── mr_extract_openai_array_9.json
│   │   │           ├── mr_extract_openai_array_10.json
│   │   │           ├── mr_extract_openai_array_11.json
│   │   │           ├── mr_extract_openai_array_12.json
│   │   │           ├── mr_extract_openai_array_13.json
│   │   │           ├── mr_extract_openai_array_14.json
│   │   │           ├── mr_extract_openai_array_15.json
│   │   │           ├── mr_extract_openai_array_16.json
│   │   │           ├── mr_extract_openai_array_17.json
│   │   │           ├── mr_extract_openai_array_18.json
│   │   │           ├── mr_extract_openai_array_19.json
│   │   │           ├── mr_extract_openai_array_20.json
│   │   │           ├── mr_extract_openai_array_21.json
│   │   │           ├── mr_extract_openai_array_22.json
│   │   │           ├── mr_extract_openai_array_23.json
│   │   │           ├── mr_extract_openai_array_24.json
│   │   │           ├── mr_extract_openai_array_25.json
│   │   │           ├── mr_extract_openai_array_26.json
│   │   │           ├── mr_extract_openai_array_27.json
│   │   │           ├── mr_extract_openai_array_28.json
│   │   │           └── mr_extract_openai_array_29.json
│   │   ├── bc4-12398167
│   │   │   ├── logs
│   │   │   │   ├── script-12398167.out
│   │   │   │   ├── slurm-12398167_0.out
│   │   │   │   ├── slurm-12398167_1.out
│   │   │   │   ├── slurm-12398167_2.out
│   │   │   │   ├── slurm-12398167_3.out
│   │   │   │   ├── slurm-12398167_4.out
│   │   │   │   ├── slurm-12398167_5.out
│   │   │   │   ├── slurm-12398167_6.out
│   │   │   │   ├── slurm-12398167_7.out
│   │   │   │   ├── slurm-12398167_8.out
│   │   │   │   ├── slurm-12398167_9.out
│   │   │   │   ├── slurm-12398167_10.out
│   │   │   │   ├── slurm-12398167_11.out
│   │   │   │   ├── slurm-12398167_12.out
│   │   │   │   ├── slurm-12398167_13.out
│   │   │   │   ├── slurm-12398167_14.out
│   │   │   │   ├── slurm-12398167_15.out
│   │   │   │   ├── slurm-12398167_16.out
│   │   │   │   ├── slurm-12398167_17.out
│   │   │   │   ├── slurm-12398167_18.out
│   │   │   │   ├── slurm-12398167_19.out
│   │   │   │   ├── slurm-12398167_20.out
│   │   │   │   ├── slurm-12398167_21.out
│   │   │   │   ├── slurm-12398167_22.out
│   │   │   │   ├── slurm-12398167_23.out
│   │   │   │   ├── slurm-12398167_24.out
│   │   │   │   ├── slurm-12398167_25.out
│   │   │   │   ├── slurm-12398167_26.out
│   │   │   │   ├── slurm-12398167_27.out
│   │   │   │   ├── slurm-12398167_28.out
│   │   │   │   └── slurm-12398167_29.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── o4-mini
│   │   │           ├── mr_extract_openai_array_0.json
│   │   │           ├── mr_extract_openai_array_1.json
│   │   │           ├── mr_extract_openai_array_2.json
│   │   │           ├── mr_extract_openai_array_3.json
│   │   │           ├── mr_extract_openai_array_4.json
│   │   │           ├── mr_extract_openai_array_5.json
│   │   │           ├── mr_extract_openai_array_6.json
│   │   │           ├── mr_extract_openai_array_7.json
│   │   │           ├── mr_extract_openai_array_8.json
│   │   │           ├── mr_extract_openai_array_9.json
│   │   │           ├── mr_extract_openai_array_10.json
│   │   │           ├── mr_extract_openai_array_11.json
│   │   │           ├── mr_extract_openai_array_12.json
│   │   │           ├── mr_extract_openai_array_13.json
│   │   │           ├── mr_extract_openai_array_14.json
│   │   │           ├── mr_extract_openai_array_15.json
│   │   │           ├── mr_extract_openai_array_16.json
│   │   │           ├── mr_extract_openai_array_17.json
│   │   │           ├── mr_extract_openai_array_18.json
│   │   │           ├── mr_extract_openai_array_19.json
│   │   │           ├── mr_extract_openai_array_20.json
│   │   │           ├── mr_extract_openai_array_21.json
│   │   │           ├── mr_extract_openai_array_22.json
│   │   │           ├── mr_extract_openai_array_23.json
│   │   │           ├── mr_extract_openai_array_24.json
│   │   │           ├── mr_extract_openai_array_25.json
│   │   │           ├── mr_extract_openai_array_26.json
│   │   │           ├── mr_extract_openai_array_27.json
│   │   │           ├── mr_extract_openai_array_28.json
│   │   │           └── mr_extract_openai_array_29.json
│   │   ├── bc4-12411151
│   │   │   ├── logs
│   │   │   │   ├── script-12411151.out
│   │   │   │   └── slurm-12411151_0.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── gpt-4-1
│   │   │           └── mr_extract_openai_array_0.json
│   │   ├── bc4-12414116
│   │   │   ├── logs
│   │   │   │   ├── script-12414116.out
│   │   │   │   ├── slurm-12414116_0.out
│   │   │   │   ├── slurm-12414116_1.out
│   │   │   │   ├── slurm-12414116_2.out
│   │   │   │   ├── slurm-12414116_3.out
│   │   │   │   ├── slurm-12414116_4.out
│   │   │   │   ├── slurm-12414116_5.out
│   │   │   │   ├── slurm-12414116_6.out
│   │   │   │   ├── slurm-12414116_7.out
│   │   │   │   ├── slurm-12414116_8.out
│   │   │   │   ├── slurm-12414116_9.out
│   │   │   │   ├── slurm-12414116_10.out
│   │   │   │   ├── slurm-12414116_11.out
│   │   │   │   ├── slurm-12414116_12.out
│   │   │   │   ├── slurm-12414116_13.out
│   │   │   │   ├── slurm-12414116_14.out
│   │   │   │   ├── slurm-12414116_15.out
│   │   │   │   ├── slurm-12414116_16.out
│   │   │   │   ├── slurm-12414116_17.out
│   │   │   │   ├── slurm-12414116_18.out
│   │   │   │   ├── slurm-12414116_19.out
│   │   │   │   ├── slurm-12414116_20.out
│   │   │   │   ├── slurm-12414116_21.out
│   │   │   │   ├── slurm-12414116_22.out
│   │   │   │   ├── slurm-12414116_23.out
│   │   │   │   ├── slurm-12414116_24.out
│   │   │   │   ├── slurm-12414116_25.out
│   │   │   │   ├── slurm-12414116_26.out
│   │   │   │   ├── slurm-12414116_27.out
│   │   │   │   ├── slurm-12414116_28.out
│   │   │   │   └── slurm-12414116_29.out
│   │   │   ├── README
│   │   │   └── results
│   │   │       └── gpt-4-1
│   │   │           ├── mr_extract_openai_array_0.json
│   │   │           ├── mr_extract_openai_array_1.json
│   │   │           ├── mr_extract_openai_array_2.json
│   │   │           ├── mr_extract_openai_array_3.json
│   │   │           ├── mr_extract_openai_array_4.json
│   │   │           ├── mr_extract_openai_array_5.json
│   │   │           ├── mr_extract_openai_array_6.json
│   │   │           ├── mr_extract_openai_array_7.json
│   │   │           ├── mr_extract_openai_array_8.json
│   │   │           ├── mr_extract_openai_array_9.json
│   │   │           ├── mr_extract_openai_array_10.json
│   │   │           ├── mr_extract_openai_array_11.json
│   │   │           ├── mr_extract_openai_array_12.json
│   │   │           ├── mr_extract_openai_array_13.json
│   │   │           ├── mr_extract_openai_array_14.json
│   │   │           ├── mr_extract_openai_array_15.json
│   │   │           ├── mr_extract_openai_array_16.json
│   │   │           ├── mr_extract_openai_array_17.json
│   │   │           ├── mr_extract_openai_array_18.json
│   │   │           ├── mr_extract_openai_array_19.json
│   │   │           ├── mr_extract_openai_array_20.json
│   │   │           ├── mr_extract_openai_array_21.json
│   │   │           ├── mr_extract_openai_array_22.json
│   │   │           ├── mr_extract_openai_array_23.json
│   │   │           ├── mr_extract_openai_array_24.json
│   │   │           ├── mr_extract_openai_array_25.json
│   │   │           ├── mr_extract_openai_array_26.json
│   │   │           ├── mr_extract_openai_array_27.json
│   │   │           ├── mr_extract_openai_array_28.json
│   │   │           └── mr_extract_openai_array_29.json
│   │   ├── isb-ai-111542
│   │   │   ├── logs
│   │   │   │   ├── script-111542.out
│   │   │   │   └── slurm-111542_0.out
│   │   │   └── results
│   │   │       └── mr_extract_llama3_sample_array_0.json
│   │   ├── isb-ai-111544
│   │   │   ├── logs
│   │   │   │   ├── script-111545.out
│   │   │   │   ├── slurm-111544_0.out
│   │   │   │   ├── slurm-111544_1.out
│   │   │   │   ├── slurm-111544_2.out
│   │   │   │   ├── slurm-111544_3.out
│   │   │   │   ├── slurm-111544_4.out
│   │   │   │   ├── slurm-111544_5.out
│   │   │   │   ├── slurm-111544_6.out
│   │   │   │   ├── slurm-111544_7.out
│   │   │   │   ├── slurm-111544_8.out
│   │   │   │   ├── slurm-111544_9.out
│   │   │   │   ├── slurm-111544_10.out
│   │   │   │   ├── slurm-111544_11.out
│   │   │   │   ├── slurm-111544_12.out
│   │   │   │   ├── slurm-111544_13.out
│   │   │   │   ├── slurm-111544_14.out
│   │   │   │   ├── slurm-111544_15.out
│   │   │   │   ├── slurm-111544_16.out
│   │   │   │   ├── slurm-111544_17.out
│   │   │   │   ├── slurm-111544_18.out
│   │   │   │   ├── slurm-111544_19.out
│   │   │   │   ├── slurm-111544_20.out
│   │   │   │   ├── slurm-111544_21.out
│   │   │   │   ├── slurm-111544_22.out
│   │   │   │   ├── slurm-111544_23.out
│   │   │   │   ├── slurm-111544_24.out
│   │   │   │   ├── slurm-111544_25.out
│   │   │   │   ├── slurm-111544_26.out
│   │   │   │   ├── slurm-111544_27.out
│   │   │   │   ├── slurm-111544_28.out
│   │   │   │   ├── slurm-111544_29.out
│   │   │   │   ├── slurm-111544_30.out
│   │   │   │   ├── slurm-111544_31.out
│   │   │   │   ├── slurm-111544_32.out
│   │   │   │   ├── slurm-111544_33.out
│   │   │   │   ├── slurm-111544_34.out
│   │   │   │   ├── slurm-111544_35.out
│   │   │   │   ├── slurm-111544_36.out
│   │   │   │   ├── slurm-111544_37.out
│   │   │   │   ├── slurm-111544_38.out
│   │   │   │   ├── slurm-111544_39.out
│   │   │   │   ├── slurm-111544_40.out
│   │   │   │   ├── slurm-111544_41.out
│   │   │   │   ├── slurm-111544_42.out
│   │   │   │   ├── slurm-111544_43.out
│   │   │   │   ├── slurm-111544_44.out
│   │   │   │   ├── slurm-111544_45.out
│   │   │   │   ├── slurm-111544_46.out
│   │   │   │   ├── slurm-111544_47.out
│   │   │   │   ├── slurm-111544_48.out
│   │   │   │   ├── slurm-111544_49.out
│   │   │   │   ├── slurm-111544_50.out
│   │   │   │   ├── slurm-111544_51.out
│   │   │   │   ├── slurm-111544_52.out
│   │   │   │   ├── slurm-111544_53.out
│   │   │   │   ├── slurm-111544_54.out
│   │   │   │   ├── slurm-111544_55.out
│   │   │   │   ├── slurm-111544_56.out
│   │   │   │   ├── slurm-111544_57.out
│   │   │   │   ├── slurm-111544_58.out
│   │   │   │   ├── slurm-111544_59.out
│   │   │   │   ├── slurm-111544_60.out
│   │   │   │   ├── slurm-111544_61.out
│   │   │   │   ├── slurm-111544_62.out
│   │   │   │   ├── slurm-111544_63.out
│   │   │   │   ├── slurm-111544_64.out
│   │   │   │   ├── slurm-111544_65.out
│   │   │   │   ├── slurm-111544_66.out
│   │   │   │   ├── slurm-111544_67.out
│   │   │   │   ├── slurm-111544_68.out
│   │   │   │   └── slurm-111544_69.out
│   │   │   └── results
│   │   │       ├── mr_extract_llama3_sample_array_0.json
│   │   │       ├── mr_extract_llama3_sample_array_1.json
│   │   │       ├── mr_extract_llama3_sample_array_2.json
│   │   │       ├── mr_extract_llama3_sample_array_3.json
│   │   │       ├── mr_extract_llama3_sample_array_4.json
│   │   │       ├── mr_extract_llama3_sample_array_5.json
│   │   │       ├── mr_extract_llama3_sample_array_6.json
│   │   │       ├── mr_extract_llama3_sample_array_7.json
│   │   │       ├── mr_extract_llama3_sample_array_8.json
│   │   │       ├── mr_extract_llama3_sample_array_9.json
│   │   │       ├── mr_extract_llama3_sample_array_10.json
│   │   │       ├── mr_extract_llama3_sample_array_11.json
│   │   │       ├── mr_extract_llama3_sample_array_12.json
│   │   │       ├── mr_extract_llama3_sample_array_13.json
│   │   │       ├── mr_extract_llama3_sample_array_14.json
│   │   │       ├── mr_extract_llama3_sample_array_15.json
│   │   │       ├── mr_extract_llama3_sample_array_16.json
│   │   │       ├── mr_extract_llama3_sample_array_17.json
│   │   │       ├── mr_extract_llama3_sample_array_18.json
│   │   │       ├── mr_extract_llama3_sample_array_19.json
│   │   │       ├── mr_extract_llama3_sample_array_20.json
│   │   │       ├── mr_extract_llama3_sample_array_21.json
│   │   │       ├── mr_extract_llama3_sample_array_22.json
│   │   │       ├── mr_extract_llama3_sample_array_23.json
│   │   │       ├── mr_extract_llama3_sample_array_24.json
│   │   │       ├── mr_extract_llama3_sample_array_25.json
│   │   │       ├── mr_extract_llama3_sample_array_26.json
│   │   │       ├── mr_extract_llama3_sample_array_27.json
│   │   │       ├── mr_extract_llama3_sample_array_28.json
│   │   │       ├── mr_extract_llama3_sample_array_29.json
│   │   │       ├── mr_extract_llama3_sample_array_30.json
│   │   │       ├── mr_extract_llama3_sample_array_31.json
│   │   │       ├── mr_extract_llama3_sample_array_32.json
│   │   │       ├── mr_extract_llama3_sample_array_33.json
│   │   │       ├── mr_extract_llama3_sample_array_34.json
│   │   │       ├── mr_extract_llama3_sample_array_35.json
│   │   │       ├── mr_extract_llama3_sample_array_36.json
│   │   │       ├── mr_extract_llama3_sample_array_37.json
│   │   │       ├── mr_extract_llama3_sample_array_38.json
│   │   │       ├── mr_extract_llama3_sample_array_39.json
│   │   │       ├── mr_extract_llama3_sample_array_40.json
│   │   │       ├── mr_extract_llama3_sample_array_41.json
│   │   │       ├── mr_extract_llama3_sample_array_42.json
│   │   │       ├── mr_extract_llama3_sample_array_43.json
│   │   │       ├── mr_extract_llama3_sample_array_44.json
│   │   │       ├── mr_extract_llama3_sample_array_45.json
│   │   │       ├── mr_extract_llama3_sample_array_46.json
│   │   │       ├── mr_extract_llama3_sample_array_47.json
│   │   │       ├── mr_extract_llama3_sample_array_48.json
│   │   │       ├── mr_extract_llama3_sample_array_49.json
│   │   │       ├── mr_extract_llama3_sample_array_50.json
│   │   │       ├── mr_extract_llama3_sample_array_51.json
│   │   │       ├── mr_extract_llama3_sample_array_52.json
│   │   │       ├── mr_extract_llama3_sample_array_53.json
│   │   │       ├── mr_extract_llama3_sample_array_54.json
│   │   │       ├── mr_extract_llama3_sample_array_55.json
│   │   │       ├── mr_extract_llama3_sample_array_56.json
│   │   │       ├── mr_extract_llama3_sample_array_57.json
│   │   │       ├── mr_extract_llama3_sample_array_58.json
│   │   │       ├── mr_extract_llama3_sample_array_59.json
│   │   │       ├── mr_extract_llama3_sample_array_60.json
│   │   │       ├── mr_extract_llama3_sample_array_61.json
│   │   │       ├── mr_extract_llama3_sample_array_62.json
│   │   │       ├── mr_extract_llama3_sample_array_63.json
│   │   │       ├── mr_extract_llama3_sample_array_64.json
│   │   │       ├── mr_extract_llama3_sample_array_65.json
│   │   │       ├── mr_extract_llama3_sample_array_66.json
│   │   │       ├── mr_extract_llama3_sample_array_67.json
│   │   │       ├── mr_extract_llama3_sample_array_68.json
│   │   │       └── mr_extract_llama3_sample_array_69.json
│   │   ├── isb-ai-111997
│   │   │   ├── logs
│   │   │   │   ├── script-111998.out
│   │   │   │   ├── slurm-111997_0.out
│   │   │   │   ├── slurm-111997_1.out
│   │   │   │   ├── slurm-111997_2.out
│   │   │   │   ├── slurm-111997_3.out
│   │   │   │   ├── slurm-111997_4.out
│   │   │   │   ├── slurm-111997_5.out
│   │   │   │   ├── slurm-111997_6.out
│   │   │   │   ├── slurm-111997_7.out
│   │   │   │   ├── slurm-111997_8.out
│   │   │   │   ├── slurm-111997_9.out
│   │   │   │   ├── slurm-111997_10.out
│   │   │   │   ├── slurm-111997_11.out
│   │   │   │   ├── slurm-111997_12.out
│   │   │   │   ├── slurm-111997_13.out
│   │   │   │   ├── slurm-111997_14.out
│   │   │   │   ├── slurm-111997_15.out
│   │   │   │   ├── slurm-111997_16.out
│   │   │   │   ├── slurm-111997_17.out
│   │   │   │   ├── slurm-111997_18.out
│   │   │   │   ├── slurm-111997_19.out
│   │   │   │   ├── slurm-111997_20.out
│   │   │   │   ├── slurm-111997_21.out
│   │   │   │   ├── slurm-111997_22.out
│   │   │   │   ├── slurm-111997_23.out
│   │   │   │   ├── slurm-111997_24.out
│   │   │   │   ├── slurm-111997_25.out
│   │   │   │   ├── slurm-111997_26.out
│   │   │   │   ├── slurm-111997_27.out
│   │   │   │   ├── slurm-111997_28.out
│   │   │   │   ├── slurm-111997_29.out
│   │   │   │   ├── slurm-111997_30.out
│   │   │   │   ├── slurm-111997_31.out
│   │   │   │   ├── slurm-111997_32.out
│   │   │   │   ├── slurm-111997_33.out
│   │   │   │   ├── slurm-111997_34.out
│   │   │   │   ├── slurm-111997_35.out
│   │   │   │   ├── slurm-111997_36.out
│   │   │   │   ├── slurm-111997_37.out
│   │   │   │   ├── slurm-111997_38.out
│   │   │   │   ├── slurm-111997_39.out
│   │   │   │   ├── slurm-111997_40.out
│   │   │   │   ├── slurm-111997_41.out
│   │   │   │   ├── slurm-111997_42.out
│   │   │   │   ├── slurm-111997_43.out
│   │   │   │   ├── slurm-111997_44.out
│   │   │   │   ├── slurm-111997_45.out
│   │   │   │   ├── slurm-111997_46.out
│   │   │   │   ├── slurm-111997_47.out
│   │   │   │   ├── slurm-111997_48.out
│   │   │   │   ├── slurm-111997_49.out
│   │   │   │   ├── slurm-111997_50.out
│   │   │   │   ├── slurm-111997_51.out
│   │   │   │   ├── slurm-111997_52.out
│   │   │   │   ├── slurm-111997_53.out
│   │   │   │   ├── slurm-111997_54.out
│   │   │   │   ├── slurm-111997_55.out
│   │   │   │   ├── slurm-111997_56.out
│   │   │   │   ├── slurm-111997_57.out
│   │   │   │   ├── slurm-111997_58.out
│   │   │   │   ├── slurm-111997_59.out
│   │   │   │   ├── slurm-111997_60.out
│   │   │   │   ├── slurm-111997_61.out
│   │   │   │   ├── slurm-111997_62.out
│   │   │   │   ├── slurm-111997_63.out
│   │   │   │   ├── slurm-111997_64.out
│   │   │   │   ├── slurm-111997_65.out
│   │   │   │   ├── slurm-111997_66.out
│   │   │   │   ├── slurm-111997_67.out
│   │   │   │   ├── slurm-111997_68.out
│   │   │   │   └── slurm-111997_69.out
│   │   │   ├── README.md
│   │   │   └── results
│   │   │       ├── mr_extract_llama3_sample_array_0.json
│   │   │       ├── mr_extract_llama3_sample_array_1.json
│   │   │       ├── mr_extract_llama3_sample_array_2.json
│   │   │       ├── mr_extract_llama3_sample_array_3.json
│   │   │       ├── mr_extract_llama3_sample_array_4.json
│   │   │       ├── mr_extract_llama3_sample_array_5.json
│   │   │       ├── mr_extract_llama3_sample_array_6.json
│   │   │       ├── mr_extract_llama3_sample_array_7.json
│   │   │       ├── mr_extract_llama3_sample_array_8.json
│   │   │       ├── mr_extract_llama3_sample_array_9.json
│   │   │       ├── mr_extract_llama3_sample_array_10.json
│   │   │       ├── mr_extract_llama3_sample_array_11.json
│   │   │       ├── mr_extract_llama3_sample_array_12.json
│   │   │       ├── mr_extract_llama3_sample_array_13.json
│   │   │       ├── mr_extract_llama3_sample_array_14.json
│   │   │       ├── mr_extract_llama3_sample_array_15.json
│   │   │       ├── mr_extract_llama3_sample_array_16.json
│   │   │       ├── mr_extract_llama3_sample_array_17.json
│   │   │       ├── mr_extract_llama3_sample_array_18.json
│   │   │       ├── mr_extract_llama3_sample_array_19.json
│   │   │       ├── mr_extract_llama3_sample_array_20.json
│   │   │       ├── mr_extract_llama3_sample_array_21.json
│   │   │       ├── mr_extract_llama3_sample_array_22.json
│   │   │       ├── mr_extract_llama3_sample_array_23.json
│   │   │       ├── mr_extract_llama3_sample_array_24.json
│   │   │       ├── mr_extract_llama3_sample_array_25.json
│   │   │       ├── mr_extract_llama3_sample_array_26.json
│   │   │       ├── mr_extract_llama3_sample_array_27.json
│   │   │       ├── mr_extract_llama3_sample_array_28.json
│   │   │       ├── mr_extract_llama3_sample_array_29.json
│   │   │       ├── mr_extract_llama3_sample_array_30.json
│   │   │       ├── mr_extract_llama3_sample_array_31.json
│   │   │       ├── mr_extract_llama3_sample_array_32.json
│   │   │       ├── mr_extract_llama3_sample_array_33.json
│   │   │       ├── mr_extract_llama3_sample_array_34.json
│   │   │       ├── mr_extract_llama3_sample_array_35.json
│   │   │       ├── mr_extract_llama3_sample_array_36.json
│   │   │       ├── mr_extract_llama3_sample_array_37.json
│   │   │       ├── mr_extract_llama3_sample_array_38.json
│   │   │       ├── mr_extract_llama3_sample_array_39.json
│   │   │       ├── mr_extract_llama3_sample_array_40.json
│   │   │       ├── mr_extract_llama3_sample_array_41.json
│   │   │       ├── mr_extract_llama3_sample_array_42.json
│   │   │       ├── mr_extract_llama3_sample_array_43.json
│   │   │       ├── mr_extract_llama3_sample_array_44.json
│   │   │       ├── mr_extract_llama3_sample_array_45.json
│   │   │       ├── mr_extract_llama3_sample_array_46.json
│   │   │       ├── mr_extract_llama3_sample_array_47.json
│   │   │       ├── mr_extract_llama3_sample_array_48.json
│   │   │       ├── mr_extract_llama3_sample_array_49.json
│   │   │       ├── mr_extract_llama3_sample_array_50.json
│   │   │       ├── mr_extract_llama3_sample_array_51.json
│   │   │       ├── mr_extract_llama3_sample_array_52.json
│   │   │       ├── mr_extract_llama3_sample_array_53.json
│   │   │       ├── mr_extract_llama3_sample_array_54.json
│   │   │       ├── mr_extract_llama3_sample_array_55.json
│   │   │       ├── mr_extract_llama3_sample_array_56.json
│   │   │       ├── mr_extract_llama3_sample_array_57.json
│   │   │       ├── mr_extract_llama3_sample_array_58.json
│   │   │       ├── mr_extract_llama3_sample_array_59.json
│   │   │       ├── mr_extract_llama3_sample_array_60.json
│   │   │       ├── mr_extract_llama3_sample_array_61.json
│   │   │       ├── mr_extract_llama3_sample_array_62.json
│   │   │       ├── mr_extract_llama3_sample_array_63.json
│   │   │       ├── mr_extract_llama3_sample_array_64.json
│   │   │       ├── mr_extract_llama3_sample_array_65.json
│   │   │       ├── mr_extract_llama3_sample_array_66.json
│   │   │       ├── mr_extract_llama3_sample_array_67.json
│   │   │       ├── mr_extract_llama3_sample_array_68.json
│   │   │       └── mr_extract_llama3_sample_array_69.json
│   │   ├── isb-ai-116732
│   │   │   ├── logs
│   │   │   │   ├── script-116733.out
│   │   │   │   ├── slurm-116732_0.out
│   │   │   │   ├── slurm-116732_1.out
│   │   │   │   ├── slurm-116732_2.out
│   │   │   │   ├── slurm-116732_3.out
│   │   │   │   ├── slurm-116732_4.out
│   │   │   │   ├── slurm-116732_5.out
│   │   │   │   ├── slurm-116732_6.out
│   │   │   │   ├── slurm-116732_7.out
│   │   │   │   ├── slurm-116732_8.out
│   │   │   │   ├── slurm-116732_9.out
│   │   │   │   ├── slurm-116732_10.out
│   │   │   │   ├── slurm-116732_11.out
│   │   │   │   ├── slurm-116732_12.out
│   │   │   │   ├── slurm-116732_13.out
│   │   │   │   ├── slurm-116732_14.out
│   │   │   │   ├── slurm-116732_15.out
│   │   │   │   ├── slurm-116732_16.out
│   │   │   │   ├── slurm-116732_17.out
│   │   │   │   ├── slurm-116732_18.out
│   │   │   │   ├── slurm-116732_19.out
│   │   │   │   ├── slurm-116732_20.out
│   │   │   │   ├── slurm-116732_21.out
│   │   │   │   ├── slurm-116732_22.out
│   │   │   │   ├── slurm-116732_23.out
│   │   │   │   ├── slurm-116732_24.out
│   │   │   │   ├── slurm-116732_25.out
│   │   │   │   ├── slurm-116732_26.out
│   │   │   │   ├── slurm-116732_27.out
│   │   │   │   ├── slurm-116732_28.out
│   │   │   │   ├── slurm-116732_29.out
│   │   │   │   ├── slurm-116732_30.out
│   │   │   │   ├── slurm-116732_31.out
│   │   │   │   ├── slurm-116732_32.out
│   │   │   │   ├── slurm-116732_33.out
│   │   │   │   ├── slurm-116732_34.out
│   │   │   │   ├── slurm-116732_35.out
│   │   │   │   ├── slurm-116732_36.out
│   │   │   │   ├── slurm-116732_37.out
│   │   │   │   ├── slurm-116732_38.out
│   │   │   │   ├── slurm-116732_39.out
│   │   │   │   ├── slurm-116732_40.out
│   │   │   │   ├── slurm-116732_41.out
│   │   │   │   ├── slurm-116732_42.out
│   │   │   │   ├── slurm-116732_43.out
│   │   │   │   ├── slurm-116732_44.out
│   │   │   │   ├── slurm-116732_45.out
│   │   │   │   ├── slurm-116732_46.out
│   │   │   │   ├── slurm-116732_47.out
│   │   │   │   ├── slurm-116732_48.out
│   │   │   │   ├── slurm-116732_49.out
│   │   │   │   ├── slurm-116732_50.out
│   │   │   │   ├── slurm-116732_51.out
│   │   │   │   ├── slurm-116732_52.out
│   │   │   │   ├── slurm-116732_53.out
│   │   │   │   ├── slurm-116732_54.out
│   │   │   │   ├── slurm-116732_55.out
│   │   │   │   ├── slurm-116732_56.out
│   │   │   │   ├── slurm-116732_57.out
│   │   │   │   ├── slurm-116732_58.out
│   │   │   │   ├── slurm-116732_59.out
│   │   │   │   ├── slurm-116732_60.out
│   │   │   │   ├── slurm-116732_61.out
│   │   │   │   ├── slurm-116732_62.out
│   │   │   │   ├── slurm-116732_63.out
│   │   │   │   ├── slurm-116732_64.out
│   │   │   │   ├── slurm-116732_65.out
│   │   │   │   ├── slurm-116732_66.out
│   │   │   │   ├── slurm-116732_67.out
│   │   │   │   ├── slurm-116732_68.out
│   │   │   │   └── slurm-116732_69.out
│   │   │   └── results
│   │   │       ├── mr_extract_llama3_sample_array_0.json
│   │   │       ├── mr_extract_llama3_sample_array_1.json
│   │   │       ├── mr_extract_llama3_sample_array_2.json
│   │   │       ├── mr_extract_llama3_sample_array_3.json
│   │   │       ├── mr_extract_llama3_sample_array_4.json
│   │   │       ├── mr_extract_llama3_sample_array_5.json
│   │   │       ├── mr_extract_llama3_sample_array_6.json
│   │   │       ├── mr_extract_llama3_sample_array_7.json
│   │   │       ├── mr_extract_llama3_sample_array_8.json
│   │   │       ├── mr_extract_llama3_sample_array_9.json
│   │   │       ├── mr_extract_llama3_sample_array_10.json
│   │   │       ├── mr_extract_llama3_sample_array_11.json
│   │   │       ├── mr_extract_llama3_sample_array_12.json
│   │   │       ├── mr_extract_llama3_sample_array_13.json
│   │   │       ├── mr_extract_llama3_sample_array_14.json
│   │   │       ├── mr_extract_llama3_sample_array_15.json
│   │   │       ├── mr_extract_llama3_sample_array_16.json
│   │   │       ├── mr_extract_llama3_sample_array_17.json
│   │   │       ├── mr_extract_llama3_sample_array_18.json
│   │   │       ├── mr_extract_llama3_sample_array_19.json
│   │   │       ├── mr_extract_llama3_sample_array_20.json
│   │   │       ├── mr_extract_llama3_sample_array_21.json
│   │   │       ├── mr_extract_llama3_sample_array_22.json
│   │   │       ├── mr_extract_llama3_sample_array_23.json
│   │   │       ├── mr_extract_llama3_sample_array_24.json
│   │   │       ├── mr_extract_llama3_sample_array_25.json
│   │   │       ├── mr_extract_llama3_sample_array_26.json
│   │   │       ├── mr_extract_llama3_sample_array_27.json
│   │   │       ├── mr_extract_llama3_sample_array_28.json
│   │   │       ├── mr_extract_llama3_sample_array_29.json
│   │   │       ├── mr_extract_llama3_sample_array_30.json
│   │   │       ├── mr_extract_llama3_sample_array_31.json
│   │   │       ├── mr_extract_llama3_sample_array_32.json
│   │   │       ├── mr_extract_llama3_sample_array_33.json
│   │   │       ├── mr_extract_llama3_sample_array_34.json
│   │   │       ├── mr_extract_llama3_sample_array_35.json
│   │   │       ├── mr_extract_llama3_sample_array_36.json
│   │   │       ├── mr_extract_llama3_sample_array_37.json
│   │   │       ├── mr_extract_llama3_sample_array_38.json
│   │   │       ├── mr_extract_llama3_sample_array_39.json
│   │   │       ├── mr_extract_llama3_sample_array_40.json
│   │   │       ├── mr_extract_llama3_sample_array_41.json
│   │   │       ├── mr_extract_llama3_sample_array_42.json
│   │   │       ├── mr_extract_llama3_sample_array_43.json
│   │   │       ├── mr_extract_llama3_sample_array_44.json
│   │   │       ├── mr_extract_llama3_sample_array_45.json
│   │   │       ├── mr_extract_llama3_sample_array_46.json
│   │   │       ├── mr_extract_llama3_sample_array_47.json
│   │   │       ├── mr_extract_llama3_sample_array_48.json
│   │   │       ├── mr_extract_llama3_sample_array_49.json
│   │   │       ├── mr_extract_llama3_sample_array_50.json
│   │   │       ├── mr_extract_llama3_sample_array_51.json
│   │   │       ├── mr_extract_llama3_sample_array_52.json
│   │   │       ├── mr_extract_llama3_sample_array_53.json
│   │   │       ├── mr_extract_llama3_sample_array_54.json
│   │   │       ├── mr_extract_llama3_sample_array_55.json
│   │   │       ├── mr_extract_llama3_sample_array_56.json
│   │   │       ├── mr_extract_llama3_sample_array_57.json
│   │   │       ├── mr_extract_llama3_sample_array_58.json
│   │   │       ├── mr_extract_llama3_sample_array_59.json
│   │   │       ├── mr_extract_llama3_sample_array_60.json
│   │   │       ├── mr_extract_llama3_sample_array_61.json
│   │   │       ├── mr_extract_llama3_sample_array_62.json
│   │   │       ├── mr_extract_llama3_sample_array_63.json
│   │   │       ├── mr_extract_llama3_sample_array_64.json
│   │   │       ├── mr_extract_llama3_sample_array_65.json
│   │   │       ├── mr_extract_llama3_sample_array_66.json
│   │   │       ├── mr_extract_llama3_sample_array_67.json
│   │   │       ├── mr_extract_llama3_sample_array_68.json
│   │   │       └── mr_extract_llama3_sample_array_69.json
│   │   ├── isb-ai-117256
│   │   │   ├── logs
│   │   │   │   ├── script-117257.out
│   │   │   │   ├── slurm-117256_0.out
│   │   │   │   ├── slurm-117256_1.out
│   │   │   │   ├── slurm-117256_2.out
│   │   │   │   ├── slurm-117256_3.out
│   │   │   │   ├── slurm-117256_4.out
│   │   │   │   ├── slurm-117256_5.out
│   │   │   │   ├── slurm-117256_6.out
│   │   │   │   ├── slurm-117256_7.out
│   │   │   │   ├── slurm-117256_8.out
│   │   │   │   ├── slurm-117256_9.out
│   │   │   │   ├── slurm-117256_10.out
│   │   │   │   ├── slurm-117256_11.out
│   │   │   │   ├── slurm-117256_12.out
│   │   │   │   ├── slurm-117256_13.out
│   │   │   │   ├── slurm-117256_14.out
│   │   │   │   ├── slurm-117256_15.out
│   │   │   │   ├── slurm-117256_16.out
│   │   │   │   ├── slurm-117256_17.out
│   │   │   │   ├── slurm-117256_18.out
│   │   │   │   ├── slurm-117256_19.out
│   │   │   │   ├── slurm-117256_20.out
│   │   │   │   ├── slurm-117256_21.out
│   │   │   │   ├── slurm-117256_22.out
│   │   │   │   ├── slurm-117256_23.out
│   │   │   │   ├── slurm-117256_24.out
│   │   │   │   ├── slurm-117256_25.out
│   │   │   │   ├── slurm-117256_26.out
│   │   │   │   ├── slurm-117256_27.out
│   │   │   │   ├── slurm-117256_28.out
│   │   │   │   ├── slurm-117256_29.out
│   │   │   │   ├── slurm-117256_30.out
│   │   │   │   ├── slurm-117256_31.out
│   │   │   │   ├── slurm-117256_32.out
│   │   │   │   ├── slurm-117256_33.out
│   │   │   │   ├── slurm-117256_34.out
│   │   │   │   ├── slurm-117256_35.out
│   │   │   │   ├── slurm-117256_36.out
│   │   │   │   ├── slurm-117256_37.out
│   │   │   │   ├── slurm-117256_38.out
│   │   │   │   ├── slurm-117256_39.out
│   │   │   │   ├── slurm-117256_40.out
│   │   │   │   ├── slurm-117256_41.out
│   │   │   │   ├── slurm-117256_42.out
│   │   │   │   ├── slurm-117256_43.out
│   │   │   │   ├── slurm-117256_44.out
│   │   │   │   ├── slurm-117256_45.out
│   │   │   │   ├── slurm-117256_46.out
│   │   │   │   ├── slurm-117256_47.out
│   │   │   │   ├── slurm-117256_48.out
│   │   │   │   ├── slurm-117256_49.out
│   │   │   │   ├── slurm-117256_50.out
│   │   │   │   ├── slurm-117256_51.out
│   │   │   │   ├── slurm-117256_52.out
│   │   │   │   ├── slurm-117256_53.out
│   │   │   │   ├── slurm-117256_54.out
│   │   │   │   ├── slurm-117256_55.out
│   │   │   │   ├── slurm-117256_56.out
│   │   │   │   ├── slurm-117256_57.out
│   │   │   │   ├── slurm-117256_58.out
│   │   │   │   ├── slurm-117256_59.out
│   │   │   │   ├── slurm-117256_60.out
│   │   │   │   ├── slurm-117256_61.out
│   │   │   │   ├── slurm-117256_62.out
│   │   │   │   ├── slurm-117256_63.out
│   │   │   │   ├── slurm-117256_64.out
│   │   │   │   ├── slurm-117256_65.out
│   │   │   │   ├── slurm-117256_66.out
│   │   │   │   ├── slurm-117256_67.out
│   │   │   │   ├── slurm-117256_68.out
│   │   │   │   └── slurm-117256_69.out
│   │   │   └── results
│   │   │       ├── mr_extract_array_0.json
│   │   │       ├── mr_extract_array_1.json
│   │   │       ├── mr_extract_array_2.json
│   │   │       ├── mr_extract_array_3.json
│   │   │       ├── mr_extract_array_4.json
│   │   │       ├── mr_extract_array_5.json
│   │   │       ├── mr_extract_array_6.json
│   │   │       ├── mr_extract_array_7.json
│   │   │       ├── mr_extract_array_8.json
│   │   │       ├── mr_extract_array_9.json
│   │   │       ├── mr_extract_array_10.json
│   │   │       ├── mr_extract_array_11.json
│   │   │       ├── mr_extract_array_12.json
│   │   │       ├── mr_extract_array_13.json
│   │   │       ├── mr_extract_array_14.json
│   │   │       ├── mr_extract_array_15.json
│   │   │       ├── mr_extract_array_16.json
│   │   │       ├── mr_extract_array_17.json
│   │   │       ├── mr_extract_array_18.json
│   │   │       ├── mr_extract_array_19.json
│   │   │       ├── mr_extract_array_20.json
│   │   │       ├── mr_extract_array_21.json
│   │   │       ├── mr_extract_array_22.json
│   │   │       ├── mr_extract_array_23.json
│   │   │       ├── mr_extract_array_24.json
│   │   │       ├── mr_extract_array_25.json
│   │   │       ├── mr_extract_array_26.json
│   │   │       ├── mr_extract_array_27.json
│   │   │       ├── mr_extract_array_28.json
│   │   │       ├── mr_extract_array_29.json
│   │   │       ├── mr_extract_array_30.json
│   │   │       ├── mr_extract_array_31.json
│   │   │       ├── mr_extract_array_32.json
│   │   │       ├── mr_extract_array_33.json
│   │   │       ├── mr_extract_array_34.json
│   │   │       ├── mr_extract_array_35.json
│   │   │       ├── mr_extract_array_36.json
│   │   │       ├── mr_extract_array_37.json
│   │   │       ├── mr_extract_array_38.json
│   │   │       ├── mr_extract_array_39.json
│   │   │       ├── mr_extract_array_40.json
│   │   │       ├── mr_extract_array_41.json
│   │   │       ├── mr_extract_array_42.json
│   │   │       ├── mr_extract_array_43.json
│   │   │       ├── mr_extract_array_44.json
│   │   │       ├── mr_extract_array_45.json
│   │   │       ├── mr_extract_array_46.json
│   │   │       ├── mr_extract_array_47.json
│   │   │       ├── mr_extract_array_48.json
│   │   │       ├── mr_extract_array_49.json
│   │   │       ├── mr_extract_array_50.json
│   │   │       ├── mr_extract_array_51.json
│   │   │       ├── mr_extract_array_52.json
│   │   │       ├── mr_extract_array_53.json
│   │   │       ├── mr_extract_array_54.json
│   │   │       ├── mr_extract_array_55.json
│   │   │       ├── mr_extract_array_56.json
│   │   │       ├── mr_extract_array_57.json
│   │   │       ├── mr_extract_array_58.json
│   │   │       ├── mr_extract_array_59.json
│   │   │       ├── mr_extract_array_60.json
│   │   │       ├── mr_extract_array_61.json
│   │   │       ├── mr_extract_array_62.json
│   │   │       ├── mr_extract_array_63.json
│   │   │       ├── mr_extract_array_64.json
│   │   │       ├── mr_extract_array_65.json
│   │   │       ├── mr_extract_array_66.json
│   │   │       ├── mr_extract_array_67.json
│   │   │       ├── mr_extract_array_68.json
│   │   │       └── mr_extract_array_69.json
│   │   ├── isb-ai-117535
│   │   │   ├── logs
│   │   │   │   ├── script-117535.out
│   │   │   │   ├── slurm-117535_0.out
│   │   │   │   ├── slurm-117535_1.out
│   │   │   │   ├── slurm-117535_2.out
│   │   │   │   ├── slurm-117535_3.out
│   │   │   │   ├── slurm-117535_4.out
│   │   │   │   ├── slurm-117535_5.out
│   │   │   │   ├── slurm-117535_6.out
│   │   │   │   ├── slurm-117535_7.out
│   │   │   │   ├── slurm-117535_8.out
│   │   │   │   ├── slurm-117535_9.out
│   │   │   │   ├── slurm-117535_10.out
│   │   │   │   ├── slurm-117535_11.out
│   │   │   │   ├── slurm-117535_12.out
│   │   │   │   ├── slurm-117535_13.out
│   │   │   │   ├── slurm-117535_14.out
│   │   │   │   ├── slurm-117535_15.out
│   │   │   │   ├── slurm-117535_16.out
│   │   │   │   ├── slurm-117535_17.out
│   │   │   │   ├── slurm-117535_18.out
│   │   │   │   ├── slurm-117535_19.out
│   │   │   │   ├── slurm-117535_20.out
│   │   │   │   ├── slurm-117535_21.out
│   │   │   │   ├── slurm-117535_22.out
│   │   │   │   ├── slurm-117535_23.out
│   │   │   │   ├── slurm-117535_24.out
│   │   │   │   ├── slurm-117535_25.out
│   │   │   │   ├── slurm-117535_26.out
│   │   │   │   ├── slurm-117535_27.out
│   │   │   │   ├── slurm-117535_28.out
│   │   │   │   ├── slurm-117535_29.out
│   │   │   │   ├── slurm-117535_30.out
│   │   │   │   ├── slurm-117535_31.out
│   │   │   │   ├── slurm-117535_32.out
│   │   │   │   ├── slurm-117535_33.out
│   │   │   │   ├── slurm-117535_34.out
│   │   │   │   ├── slurm-117535_35.out
│   │   │   │   ├── slurm-117535_36.out
│   │   │   │   ├── slurm-117535_37.out
│   │   │   │   ├── slurm-117535_38.out
│   │   │   │   ├── slurm-117535_39.out
│   │   │   │   ├── slurm-117535_40.out
│   │   │   │   ├── slurm-117535_41.out
│   │   │   │   ├── slurm-117535_42.out
│   │   │   │   ├── slurm-117535_43.out
│   │   │   │   ├── slurm-117535_44.out
│   │   │   │   ├── slurm-117535_45.out
│   │   │   │   ├── slurm-117535_46.out
│   │   │   │   ├── slurm-117535_47.out
│   │   │   │   ├── slurm-117535_48.out
│   │   │   │   ├── slurm-117535_49.out
│   │   │   │   ├── slurm-117535_50.out
│   │   │   │   ├── slurm-117535_51.out
│   │   │   │   ├── slurm-117535_52.out
│   │   │   │   ├── slurm-117535_53.out
│   │   │   │   ├── slurm-117535_54.out
│   │   │   │   ├── slurm-117535_55.out
│   │   │   │   ├── slurm-117535_56.out
│   │   │   │   ├── slurm-117535_57.out
│   │   │   │   ├── slurm-117535_58.out
│   │   │   │   ├── slurm-117535_59.out
│   │   │   │   ├── slurm-117535_60.out
│   │   │   │   ├── slurm-117535_61.out
│   │   │   │   ├── slurm-117535_62.out
│   │   │   │   ├── slurm-117535_63.out
│   │   │   │   ├── slurm-117535_64.out
│   │   │   │   ├── slurm-117535_65.out
│   │   │   │   ├── slurm-117535_66.out
│   │   │   │   ├── slurm-117535_67.out
│   │   │   │   ├── slurm-117535_68.out
│   │   │   │   └── slurm-117535_69.out
│   │   │   ├── README.md
│   │   │   └── results
│   │   │       ├── mr_extract_array_0.json
│   │   │       ├── mr_extract_array_1.json
│   │   │       ├── mr_extract_array_2.json
│   │   │       ├── mr_extract_array_3.json
│   │   │       ├── mr_extract_array_4.json
│   │   │       ├── mr_extract_array_5.json
│   │   │       ├── mr_extract_array_6.json
│   │   │       ├── mr_extract_array_7.json
│   │   │       ├── mr_extract_array_8.json
│   │   │       ├── mr_extract_array_9.json
│   │   │       ├── mr_extract_array_10.json
│   │   │       ├── mr_extract_array_11.json
│   │   │       ├── mr_extract_array_12.json
│   │   │       ├── mr_extract_array_13.json
│   │   │       ├── mr_extract_array_14.json
│   │   │       ├── mr_extract_array_15.json
│   │   │       ├── mr_extract_array_16.json
│   │   │       ├── mr_extract_array_17.json
│   │   │       ├── mr_extract_array_18.json
│   │   │       ├── mr_extract_array_19.json
│   │   │       ├── mr_extract_array_20.json
│   │   │       ├── mr_extract_array_21.json
│   │   │       ├── mr_extract_array_22.json
│   │   │       ├── mr_extract_array_23.json
│   │   │       ├── mr_extract_array_24.json
│   │   │       ├── mr_extract_array_25.json
│   │   │       ├── mr_extract_array_26.json
│   │   │       ├── mr_extract_array_27.json
│   │   │       ├── mr_extract_array_28.json
│   │   │       ├── mr_extract_array_29.json
│   │   │       ├── mr_extract_array_30.json
│   │   │       ├── mr_extract_array_31.json
│   │   │       ├── mr_extract_array_32.json
│   │   │       ├── mr_extract_array_33.json
│   │   │       ├── mr_extract_array_34.json
│   │   │       ├── mr_extract_array_35.json
│   │   │       ├── mr_extract_array_36.json
│   │   │       ├── mr_extract_array_37.json
│   │   │       ├── mr_extract_array_38.json
│   │   │       ├── mr_extract_array_39.json
│   │   │       ├── mr_extract_array_40.json
│   │   │       ├── mr_extract_array_41.json
│   │   │       ├── mr_extract_array_42.json
│   │   │       ├── mr_extract_array_43.json
│   │   │       ├── mr_extract_array_44.json
│   │   │       ├── mr_extract_array_45.json
│   │   │       ├── mr_extract_array_46.json
│   │   │       ├── mr_extract_array_47.json
│   │   │       ├── mr_extract_array_48.json
│   │   │       ├── mr_extract_array_49.json
│   │   │       ├── mr_extract_array_50.json
│   │   │       ├── mr_extract_array_51.json
│   │   │       ├── mr_extract_array_52.json
│   │   │       ├── mr_extract_array_53.json
│   │   │       ├── mr_extract_array_54.json
│   │   │       ├── mr_extract_array_55.json
│   │   │       ├── mr_extract_array_56.json
│   │   │       ├── mr_extract_array_57.json
│   │   │       ├── mr_extract_array_58.json
│   │   │       ├── mr_extract_array_59.json
│   │   │       ├── mr_extract_array_60.json
│   │   │       ├── mr_extract_array_61.json
│   │   │       ├── mr_extract_array_62.json
│   │   │       ├── mr_extract_array_63.json
│   │   │       ├── mr_extract_array_64.json
│   │   │       ├── mr_extract_array_65.json
│   │   │       ├── mr_extract_array_66.json
│   │   │       ├── mr_extract_array_67.json
│   │   │       ├── mr_extract_array_68.json
│   │   │       └── mr_extract_array_69.json
│   │   └── isb-ai-117536
│   │       ├── logs
│   │       │   ├── script-117536.out
│   │       │   ├── slurm-117536_0.out
│   │       │   ├── slurm-117536_1.out
│   │       │   ├── slurm-117536_2.out
│   │       │   ├── slurm-117536_3.out
│   │       │   ├── slurm-117536_4.out
│   │       │   ├── slurm-117536_5.out
│   │       │   ├── slurm-117536_6.out
│   │       │   ├── slurm-117536_7.out
│   │       │   ├── slurm-117536_8.out
│   │       │   ├── slurm-117536_9.out
│   │       │   ├── slurm-117536_10.out
│   │       │   ├── slurm-117536_11.out
│   │       │   ├── slurm-117536_12.out
│   │       │   ├── slurm-117536_13.out
│   │       │   ├── slurm-117536_14.out
│   │       │   ├── slurm-117536_15.out
│   │       │   ├── slurm-117536_16.out
│   │       │   ├── slurm-117536_17.out
│   │       │   ├── slurm-117536_18.out
│   │       │   ├── slurm-117536_19.out
│   │       │   ├── slurm-117536_20.out
│   │       │   ├── slurm-117536_21.out
│   │       │   ├── slurm-117536_22.out
│   │       │   ├── slurm-117536_23.out
│   │       │   ├── slurm-117536_24.out
│   │       │   ├── slurm-117536_25.out
│   │       │   ├── slurm-117536_26.out
│   │       │   ├── slurm-117536_27.out
│   │       │   ├── slurm-117536_28.out
│   │       │   ├── slurm-117536_29.out
│   │       │   ├── slurm-117536_30.out
│   │       │   ├── slurm-117536_31.out
│   │       │   ├── slurm-117536_32.out
│   │       │   ├── slurm-117536_33.out
│   │       │   ├── slurm-117536_34.out
│   │       │   ├── slurm-117536_35.out
│   │       │   ├── slurm-117536_36.out
│   │       │   ├── slurm-117536_37.out
│   │       │   ├── slurm-117536_38.out
│   │       │   ├── slurm-117536_39.out
│   │       │   ├── slurm-117536_40.out
│   │       │   ├── slurm-117536_41.out
│   │       │   ├── slurm-117536_42.out
│   │       │   ├── slurm-117536_43.out
│   │       │   ├── slurm-117536_44.out
│   │       │   ├── slurm-117536_45.out
│   │       │   ├── slurm-117536_46.out
│   │       │   ├── slurm-117536_47.out
│   │       │   ├── slurm-117536_48.out
│   │       │   ├── slurm-117536_49.out
│   │       │   ├── slurm-117536_50.out
│   │       │   ├── slurm-117536_51.out
│   │       │   ├── slurm-117536_52.out
│   │       │   ├── slurm-117536_53.out
│   │       │   ├── slurm-117536_54.out
│   │       │   ├── slurm-117536_55.out
│   │       │   ├── slurm-117536_56.out
│   │       │   └── slurm-117536_57.out
│   │       ├── README.md
│   │       └── results
│   │           ├── mr_extract_array_0.json
│   │           ├── mr_extract_array_1.json
│   │           ├── mr_extract_array_2.json
│   │           ├── mr_extract_array_3.json
│   │           ├── mr_extract_array_4.json
│   │           ├── mr_extract_array_5.json
│   │           ├── mr_extract_array_6.json
│   │           ├── mr_extract_array_7.json
│   │           ├── mr_extract_array_8.json
│   │           ├── mr_extract_array_9.json
│   │           ├── mr_extract_array_10.json
│   │           ├── mr_extract_array_11.json
│   │           ├── mr_extract_array_12.json
│   │           ├── mr_extract_array_13.json
│   │           ├── mr_extract_array_14.json
│   │           ├── mr_extract_array_15.json
│   │           ├── mr_extract_array_16.json
│   │           ├── mr_extract_array_17.json
│   │           ├── mr_extract_array_18.json
│   │           ├── mr_extract_array_19.json
│   │           ├── mr_extract_array_20.json
│   │           ├── mr_extract_array_21.json
│   │           ├── mr_extract_array_22.json
│   │           ├── mr_extract_array_23.json
│   │           ├── mr_extract_array_24.json
│   │           ├── mr_extract_array_25.json
│   │           ├── mr_extract_array_26.json
│   │           ├── mr_extract_array_27.json
│   │           ├── mr_extract_array_28.json
│   │           ├── mr_extract_array_29.json
│   │           ├── mr_extract_array_30.json
│   │           ├── mr_extract_array_31.json
│   │           ├── mr_extract_array_32.json
│   │           ├── mr_extract_array_33.json
│   │           ├── mr_extract_array_34.json
│   │           ├── mr_extract_array_35.json
│   │           ├── mr_extract_array_36.json
│   │           ├── mr_extract_array_37.json
│   │           ├── mr_extract_array_38.json
│   │           ├── mr_extract_array_39.json
│   │           ├── mr_extract_array_40.json
│   │           ├── mr_extract_array_41.json
│   │           ├── mr_extract_array_42.json
│   │           ├── mr_extract_array_43.json
│   │           ├── mr_extract_array_44.json
│   │           ├── mr_extract_array_45.json
│   │           ├── mr_extract_array_46.json
│   │           ├── mr_extract_array_47.json
│   │           ├── mr_extract_array_48.json
│   │           ├── mr_extract_array_49.json
│   │           ├── mr_extract_array_50.json
│   │           ├── mr_extract_array_51.json
│   │           ├── mr_extract_array_52.json
│   │           ├── mr_extract_array_53.json
│   │           ├── mr_extract_array_54.json
│   │           ├── mr_extract_array_55.json
│   │           ├── mr_extract_array_56.json
│   │           └── mr_extract_array_57.json
│   ├── llm-results-aggregated
│   │   ├── deepseek-r1-distilled
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── gpt-4-1
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── gpt-4o
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── llama3
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── llama3-2
│   │   │   ├── processed_results.json
│   │   │   ├── processed_results_invalid.json
│   │   │   ├── processed_results_valid.json
│   │   │   └── raw_results.json
│   │   ├── logs
│   │   │   ├── deepseek-r1-distilled_schema_validation_errors.log
│   │   │   ├── gpt-4-1_schema_validation_errors.log
│   │   │   ├── gpt-4o_schema_validation_errors.log
│   │   │   ├── llama3-2_schema_validation_errors.log
│   │   │   ├── llama3_schema_validation_errors.log
│   │   │   └── o4-mini_schema_validation_errors.log
│   │   └── o4-mini
│   │       ├── processed_results.json
│   │       ├── processed_results_invalid.json
│   │       ├── processed_results_valid.json
│   │       └── raw_results.json
│   ├── mr-pubmed-data
│   │   ├── backups
│   │   │   ├── 2025-05-23
│   │   │   │   └── mr-pubmed-data.json
│   │   │   └── 2025-06-26
│   │   │       └── mr-pubmed-data.json
│   │   ├── mr-pubmed-data-sample-lite.json
│   │   ├── mr-pubmed-data-sample.json
│   │   └── mr-pubmed-data.json
│   └── openai-batch-results
│       └── o4-mini
│           └── mr_extract_openai_array_0_pilot.json
└── raw
  └── mr-pubmed-abstracts
    ├── data
    │   ├── 211_universities.txt
    │   ├── 985_universities.txt
    │   ├── author_processing_20250502.json
    │   ├── countries.csv
    │   ├── countries.xlsx
    │   ├── missing_pmids.txt
    │   ├── nature_index
    │   │   ├── 2016-research-leading-countries.csv
    │   │   ├── 2017-research-leading-countries.csv
    │   │   ├── 2018-research-leading-countries.csv
    │   │   ├── 2019-research-leading-countries.csv
    │   │   ├── 2020-research-leading-countries.csv
    │   │   ├── 2021-research-leading-countries.csv
    │   │   ├── 2022-research-leading-countries.csv
    │   │   ├── 2023-research-leading-countries.csv
    │   │   └── 2024-research-leading-countries.csv
    │   ├── predatory_journals.txt
    │   ├── pubmed.json
    │   ├── pubmed_abstracts.json
    │   ├── pubmed_abstracts2.json
    │   ├── pubmed_abstracts3.json
    │   ├── pubmed_abstracts_20250502.json
    │   ├── pubmed_abstracts_20250519.json
    │   ├── pubmed_abstracts_new.json
    │   ├── pubmed_abstracts_processed_20250502.json
    │   ├── pubmed_authors.json
    │   ├── pubmed_authors_20250502.json
    │   ├── pubmed_counts
    │   │   ├── PubMed_Timeline_Results_by_Year-3.csv
    │   │   ├── PubMed_Timeline_Results_by_Year-4.csv
    │   │   ├── PubMed_Timeline_Results_by_Year-5.csv
    │   │   ├── PubMed_Timeline_Results_by_Year-6.csv
    │   │   ├── PubMed_Timeline_Results_by_Year-7.csv
    │   │   └── 'PubMed by Year.csv'
    │   ├── pubmed_new.json
    │   └── results.RData
    └── data-pre-2025-05-23
      ├── 211_universities.txt
      ├── 985_universities.txt
      ├── countries.csv
      ├── countries.xlsx
      ├── missing_pmids.txt
      ├── nature_index
      │   ├── 2016-research-leading-countries.csv
      │   ├── 2017-research-leading-countries.csv
      │   ├── 2018-research-leading-countries.csv
      │   ├── 2019-research-leading-countries.csv
      │   ├── 2020-research-leading-countries.csv
      │   ├── 2021-research-leading-countries.csv
      │   ├── 2022-research-leading-countries.csv
      │   ├── 2023-research-leading-countries.csv
      │   └── 2024-research-leading-countries.csv
      ├── predatory_journals.txt
      ├── pubmed.json
      ├── pubmed_abstracts.json
      ├── pubmed_abstracts2.json
      ├── pubmed_abstracts3.json
      ├── pubmed_abstracts_new.json
      ├── pubmed_authors.json
      ├── pubmed_counts
      │   ├── PubMed_Timeline_Results_by_Year-3.csv
      │   ├── PubMed_Timeline_Results_by_Year-4.csv
      │   ├── PubMed_Timeline_Results_by_Year-5.csv
      │   ├── PubMed_Timeline_Results_by_Year-6.csv
      │   ├── PubMed_Timeline_Results_by_Year-7.csv
      │   └── 'PubMed by Year.csv'
      ├── pubmed_new.json
      └── results.RData
```
