"""
Process raw results from

data / intermediate / <MODEL-NAME> / raw_results.json.

to

data / intermediate / <MODEL-NAME> / processed_results.json.
"""

import json

import jsonschema
import pandas as pd
from loguru import logger

from local_funcs import parsers
from yiutils.project_utils import find_project_root


def validate_with_schema(item, schema, log_file) -> bool:
    try:
        jsonschema.validate(instance=item, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        with open(log_file, "a") as errfile:
            errfile.write(f"Validation error: {e.message}\nInstance: {json.dumps(item, ensure_ascii=False)}\n\n")
        return False


def process_deepseek_r1_distilled(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- schema files ----
    with open(model_config["schema"]["metadata"]) as f:
        meta_schema = json.load(f)
    with open(model_config["schema"]["results"]) as f:
        results_schema = json.load(f)

    # ---- process results ----
    # Parsing metadata and results
    logger.info("deepseek-r1-distilled: parsing metadata and results")
    results_df = raw_results_df.assign(
        metadata_thinking=lambda df: df["completion_metadata"].apply(
            parsers.extract_thinking
        ),
        metadata=lambda df: df["completion_metadata"].apply(
            parsers.extract_json_from_markdown
        ),
        results_thinking=lambda df: df["completion_results"].apply(
            parsers.extract_thinking
        ),
        results=lambda df: df["completion_results"].apply(
            parsers.extract_json_from_markdown
        ),
    )[
        [
            "pmid",
            "metadata_thinking",
            "metadata",
            "results_thinking",
            "results",
        ]
    ]
    logger.info("deepseek-r1-distilled: parsing metadata and results, done")

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)

    # ---- Schema validation ----
    model_config["error_log"].parent.mkdir(parents=True, exist_ok=True)
    results_df = results_df.assign(
        metadata_valid=lambda df: df["metadata"].apply(
            validate_with_schema, schema=meta_schema, log_file=model_config["error_log"]
        ),
        results_valid=lambda df: df["results"].apply(
            validate_with_schema, schema=results_schema, log_file=model_config["error_log"]
        ),
    )
    print(f"metadata_valid sum: {results_df['metadata_valid'].sum()}")
    print(f"results_valid sum: {results_df['results_valid'].sum()}")
    results_df_valid = results_df[
        results_df["metadata_valid"] & results_df["results_valid"]
    ].drop(columns=["metadata_valid", "results_valid"])

    results_df_invalid = results_df[
        ~(results_df["metadata_valid"] & results_df["results_valid"])
    ].drop(columns=["metadata_valid", "results_valid"])

    print("Deepseek-r1-distilled processed results_df:")
    results_df_valid.info()
    print("Deepseek-r1-distilled invalid results_df:")
    results_df_invalid.info()

    output_path = model_config["data_dir"] / "processed_results_valid.json"
    with open(output_path, "w") as f:
        results_df_valid.to_json(f, orient="records", indent=2)

    output_path_invalid = model_config["data_dir"] / "processed_results_invalid.json"
    with open(output_path_invalid, "w") as f:
        results_df_invalid.to_json(f, orient="records", indent=2)


def process_llama3_2(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- schema files ----
    with open(model_config["schema"]["metadata"]) as f:
        meta_schema = json.load(f)
    with open(model_config["schema"]["results"]) as f:
        results_schema = json.load(f)

    # ---- process results ----
    logger.info("llama3-2: parsing metadata and results")
    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )[
        [
            "pmid",
            "metadata",
            "results",
        ]
    ]
    logger.info("llama3-2: parsing metadata and results, done")

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)

    # ---- Schema validation ----
    model_config["error_log"].parent.mkdir(parents=True, exist_ok=True)
    results_df = results_df.assign(
        metadata_valid=lambda df: df["metadata"].apply(
            validate_with_schema, schema=meta_schema, log_file=model_config["error_log"]
        ),
        results_valid=lambda df: df["results"].apply(
            validate_with_schema, schema=results_schema, log_file=model_config["error_log"]
        ),
    )
    print(f"metadata_valid sum: {results_df['metadata_valid'].sum()}")
    print(f"results_valid sum: {results_df['results_valid'].sum()}")
    results_df_valid = results_df[
        results_df["metadata_valid"] & results_df["results_valid"]
    ].drop(columns=["metadata_valid", "results_valid"])

    results_df_invalid = results_df[
        ~(results_df["metadata_valid"] & results_df["results_valid"])
    ].drop(columns=["metadata_valid", "results_valid"])

    print("llama3-2 processed results_df:")
    results_df_valid.info()
    print("llama3-2 invalid results_df:")
    results_df_invalid.info()

    output_path = model_config["data_dir"] / "processed_results_valid.json"
    with open(output_path, "w") as f:
        results_df_valid.to_json(f, orient="records", indent=2)

    output_path_invalid = model_config["data_dir"] / "processed_results_invalid.json"
    with open(output_path_invalid, "w") as f:
        results_df_invalid.to_json(f, orient="records", indent=2)


def process_llama3(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- schema files ----
    with open(model_config["schema"]["metadata"]) as f:
        meta_schema = json.load(f)
    with open(model_config["schema"]["results"]) as f:
        results_schema = json.load(f)

    # ---- process results ----
    logger.info("llama3: parsing metadata and results")
    results_df = raw_results_df[
        [
            "pmid",
            "metadata",
            "results",
        ]
    ]
    logger.info("llama3: parsing metadata and results, done")

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)

    # ---- Schema validation ----
    model_config["error_log"].parent.mkdir(parents=True, exist_ok=True)
    results_df = results_df.assign(
        metadata_valid=lambda df: df["metadata"].apply(
            validate_with_schema, schema=meta_schema, log_file=model_config["error_log"]
        ),
        results_valid=lambda df: df["results"].apply(
            validate_with_schema, schema=results_schema, log_file=model_config["error_log"]
        ),
    )
    print(f"metadata_valid sum: {results_df['metadata_valid'].sum()}")
    print(f"results_valid sum: {results_df['results_valid'].sum()}")
    results_df_valid = results_df[
        results_df["metadata_valid"] & results_df["results_valid"]
    ].drop(columns=["metadata_valid", "results_valid"])

    results_df_invalid = results_df[
        ~(results_df["metadata_valid"] & results_df["results_valid"])
    ].drop(columns=["metadata_valid", "results_valid"])

    print("llama3 processed results_df:")
    results_df_valid.info()
    print("llama3 invalid results_df:")
    results_df_invalid.info()

    output_path = model_config["data_dir"] / "processed_results_valid.json"
    with open(output_path, "w") as f:
        results_df_valid.to_json(f, orient="records", indent=2)

    output_path_invalid = model_config["data_dir"] / "processed_results_invalid.json"
    with open(output_path_invalid, "w") as f:
        results_df_invalid.to_json(f, orient="records", indent=2)


def main():
    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data"
    agg_data_dir = data_dir / "intermediate" / "llm-results-aggregated"
    assert agg_data_dir.exists()

    model_configs = {
        "deepseek-r1-distilled": {
            "data_dir": agg_data_dir / "deepseek-r1-distilled",
            "schema": {
                "metadata": data_dir
                / "assets"
                / "data-schema"
                / "deepseek-r1-distilled"
                / "metadata.json.schema",
                "results": data_dir
                / "assets"
                / "data-schema"
                / "deepseek-r1-distilled"
                / "results.json.schema",
            },
            "error_log": proj_root / "output" / "logs" / "deepseek-r1-distilled_schema_validation_errors.log",
        },
        "llama3": {
            "data_dir": agg_data_dir / "llama3",
            "schema": {
                "metadata": data_dir
                / "assets"
                / "data-schema"
                / "llama3"
                / "metadata.json.schema",
                "results": data_dir
                / "assets"
                / "data-schema"
                / "llama3"
                / "results.json.schema",
            },
            "error_log": proj_root / "output" / "logs" / "llama3_schema_validation_errors.log",
        },
        "llama3-2": {
            "data_dir": agg_data_dir / "llama3-2",
            "schema": {
                "metadata": data_dir
                / "assets"
                / "data-schema"
                / "llama3-2"
                / "metadata.json.schema",
                "results": data_dir
                / "assets"
                / "data-schema"
                / "llama3-2"
                / "results.json.schema",
            },
            "error_log": proj_root / "output" / "logs" / "llama3-2_schema_validation_errors.log",
        },
    }
    for k, v in model_configs.items():
        assert v["data_dir"].exists(), (
            f"Input path for {k} does not exist: {v['data_dir']}"
        )

    logger.info("deepseek-r1-distilled")
    process_deepseek_r1_distilled(model_configs["deepseek-r1-distilled"])

    logger.info("llama3")
    process_llama3(model_configs["llama3"])

    logger.info("llama3_2")
    process_llama3_2(model_configs["llama3-2"])


if __name__ == "__main__":
    main()
