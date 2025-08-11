"""
Process raw results from OpenAI SP extraction from

data / intermediate / llm-results-aggregated-sp / <MODEL-NAME> / raw_results.json.

to

data / intermediate / llm-results-aggregated-sp / <MODEL-NAME> / processed_results.json.

NOTE: This uses the legacy prompt functions (make_message_metadata and make_message_results)
but still applies schema validation to ensure data quality.
"""

import argparse
import json
from typing import Optional

import jsonschema
import pandas as pd
from loguru import logger

from local_funcs import parsers
from yiutils.project_utils import find_project_root

# Remapping of commonly wrong keys
RESULT_REMAPS = {
    "95% confidence interval": "95% CI",
    "95%_CI": "95% CI",
    "standard error": "SE",
    "odds_ratio": "odds ratio",
    "hazard_ratio": "hazard ratio",
    "Direction": "direction",
}


def validate_item_with_schema(item, schema, log_file: Optional[str] = None) -> bool:
    try:
        jsonschema.validate(instance=item, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        if log_file is not None:
            with open(log_file, "a") as errfile:
                instance = json.dumps(item, ensure_ascii=False, indent=2)
                errfile.write(
                    f"Validation error: {e.message}\nInstance: \n{instance}\n\n"
                )
        else:
            instance = json.dumps(item, ensure_ascii=False, indent=2)
            logger.info(f"Validation error: {e.message}\nInstance: \n{instance}\n\n")
        return False


def load_raw_results(model_config) -> pd.DataFrame:
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")
    return raw_results_df


def load_schema_files(global_config) -> tuple:
    # ---- schema files ----
    with open(global_config["schema"]["metadata"]) as f:
        meta_schema = json.load(f)
    with open(global_config["schema"]["results"]) as f:
        results_schema = json.load(f)
    return (meta_schema, results_schema)


def process_metadata(metadata):
    """
    - If "metadata" property is found, return its value
    - If an error occurs, return None
    """
    try:
        res = metadata
        if isinstance(res, dict) and "metadata" in res.keys():
            res = res["metadata"]
        if isinstance(res, dict) and "metainformation" in res.keys():
            res.pop("metainformation")
        return res
    except Exception as e:
        print(f"Error processing metadata: \n{e}")
        return None


def process_results(results):
    """
    - If "results" property is found, return its value
    - After this step, results is expected to be a list of dicts.
      If a key "95% confidence interval" is found in any dict, replace it with "95% CI".
    - If an error occurs, return None
    """
    try:
        res = results
        if isinstance(results, dict) and "results" in results.keys():
            res = results["results"]
        if isinstance(res, list):
            for d in res:
                if isinstance(d, dict):
                    for k, v in RESULT_REMAPS.items():
                        if k in d:
                            d[v] = d.pop(k)
        return res
    except Exception as e:
        print(f"Error processing results: \n{e}")
        return None


def validate_schema(model_config, results_df, meta_schema, results_schema):
    log_file = model_config["error_log"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("")
    results_df = results_df.assign(
        metadata_valid=lambda df: df["metadata"].apply(
            validate_item_with_schema, schema=meta_schema, log_file=log_file
        ),
        results_valid=lambda df: df["results"].apply(
            validate_item_with_schema, schema=results_schema, log_file=log_file
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

    print(f"{model_config['name']}: valid results_df")
    results_df_valid.info()
    print(f"{model_config['name']}: invalid results_df")
    results_df_invalid.info()

    output_path = model_config["data_dir"] / "processed_results_valid.json"
    with open(output_path, "w") as f:
        results_df_valid.to_json(f, orient="records", indent=2)

    output_path_invalid = model_config["data_dir"] / "processed_results_invalid.json"
    with open(output_path_invalid, "w") as f:
        results_df_invalid.to_json(f, orient="records", indent=2)


# ==== Processing functions for each model ====


def process_openai_model(model_config, global_config):
    """
    Process OpenAI model results from SP extraction.
    Uses legacy prompt functions but still applies schema validation.
    """
    # ---- init ----
    logger.info(f"{model_config['name']}")

    raw_results_df = load_raw_results(model_config)
    logger.info(f"{model_config['name']}: raw_results_df info")
    raw_results_df.info()

    meta_schema, results_schema = load_schema_files(global_config)

    # ---- process results ----
    # For SP results using legacy prompts, we parse JSON directly from the completion text
    logger.info(f"{model_config['name']}: parsing metadata and results")
    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )
    logger.info(f"{model_config['name']}: parsing metadata and results, done")

    results_df = results_df[["pmid", "metadata", "results"]]
    results_df = results_df.dropna(subset=["metadata", "results"]).assign(
        metadata=lambda df: df["metadata"].apply(process_metadata),
        results=lambda df: df["results"].apply(process_results),
    )
    results_df.info()

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)

    # ---- Schema validation ----
    validate_schema(model_config, results_df, meta_schema, results_schema)

    logger.info(f"{model_config['name']}: saved processed results to {output_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process and validate OpenAI SP aggregated results."
    )
    # ---- --models ----
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["o4-mini", "gpt-4o", "gpt-4-1", "gpt-5", "gpt-5-mini"],
        help="Specify one or more models to process. If not supplied, all models will be processed.",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data"
    agg_data_dir = data_dir / "intermediate" / "llm-results-aggregated-sp"
    assert agg_data_dir.exists(), (
        f"Aggregated data directory does not exist: {agg_data_dir}"
    )

    global_config = {
        "schema": {
            "metadata": data_dir
            / "assets"
            / "data-schema"
            / "processed_results"
            / "metadata.schema.json",
            "results": data_dir
            / "assets"
            / "data-schema"
            / "processed_results"
            / "results.schema.json",
        }
    }

    model_configs = {
        "o4-mini": {
            "name": "o4-mini",
            "data_dir": agg_data_dir / "o4-mini",
            "error_log": agg_data_dir / "logs" / "o4-mini_schema_validation_errors.log",
            "func": process_openai_model,
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "data_dir": agg_data_dir / "gpt-4o",
            "error_log": agg_data_dir / "logs" / "gpt-4o_schema_validation_errors.log",
            "func": process_openai_model,
        },
        "gpt-4-1": {
            "name": "gpt-4-1",
            "data_dir": agg_data_dir / "gpt-4-1",
            "error_log": agg_data_dir / "logs" / "gpt-4-1_schema_validation_errors.log",
            "func": process_openai_model,
        },
        "gpt-5": {
            "name": "gpt-5",
            "data_dir": agg_data_dir / "gpt-5",
            "error_log": agg_data_dir / "logs" / "gpt-5_schema_validation_errors.log",
            "func": process_openai_model,
        },
        "gpt-5-mini": {
            "name": "gpt-5-mini",
            "data_dir": agg_data_dir / "gpt-5-mini",
            "error_log": agg_data_dir / "logs" / "gpt-5-mini_schema_validation_errors.log",
            "func": process_openai_model,
        },
    }

    if not args.models:
        for model, model_config in model_configs.items():
            if model_config["data_dir"].exists():
                logger.info(f"Processing model: {model}")
                model_config["func"](model_config, global_config)
            else:
                logger.warning(
                    f"Input path for {model} does not exist: {model_config['data_dir']}"
                )
    else:
        for model in args.models:
            model_config = model_configs[model]
            if model_config["data_dir"].exists():
                logger.info(f"Processing model: {model}")
                model_config["func"](model_config, global_config)
            else:
                logger.warning(
                    f"Input path for {model} does not exist: {model_config['data_dir']}"
                )


if __name__ == "__main__":
    main()
