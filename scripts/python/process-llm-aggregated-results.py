"""
Process raw results from

data / intermediate / <MODEL-NAME> / raw_results.json.

to

data / intermediate / <MODEL-NAME> / processed_results.json.
"""

import argparse
import json

import jsonschema
import pandas as pd
from loguru import logger

from local_funcs import parsers
from yiutils.project_utils import find_project_root

# Remapping of commonly wrong keys
RESULT_REMAPS = {
    "95% confidence interval": "95% CI",
    "95_CI": "95% CI",
    "standard error": "SE",
    "odds_ratio": "odds ratio",
    "hazard_ratio": "hazard ratio",
    "Direction": "direction",
}


def validate_item_with_schema(item, schema, log_file) -> bool:
    try:
        jsonschema.validate(instance=item, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        with open(log_file, "a") as errfile:
            instance = json.dumps(item, ensure_ascii=False, indent=2)
            errfile.write(f"Validation error: {e.message}\nInstance: \n{instance}\n\n")
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
    """
    res = metadata
    if isinstance(metadata, dict) and "metadata" in metadata.keys():
        res = metadata["metadata"]
    return res


def process_results(results):
    """
    - If "results" property is found, return its value
    - After this step, results is expected to be a list of dicts.
      If a key "95% confidence interval" is found in any dict, replace it with "95% CI".
    """
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


def process_deepseek_r1_distilled(model_config, global_config):
    # ---- init ----
    logger.info(f"{model_config['name']}")

    raw_results_df = load_raw_results(model_config)
    logger.info(f"{model_config['name']}: raw_results_df info")
    raw_results_df.info()

    meta_schema, results_schema = load_schema_files(global_config)

    # ---- process results ----
    # Parsing metadata and results
    logger.info(f"{model_config['name']}: parsing metadata and results")
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
    )
    logger.info(f"{model_config['name']}: parsing metadata and results, done")
    results_df = results_df[
        ["pmid", "metadata_thinking", "metadata", "results_thinking", "results"]
    ]
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


def process_llama3_2(model_config, global_config):
    # ---- init ----
    logger.info(f"{model_config['name']}")

    raw_results_df = load_raw_results(model_config)
    logger.info(f"{model_config['name']}: raw_results_df info")
    raw_results_df.info()

    meta_schema, results_schema = load_schema_files(global_config)

    # ---- process results ----
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


def process_llama3(model_config, global_config):
    # ---- init ----
    logger.info(f"{model_config['name']}")

    raw_results_df = load_raw_results(model_config)
    logger.info(f"{model_config['name']}: raw_results_df info")
    raw_results_df.info()

    meta_schema, results_schema = load_schema_files(global_config)

    # ---- process results ----
    results_df = raw_results_df[["pmid", "metadata", "results"]]
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


def main():
    parser = argparse.ArgumentParser(
        description="Process and validate LLM aggregated results."
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["deepseek-r1-distilled", "llama3", "llama3-2"],
        help="Specify one or more models to process. If not supplied, all models will be processed.",
    )
    args = parser.parse_args()

    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data"
    agg_data_dir = data_dir / "intermediate" / "llm-results-aggregated"
    assert agg_data_dir.exists()

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
        "deepseek-r1-distilled": {
            "name": "deepseek-r1-distilled",
            "data_dir": agg_data_dir / "deepseek-r1-distilled",
            "error_log": proj_root
            / "output"
            / "logs"
            / "deepseek-r1-distilled_schema_validation_errors.log",
            "func": process_deepseek_r1_distilled,
        },
        "llama3": {
            "name": "llama3",
            "data_dir": agg_data_dir / "llama3",
            "error_log": proj_root
            / "output"
            / "logs"
            / "llama3_schema_validation_errors.log",
            "func": process_llama3,
        },
        "llama3-2": {
            "name": "llama3-2",
            "data_dir": agg_data_dir / "llama3-2",
            "error_log": proj_root
            / "output"
            / "logs"
            / "llama3-2_schema_validation_errors.log",
            "func": process_llama3_2,
        },
    }
    for k, v in model_configs.items():
        assert v["data_dir"].exists(), (
            f"Input path for {k} does not exist: {v['data_dir']}"
        )

    if not args.model:
        for model, model_config in model_configs.items():
            model_config["func"](model_config, global_config)
    else:
        for model in args.model:
            model_config = model_configs[model]
            model_config["func"](model_config, global_config)


if __name__ == "__main__":
    main()
