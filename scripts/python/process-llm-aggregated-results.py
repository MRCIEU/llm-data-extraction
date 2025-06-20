"""
Process raw results from

data / intermediate / <MODEL-NAME> / raw_results.json.

to

data / intermediate / <MODEL-NAME> / processed_results.json.
"""

import pandas as pd
from loguru import logger

from local_funcs import parsers
from yiutils.project_utils import find_project_root


def process_deepseek_r1_distilled(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- process results ----
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
    print("Deepseek-r1-distilled processed results_df:")
    results_df.info()

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def process_llama3_2(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- process results ----
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
    print("llama3-2 processed results_df:")
    results_df.info()

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def process_llama3(model_config):
    # ---- read raw results ----
    path_to_raw_results = model_config["data_dir"] / "raw_results.json"
    assert path_to_raw_results.exists(), (
        f"Raw results file does not exist: {path_to_raw_results}"
    )
    raw_results_df = pd.read_json(path_to_raw_results, orient="records")

    # ---- process results ----
    results_df = raw_results_df[
        [
            "pmid",
            "metadata",
            "results",
        ]
    ]
    print("llama3 processed results_df:")
    results_df.info()

    output_path = model_config["data_dir"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def main():
    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data" / "intermediate" / "llm-results-aggregated"
    assert data_dir.exists()

    model_configs = {
        "deepseek-r1-distilled": {
            "data_dir": data_dir / "deepseek-r1-distilled",
        },
        "llama3": {
            "data_dir": data_dir / "llama3",
        },
        "llama3-2": {
            "data_dir": data_dir / "llama3-2",
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
