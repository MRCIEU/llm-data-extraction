"""
Process and aggregate results from LLM batch inference.
"""

import json

import pandas as pd

from local_funcs import parsers
from yiutils.project_utils import find_project_root

# experiment ID directories for model results
DEEPSEEK_R1_DISTILLED_INPUT = "isb-ai-117256"
LLAMA3_2_INPUT = "isb-ai-117535"
LLAMA3_INPUT = "isb-ai-116732"


def process_deepseek_r1_distilled(model_config):
    json_files = list(model_config["input"].glob("*.json"))
    print(f"Deepseek-r1-distilled: {len(json_files)} files")

    # ---- load data ----
    json_data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = {"data": json.load(f), "filename": str(json_file.name)}
            json_data.append(data)

    # ---- raw results ----
    raw_results_df = pd.concat(
        [
            pd.DataFrame(data["data"]).assign(filename=data["filename"])
            for data in json_data
        ],
    ).reset_index(drop=True)
    print("Deepseek-r1-distilled raw_results_df:")
    raw_results_df.info()

    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)

    # ---- processed results ----
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
            "ab",
            "title",
            "metadata_thinking",
            "metadata",
            "results_thinking",
            "results",
        ]
    ]
    print("Deepseek-r1-distilled processed results_df:")
    results_df.info()

    output_path = model_config["output"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def process_llama3_2(model_config):
    path_to_llama3_2_result_dir = model_config["input"]
    assert path_to_llama3_2_result_dir.exists()

    json_files = list(path_to_llama3_2_result_dir.glob("*.json"))
    print(f"llama3-2: {len(json_files)} files")

    # ---- load data ----
    json_data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = {"data": json.load(f), "filename": str(json_file.name)}
            json_data.append(data)

    # ---- raw results ----
    raw_results_df = pd.concat(
        [
            pd.DataFrame(data["data"]).assign(filename=data["filename"])
            for data in json_data
        ],
    ).reset_index(drop=True)
    print("llama3-2 raw_results_df:")
    raw_results_df.info()

    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)

    # ---- processed results ----
    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )[
        [
            "pmid",
            "ab",
            "title",
            "metadata",
            "results",
        ]
    ]
    print("llama3-2 processed results_df:")
    results_df.info()

    output_path = model_config["output"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def process_llama3(model_config):
    json_files = list(model_config["input"].glob("*.json"))
    print(f"llama3: {len(json_files)} files")

    # ---- load data ----
    json_data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = {"data": json.load(f), "filename": str(json_file.name)}
            json_data.append(data)

    # ---- raw results ----
    raw_results_df = pd.concat(
        [
            pd.DataFrame(data["data"]).assign(filename=data["filename"])
            for data in json_data
        ],
    ).reset_index(drop=True)
    print("llama3 raw_results_df:")
    raw_results_df.info()

    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)

    # ---- processed results ----
    results_df = raw_results_df[
        [
            "pmid",
            "ab",
            "title",
            "metadata",
            "results",
        ]
    ]
    print("llama3 processed results_df:")
    results_df.info()

    output_path = model_config["output"] / "processed_results.json"
    with open(output_path, "w") as f:
        results_df.to_json(f, orient="records", indent=2)


def main():
    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data"
    assert data_dir.exists()

    output_dir = data_dir / "intermediate" / "llm-results-aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_results_dir = data_dir / "intermediate" / "llm-results"

    model_configs = {
        "deepseek-r1-distilled": {
            "input": llm_results_dir / DEEPSEEK_R1_DISTILLED_INPUT / "results",
            "output": output_dir / "deepseek-r1-distilled",
        },
        "llama3": {
            "input": llm_results_dir / LLAMA3_INPUT / "results",
            "output": output_dir / "llama3",
        },
        "llama3-2": {
            "input": llm_results_dir / LLAMA3_2_INPUT / "results",
            "output": output_dir / "llama3-2",
        },
    }
    for k, v in model_configs.items():
        assert v["input"].exists(), f"Input path for {k} does not exist: {v['input']}"

    process_deepseek_r1_distilled(model_configs["deepseek-r1-distilled"])
    process_llama3(model_configs["llama3"])
    process_llama3_2(model_configs["llama3-2"])


if __name__ == "__main__":
    main()
