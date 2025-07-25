"""
Aggregate results from LLM batch inference from

data / intermediate / llm-results / <EXPERIMENT-ID> / results / *.json.

to

data / intermediate / llm-results-aggregated / <MODEL-NAME> / raw_results.json.
"""

import argparse
import json
import sys

import pandas as pd

from yiutils.project_utils import find_project_root

# experiment ID directories for model results
DEEPSEEK_R1_DISTILLED_INPUT = "isb-ai-117256"
LLAMA3_2_INPUT = "isb-ai-117535"
LLAMA3_INPUT = "isb-ai-116732"
OPENAI_O4MINI_INPUT = "bc4-12390298"
OPENAI_GPT4_1_INPUT = "TODO"


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


def process_o4_mini(model_config):
    path_to_result_dir = model_config["input"]
    assert path_to_result_dir.exists()

    json_files = list(path_to_result_dir.glob("*.json"))
    print(f"o4-mini: {len(json_files)} files")

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
    print("o4-mini raw_results_df:")
    raw_results_df.info()

    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)


def process_gpt_4_1(model_config):
    path_to_result_dir = model_config["input"]
    assert path_to_result_dir.exists()

    json_files = list(path_to_result_dir.glob("*.json"))
    print(f"gpt-4-1: {len(json_files)} files")

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
    print("gpt-4-1 raw_results_df:")
    raw_results_df.info()

    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LLM batch results for selected models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["deepseek-r1-distilled", "llama3", "llama3-2", "o4-mini", "gpt-4-1"],
        help="List of models to process (space separated).",
    )
    parser.add_argument("--all", action="store_true", help="Process all models.")
    args = parser.parse_args()

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
            "func": process_deepseek_r1_distilled,
        },
        "llama3": {
            "input": llm_results_dir / LLAMA3_INPUT / "results",
            "output": output_dir / "llama3",
            "func": process_llama3,
        },
        "llama3-2": {
            "input": llm_results_dir / LLAMA3_2_INPUT / "results",
            "output": output_dir / "llama3-2",
            "func": process_llama3_2,
        },
        "o4-mini": {
            "input": llm_results_dir / OPENAI_O4MINI_INPUT / "results" / "o4-mini",
            "output": output_dir / "o4-mini",
            "func": process_o4_mini,
        },
        "gpt-4-1": {
            "input": llm_results_dir / OPENAI_GPT4_1_INPUT / "results" / "gpt-4-1",
            "output": output_dir / "gpt-4-1",
            "func": process_gpt_4_1,
        },
    }

    # Validate arguments
    if not args.all and not args.models:
        print(
            "Error: You must specify either --all or --models <model>.", file=sys.stderr
        )
        parser.print_help()
        sys.exit(1)

    # If --all, process all models with a defined function
    if args.all:
        for model, config in model_configs.items():
            if config["func"] is not None:
                print(f"Processing model: {model}")
                config["func"](config)
        return

    # If --models, process only those models
    for model in args.models:
        if model not in model_configs:
            print(f"Error: Unknown model '{model}'.", file=sys.stderr)
            continue
        config = model_configs[model]
        if config["func"] is None:
            print(
                f"Warning: No processing function defined for model '{model}'. Skipping.",
                file=sys.stderr,
            )
            continue
        print(f"Processing model: {model}")
        config["func"](config)


if __name__ == "__main__":
    main()
