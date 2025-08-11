"""
Aggregate results from OpenAI SP batch inference from

data / intermediate / openai-sp-batch-results / <MODEL-NAME> / mr_extract_openai_sp.json.

to

data / intermediate / llm-results-aggregated-sp / <MODEL-NAME> / raw_results.json.
"""

import argparse
import json
import sys

import pandas as pd

from yiutils.project_utils import find_project_root


def process_openai_model(model_config):
    """Process OpenAI model results from SP extraction"""
    input_file = model_config["input"]
    print(f"{model_config['name']}: processing {input_file}")

    assert input_file.exists(), f"Input file does not exist: {input_file}"

    # ---- load data ----
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"{model_config['name']}: loaded {len(data)} records")

    # ---- create raw results DataFrame ----
    raw_results_df = pd.DataFrame(data)
    print(f"{model_config['name']} raw_results_df:")
    raw_results_df.info()

    # ---- save output ----
    output_path = model_config["output"] / "raw_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        raw_results_df.to_json(f, orient="records", indent=2)

    print(f"{model_config['name']}: saved to {output_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate OpenAI SP batch results for selected models."
    )
    # ---- --models ----
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["o4-mini", "gpt-4o", "gpt-4-1", "gpt-5", "gpt-5-mini"],
        help="List of models to process (space separated).",
    )
    # ---- --all ----
    parser.add_argument("--all", action="store_true", help="Process all models.")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")

    data_dir = proj_root / "data"
    assert data_dir.exists()

    output_dir = data_dir / "intermediate" / "llm-results-aggregated-sp"
    output_dir.mkdir(parents=True, exist_ok=True)

    openai_sp_results_dir = data_dir / "intermediate" / "openai-sp-batch-results"
    assert openai_sp_results_dir.exists(), (
        f"OpenAI SP results directory does not exist: {openai_sp_results_dir}"
    )

    model_configs = {
        "o4-mini": {
            "name": "o4-mini",
            "input": openai_sp_results_dir / "o4-mini" / "mr_extract_openai_sp.json",
            "output": output_dir / "o4-mini",
            "func": process_openai_model,
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "input": openai_sp_results_dir / "gpt-4o" / "mr_extract_openai_sp.json",
            "output": output_dir / "gpt-4o",
            "func": process_openai_model,
        },
        "gpt-4-1": {
            "name": "gpt-4-1",
            "input": openai_sp_results_dir / "gpt-4-1" / "mr_extract_openai_sp.json",
            "output": output_dir / "gpt-4-1",
            "func": process_openai_model,
        },
        "gpt-5": {
            "name": "gpt-5",
            "input": openai_sp_results_dir / "gpt-5" / "mr_extract_openai_sp.json",
            "output": output_dir / "gpt-5",
            "func": process_openai_model,
        },
        "gpt-5-mini": {
            "name": "gpt-5-mini",
            "input": openai_sp_results_dir / "gpt-5-mini" / "mr_extract_openai_sp.json",
            "output": output_dir / "gpt-5-mini",
            "func": process_openai_model,
        },
    }

    # Validate arguments
    if not args.all and not args.models:
        print(
            "Error: You must specify either --all or --models <model>.", file=sys.stderr
        )
        parser.print_help()
        sys.exit(1)

    # If --all, process all models
    if args.all:
        for model, config in model_configs.items():
            if config["input"].exists():
                print(f"Processing model: {model}")
                config["func"](config)
            else:
                print(
                    f"Warning: Input file for model '{model}' does not exist: {config['input']}"
                )
        return

    # If --models, process only those models
    for model in args.models:
        if model not in model_configs:
            print(f"Error: Unknown model '{model}'.", file=sys.stderr)
            continue
        config = model_configs[model]
        if not config["input"].exists():
            print(
                f"Warning: Input file for model '{model}' does not exist: {config['input']}"
            )
            continue
        print(f"Processing model: {model}")
        config["func"](config)


if __name__ == "__main__":
    main()
