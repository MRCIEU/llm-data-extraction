"""
Extract mr pubmed abstracts data using OpenAI models

NOTE: this should be part of a slurm job submission

Logics
- Processes abstracts in batches based on SLURM array task ID.
"""

import argparse
import json
from pathlib import Path

from environs import env
from openai import OpenAI
from tqdm import tqdm

from local_funcs import openai_funcs, prompt_funcs
from yiutils.project_utils import find_project_root

# ==== params ====
PROJECT_ROOT = find_project_root("justfile")
DATA_DIR = PROJECT_ROOT / "data"
PATH_DATA = DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"

MODEL_CONFIGS = {
    "o4-mini": {"model_id": "o4-mini"},
    "gpt-4o": {"model_id": "gpt-4o"},
}


def main():
    # ==== init ====
    proj_root = find_project_root()
    env.read_env()

    # ==== arg parser ====
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=proj_root / "output",
        help="Directory to save the output JSON file. Defaults to 'output' in the project root.",
    )
    parser.add_argument(
        "--path_data",
        type=str,
        default=str(PATH_DATA),
        help="Path to mr pubmed abstracts data",
    )
    parser.add_argument(
        "--array-id",
        type=int,
        default=0,
        help="Array ID",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Enable pilot mode. Defaults to False.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Which model to use. Must not be empty",
    )
    args = parser.parse_args()
    print(f"args: {args}")

    # ==== Config params ====
    # {{{
    array_task_id = args.array_id
    openai_api_key = env("OPENAI_API_KEY")
    num_docs = 100
    if args.pilot:
        print(f"Running in pilot mode with {num_docs} documents.")
        startpoint = 0
        endpoint = startpoint + num_docs
    else:
        startpoint = array_task_id * num_docs
        endpoint = startpoint + num_docs

    path_to_pubmed = Path(args.path_data)
    assert args.path_data is not None and path_to_pubmed.exists(), print(
        "pubmed data not found"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_suffix = "_pilot" if args.pilot else ""
    out_file = (
        output_dir / f"mr_extract_openai_array_{array_task_id}{pilot_suffix}.json"
    )

    model_config_name = args.model
    assert (
        model_config_name is not None and model_config_name in MODEL_CONFIGS.keys()
    ), print("--model must not be empty and must be one of the configs")
    model_config = MODEL_CONFIGS[model_config_name]
    # }}}

    # ==== data loading ====
    with path_to_pubmed.open("r") as f:
        pubmed = json.load(f)
    print("Loaded abstracts")

    # ==== Set up OpenAI client ====
    client = OpenAI(api_key=openai_api_key)
    print("Loaded OpenAI client")

    fulldata = []

    # ==== Loop overall specified abstracts in the dataset ====
    for article_data in tqdm(pubmed[startpoint:endpoint]):
        try:
            # Use prompt_funcs to generate messages for OpenAI
            message_metadata = prompt_funcs.make_message_metadata(article_data["ab"])
            # Call the appropriate OpenAI function
            if model_config_name == "o4-mini":
                completion_metadata = openai_funcs.get_o4_mini_result(
                    client, article_data
                )
            elif model_config_name == "gpt-4o":
                completion_metadata = openai_funcs.get_gpt_4o_result(
                    client, article_data
                )
            else:
                raise ValueError(f"Unknown model: {model_config_name}")
            result = {
                "completion_metadata": completion_metadata,
            }
            output = dict(article_data, **result)
            fulldata.append(output)
        except Exception as e:
            print(
                f"""\n\n=========== {article_data.get("pmid", "NO PMID")} =========="""
            )
            print("""\n=========== FAILED! ==========""")
            print(e)
            result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
            output = dict(article_data, **result1)
            print(f"Output: {output}")

    # ==== wrap up ====
    with out_file.open("w") as f:
        json.dump(fulldata, f, indent=4)
    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
