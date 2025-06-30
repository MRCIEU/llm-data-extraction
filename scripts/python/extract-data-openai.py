"""
Extract mr pubmed abstracts data using OpenAI models

NOTE: this should be part of a slurm job submission

Logics
- Processes abstracts in batches based on SLURM array task ID.
"""

import argparse
import json
import sys
from pathlib import Path

from environs import env
from openai import OpenAI
from tqdm import tqdm

from local_funcs import openai_funcs, prompt_funcs
from local_funcs.funcs import calculate_start_end
from yiutils.project_utils import find_project_root

# ==== params ====
PROJECT_ROOT = find_project_root("justfile")
DATA_DIR = PROJECT_ROOT / "data"
PATH_DATA = DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data-sample.json"

MODEL_CONFIGS = {
    "o4-mini": {"model_id": "o4-mini"},
    "gpt-4o": {"model_id": "gpt-4o"},
}
NUM_DOCS = 100
ARRAY_LENGTH = 30


def parse_args():
    proj_root = find_project_root()
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
    return args


def get_config(args):
    env.read_env()
    array_task_id = args.array_id
    openai_api_key = env("OPENAI_API_KEY")
    num_docs = NUM_DOCS
    array_length = ARRAY_LENGTH

    path_to_pubmed = Path(args.path_data)
    assert args.path_data is not None and path_to_pubmed.exists(), print(
        "pubmed data not found"
    )

    # Load data length for correct startpoint/endpoint calculation
    with path_to_pubmed.open("r") as f:
        pubmed = json.load(f)
    data_length = len(pubmed)

    # Use calculate_start_end to determine startpoint and endpoint
    startpoint, endpoint = calculate_start_end(
        array_task_id=array_task_id,
        array_length=array_length,
        num_docs=num_docs,
        data_length=data_length,
        pilot=args.pilot,
    )
    if startpoint is None or endpoint is None:
        print(
            f"WARNING: startpoint {startpoint} exceeds data length {data_length}. Exiting."
        )
        sys.exit(0)
    if args.pilot:
        print(f"Running in pilot mode with {num_docs} documents.")
    elif endpoint > data_length:
        print(
            f"Endpoint {endpoint} exceeds data length {data_length}. Truncating to {data_length}."
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

    return {
        "array_task_id": array_task_id,
        "openai_api_key": openai_api_key,
        "num_docs": num_docs,
        "startpoint": startpoint,
        "endpoint": endpoint,
        "path_to_pubmed": path_to_pubmed,
        "output_dir": output_dir,
        "out_file": out_file,
        "model_config_name": model_config_name,
        "model_config": model_config,
        "pubmed": pubmed,
    }


def load_pubmed(path_to_pubmed):
    with path_to_pubmed.open("r") as f:
        pubmed = json.load(f)
    print("Loaded abstracts")
    return pubmed


def setup_openai_client(api_key):
    client = OpenAI(api_key=api_key)
    print("Loaded OpenAI client")
    return client


def process_abstracts(pubmed, startpoint, endpoint, client, model_config_name):
    fulldata = []
    for article_data in tqdm(pubmed[startpoint:endpoint]):
        try:
            # Use prompt_funcs to generate messages for OpenAI
            prompt_funcs.make_message_metadata(article_data["ab"])
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
    return fulldata


def write_output(fulldata, out_file):
    with out_file.open("w") as f:
        json.dump(fulldata, f, indent=4)
    print(f"Wrote results to {out_file}")


def main():
    args = parse_args()
    config = get_config(args=args)
    client = setup_openai_client(api_key=config["openai_api_key"])
    fulldata = process_abstracts(
        pubmed=config["pubmed"],
        startpoint=config["startpoint"],
        endpoint=config["endpoint"],
        client=client,
        model_config_name=config["model_config_name"],
    )
    write_output(fulldata=fulldata, out_file=config["out_file"])


if __name__ == "__main__":
    main()
