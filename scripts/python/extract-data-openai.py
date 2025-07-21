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
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from local_funcs import openai_funcs, prompt_funcs
from local_funcs.schema_funcs import load_schema_data
from yiutils.chunking import calculate_chunk_start_end
from yiutils.project_utils import find_project_root

# ==== params ====
PROJECT_ROOT = find_project_root("justfile")
DATA_DIR = PROJECT_ROOT / "data"
PATH_DATA = DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data-sample.json"
PATH_SCHEMA_DIR = DATA_DIR / "assets" / "data-schema" / "example-data"

MODEL_CONFIGS = {
    "o4-mini": {"model_id": "o4-mini", "chat_func": openai_funcs.get_o4_mini_result},
    "gpt-4o": {"model_id": "gpt-4o", "chat_func": openai_funcs.get_gpt_4o_result},
}
PILOT_NUM_DOCS = 5
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
        "--array-length",
        type=int,
        default=ARRAY_LENGTH,
        help=f"Number of chunks to split the data into (default: {ARRAY_LENGTH})",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set, print config and schema data then exit without processing.",
    )
    args = parser.parse_args()
    print(f"args: {args}")
    return args


def get_config(args):
    env.read_env()
    array_task_id = args.array_id
    openai_api_key = env("PROJECT_OPENAI_API_KEY")
    pilot_num_docs = PILOT_NUM_DOCS
    array_length = args.array_length

    # ==== chunking data ====
    path_to_pubmed = Path(args.path_data)
    assert args.path_data is not None and path_to_pubmed.exists(), print(
        "pubmed data not found"
    )

    # Load data length for correct startpoint/endpoint calculation
    print(f"Loading data from {path_to_pubmed}")
    with path_to_pubmed.open("r") as f:
        pubmed_data = json.load(f)
    data_length = len(pubmed_data)

    # Use calculate_start_end to determine startpoint and endpoint
    startpoint, endpoint = calculate_chunk_start_end(
        chunk_id=array_task_id,
        num_chunks=array_length,
        data_length=data_length,
        pilot_num_docs=pilot_num_docs,
        pilot=args.pilot,
        verbose=True,
    )
    if startpoint is None or endpoint is None:
        print(f"WARNING: startpoint {startpoint} endpoint {endpoint}")
        sys.exit(0)
    pubmed_data = pubmed_data[startpoint:endpoint]

    # ==== model config ====
    model_config_name = args.model
    assert (
        model_config_name is not None and model_config_name in MODEL_CONFIGS.keys()
    ), print("--model must not be empty and must be one of the configs")
    model_config = MODEL_CONFIGS[model_config_name]

    # ==== output directory and file ====
    output_dir = Path(args.output_dir) / model_config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_suffix = "_pilot" if args.pilot else ""
    out_file = (
        output_dir / f"mr_extract_openai_array_{array_task_id}{pilot_suffix}.json"
    )

    # ==== return config and data ====
    config = {
        "array_task_id": array_task_id,
        "openai_api_key": openai_api_key,
        "num_docs": len(pubmed_data),
        "path_to_pubmed": path_to_pubmed,
        "output_dir": output_dir,
        "out_file": out_file,
        "model_config_name": model_config_name,
        "model_config": model_config,
    }
    print(f"Config: {config}")
    print(f"Loaded {len(pubmed_data)} abstracts from {path_to_pubmed}")
    res = (config, pubmed_data)
    return res


def load_pubmed(path_to_pubmed):
    with path_to_pubmed.open("r") as f:
        pubmed_data = json.load(f)
    print("Loaded abstracts")
    return pubmed_data


def setup_openai_client(api_key):
    client = OpenAI(api_key=api_key)
    print("Loaded OpenAI client")
    return client


def process_abstract(article_data, schema_data, client, model_config):
    try:
        chat_func = model_config["chat_func"]
        input_prompt_metadata = prompt_funcs.make_message_metadata_new(
            abstract=article_data["ab"],
            json_example=schema_data["metadata"]["example"],
            json_schema=schema_data["metadata"]["schema"],
        )
        input_prompt_results = prompt_funcs.make_message_results_new(
            abstract=article_data["ab"],
            json_example=schema_data["results"]["example"],
            json_schema=schema_data["results"]["schema"],
        )
        completion_metadata = chat_func(client, input_prompt_metadata)
        completion_results = chat_func(client, input_prompt_results)
        result = {
            "completion_metadata": completion_metadata,
            "completion_results": completion_results,
        }
        output = dict(article_data, **result)
        return output
    except Exception as e:
        print(f"""\n\n=========== {article_data.get("pmid", "NO PMID")} ==========""")
        print("""\n=========== FAILED! ==========""")
        print(e)
        result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
        output = dict(article_data, **result1)
        print(f"Output: {output}")
        return output


def main():
    # ==== init ====
    args = parse_args()
    config, pubmed_data = get_config(args=args)
    client = setup_openai_client(api_key=config["openai_api_key"])
    schema_data = load_schema_data()

    if args.dry_run:
        print("Dry run enabled. Printing config and schema_data, then exiting.")
        print("Config:")
        print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
        sys.exit(0)

    # ==== process abstracts ====
    logger.info("Processing abstracts")
    fulldata = []
    for article_data in tqdm(pubmed_data):
        output = process_abstract(
            article_data=article_data,
            schema_data=schema_data,
            client=client,
            model_config=config["model_config"],
        )
        fulldata.append(output)
    logger.info("Processing abstracts, done")

    # ==== save output ====
    out_file = config["out_file"]
    logger.info(f"Wrote results to {out_file}")
    with out_file.open("w") as f:
        json.dump(fulldata, f, indent=4)


if __name__ == "__main__":
    main()
