"""
Extract mr pubmed abstracts data using llms

NOTE: this should be part of a slurm job submission

Logics
- Processes abstracts in batches based on SLURM array task ID.
"""

import argparse
import json
from pathlib import Path

import torch
from environs import env
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_funcs import chat_funcs, prompt_funcs
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("justfile")
DATA_DIR = PROJECT_ROOT / "data"
PATH_DATA = DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"

MODEL_CONFIGS = {
    "llama-3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "llama-3.2": {"model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
    "deepseek-r1": {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
    "deepseek-prover": {"model_id": "deepseek-ai/DeepSeek-Prover-V2-7B"},
}


def main():
    # init
    proj_root = find_project_root()
    env.read_env()

    # Parse arguments
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

    # params
    # {{{
    array_task_id = args.array_id
    access_token = env("HUGGINGFACE_TOKEN")
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
    out_file = output_dir / f"mr_extract_array_{array_task_id}{pilot_suffix}.json"

    model_config_name = args.model
    assert (
        model_config_name is not None and model_config_name in MODEL_CONFIGS.keys()
    ), print("--model must not be empty and must be one of the configs")
    model_config = MODEL_CONFIGS[model_config_name]
    # }}}

    # Get abstracts
    with path_to_pubmed.open("r") as f:
        pubmed = json.load(f)
    print("Loaded abstracts")

    # Set up model
    model_id = model_config["model_id"]
    device = "cuda"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        token=access_token,
    )
    print("Loaded model")

    fulldata = []

    # Loop overall specified abstracts in the dataset
    for article_data in tqdm(pubmed[startpoint:endpoint]):
        try:
            message_metadata = prompt_funcs.make_message_metadata(article_data["ab"])
            message_results = prompt_funcs.make_message_results(article_data["ab"])
            completion_metadata = chat_funcs.extract(message_metadata, tokenizer, model)
            completion_results = chat_funcs.extract(message_results, tokenizer, model)
            result = {
                "completion_metadata": completion_metadata,
                "completion_results": completion_results,
            }
            output = dict(article_data, **result)
            fulldata.append(output)
        except Exception as e:
            print(f"""\n\n=========== {article_data["pmid"]} ==========""")
            print("""\n=========== FAILED! ==========""")
            # print(abstract)
            print(e)
            result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
            result2 = {"results": {}, "resultsinformation": {"error": f"Failed {e}"}}
            output = dict(article_data, **result1, **result2)
            print(f"Output: {output}")

    with out_file.open("w") as f:
        json.dump(fulldata, f, indent=4)


if __name__ == "__main__":
    main()
