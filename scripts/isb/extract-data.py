import argparse
import json
from pathlib import Path

import torch
from environs import env
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

from local_funcs import chat_funcs, prompt_funcs
from yiutils.project_utils import find_project_root


def main():
    # init
    proj_root = find_project_root()
    env.read_env()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=proj_root / "output",
        help="Directory to save the output JSON file. Defaults to 'output' in the project root.",
    )
    parser.add_argument(
        "--path_data",
        type=str,
        default=None,
        help="Path to mr pubmed abstracts data",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Enable pilot mode. Defaults to False.",
    )
    args = parser.parse_args()
    print(f"args: {args}")

    # params
    # {{{
    array_task_id = env.int("SLURM_ARRAY_TASK_ID")
    access_token = env("HUGGINGFACE_TOKEN")
    num_docs = 100
    if args.pilot:
        print("Running in pilot mode with 100 documents.")
        startpoint = 0
        endpoint = 101
    else:
        startpoint = array_task_id * num_docs
        endpoint = startpoint + num_docs + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"mr_extract_llama3_sample_array_{array_task_id}.json"
    # }}}

    # Get abstracts
    path_to_pubmed = Path(args.path_data)
    assert args.path_data is not None and path_to_pubmed.exists(), print(
        "pubmed data not found"
    )
    with path_to_pubmed.open("r") as f:
        pubmed = json.load(f)
    print("Loaded abstracts")

    # Set up model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    if args.pilot:
        quantization_config = QuantoConfig(weights="int4")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            token=access_token,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            token=access_token,
        )
    print("Loaded model")

    fulldata = []

    # Loop over all specified abstracts in the dataset
    for abstract in tqdm(pubmed[startpoint:endpoint]):
        try:
            message_metadata = prompt_funcs.make_message_metadata(abstract["ab"])
            message_results = prompt_funcs.make_message_results(abstract["ab"])
            completion_metadata = chat_funcs.extract(message_metadata, tokenizer, model)
            completion_results = chat_funcs.extract(message_results, tokenizer, model)
            result_metadata = chat_funcs.clean_result(completion_metadata)
            result_results = chat_funcs.clean_result(completion_results)
            output = dict(abstract, **result_metadata, **result_results)
            fulldata.append(output)
        except Exception as e:
            print(f"""\n\n=========== {abstract["pmid"]} ==========""")
            print("""\n=========== FAILED! ==========""")
            # print(abstract)
            print(e)
            result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
            result2 = {"results": {}, "resultsinformation": {"error": f"Failed {e}"}}
            output = dict(abstract, **result1, **result2)
            fulldata.append(output)

    with out_file.open("w") as f:
        json.dump(fulldata, f, indent=4)


if __name__ == "__main__":
    main()
