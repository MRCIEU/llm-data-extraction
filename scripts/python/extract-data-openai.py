#!/usr/bin/env python3

import json
from pprint import pprint
from pathlib import Path

from environs import env
from openai import OpenAI
from tqdm import tqdm

from local_funcs import prompt_templates, openai_funcs
from yiutils.project_utils import find_project_root


def generate_message(abstract):
    messages = [
        {
            "role": "system",
            "content": "You are a data scientist responsible for extracting accurate information from research papers. You answer each question with a single JSON string.",
        },
        {
            "role": "user",
            "content": f"""
                This is an abstract from a Mendelian randomization study.
                    "{abstract["ab"]}"   """,
        },
        prompt_templates.metadataexample,
        prompt_templates.metadataprompt,
    ]
    return messages


def main():
    proj_root = find_project_root(anchor_file="justfile")

    path_to_env = proj_root / ".env"
    if not path_to_env.exists():
        raise FileNotFoundError(f"File not found: {path_to_env}")

    env.read_env(path_to_env)
    openai_api_key = env("OPENAI_API_KEY")

    path_to_data_dir = proj_root / "data"
    if not path_to_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {path_to_data_dir}")

    path_to_data = (
        path_to_data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
    )
    if not path_to_data.exists():
        raise FileNotFoundError(f"File not found: {path_to_data}")

    output_dir = proj_root / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    with path_to_data.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} abstracts.")

    client = OpenAI(api_key=openai_api_key)

    # Example: print the first abstract
    print("First abstract:")
    pprint(data[0])

    # Example: print generated message for first abstract
    print("\nGenerated message for first abstract:")
    pprint(generate_message(data[0]))

    # Single case: o4-mini
    print("\nRequesting o4-mini for first abstract...")
    response_o4 = client.responses.create(
        model="o4-mini",
        input=generate_message(data[0]),
        reasoning={"effort": "medium"},
    )
    print("\no4-mini output_text:")
    print(response_o4.output_text)
    print("\no4-mini reasoning:")
    print(response_o4.reasoning)

    # Single case: gpt-4o
    print("\nRequesting gpt-4o for first abstract...")
    response_gpt4o = client.responses.create(
        model="gpt-4o",
        input=generate_message(data[0]),
    )
    print("\ngpt-4o output_text:")
    print(response_gpt4o.output_text)

    # Batch processing (first 10 abstracts)
    data_batch = data[:10]
    print(f"\nProcessing batch of {len(data_batch)} abstracts with o4-mini...")
    result_batch_o4_mini = [
        openai_funcs.get_o4_mini_result(
            client=client,
            abstract=abstract,
        )
        for abstract in tqdm(data_batch, desc="Processing batch")
    ]

    print(f"\nProcessing batch of {len(data_batch)} abstracts with gpt-4o...")
    result_gpt_4o = [
        openai_funcs.get_gpt_4o_result(
            client=client,
            abstract=abstract,
        )
        for abstract in tqdm(data_batch, desc="Processing batch")
    ]

    # Print first result for inspection
    print("\nFirst o4-mini batch result:")
    print(result_batch_o4_mini[0])
    print("\nFirst gpt-4o batch result:")
    print(result_gpt_4o[0])

    # Write results to output files
    output_path = output_dir / "sample_results_o4_mini.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_batch_o4_mini, f, indent=2, ensure_ascii=False)
    print(f"\nWrote o4-mini batch results to {output_path}")

    output_path = output_dir / "sample_results_gpt_4o.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_gpt_4o, f, indent=2, ensure_ascii=False)
    print(f"Wrote gpt-4o batch results to {output_path}")


if __name__ == "__main__":
    main()
