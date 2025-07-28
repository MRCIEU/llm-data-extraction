"""
Check generated prompt message
"""

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def init():
    import json
    from pprint import pprint

    from local_funcs import prompt_funcs
    from local_funcs.schema_funcs import load_schema_data
    from yiutils.project_utils import find_project_root

    return find_project_root, json, load_schema_data, pprint, prompt_funcs


@app.cell
def load_abstract(find_project_root, json, load_schema_data):
    project_root = find_project_root("justfile")
    data_dir = project_root / "data"
    path_to_pubmed = (
        data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data-sample.json"
    )
    assert path_to_pubmed.exists(), f"File not found: {path_to_pubmed}"

    with path_to_pubmed.open("r") as f:
        pubmed_data = json.load(f)

    article_data = pubmed_data[0]
    print(article_data.keys())

    schema_data = load_schema_data()
    return article_data, project_root, schema_data


@app.cell
def prompt_metadata(
    article_data,
    json,
    pprint,
    project_root,
    prompt_funcs,
    schema_data,
):
    input_prompt_metadata = prompt_funcs.make_message_metadata_new(
        abstract=article_data["ab"],
        json_example=schema_data["metadata"]["example"],
        json_schema=schema_data["metadata"]["schema"],
    )
    pprint(input_prompt_metadata)

    with (project_root / "output" / "input_prompt_metadata.json").open("w") as _:
        json.dump(input_prompt_metadata, _, indent=2)
    return


@app.cell
def prompt_results(
    article_data,
    json,
    pprint,
    project_root,
    prompt_funcs,
    schema_data,
):
    input_prompt_results = prompt_funcs.make_message_results_new(
        abstract=article_data["ab"],
        json_example=schema_data["results"]["example"],
        json_schema=schema_data["results"]["schema"],
    )
    pprint(input_prompt_results)

    with (project_root / "output" / "input_prompt_results.json").open("w") as _:
        json.dump(input_prompt_results, _, indent=2)
    return


if __name__ == "__main__":
    app.run()
