"""
Validate output with schema passing, single case
"""

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def init():
    import json
    import jsonschema
    import pandas as pd

    from local_funcs import parsers
    from yiutils.project_utils import find_project_root

    project_root = find_project_root("justfile")
    return json, jsonschema, parsers, pd, project_root


@app.cell
def module_init(project_root):
    path_to_script = (
        project_root / "scripts" / "python" / "process-llm-aggregated-results.py"
    )
    assert path_to_script.exists()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "process_results", str(path_to_script)
    )
    proc = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(proc)  # type: ignore

    process_metadata = proc.process_metadata  # type: ignore
    process_results = proc.process_results  # type: ignore
    return process_metadata, process_results


@app.cell
def setup_data(pd, project_root):
    result_path = project_root / "output" / "openai_batch.json"
    raw_results_df = pd.read_json(result_path, orient="records")
    return (raw_results_df,)


@app.cell
def process_data(parsers, process_metadata, process_results, raw_results_df):
    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )
    results_df = results_df[["pmid", "metadata", "results"]]
    results_df = results_df.dropna(subset=["metadata", "results"]).assign(
        metadata=lambda df: df["metadata"].apply(process_metadata),
        results=lambda df: df["results"].apply(process_results),
    )
    results_df.info()
    return (results_df,)


@app.cell
def load_schema(json, project_root):
    data_dir = project_root / "data"
    path_to_schema_metadata = (
        data_dir
        / "assets"
        / "data-schema"
        / "processed_results"
        / "metadata.schema.json"
    )
    with open(path_to_schema_metadata) as _:
        meta_schema = json.load(_)

    path_to_schema_results = (
        data_dir
        / "assets"
        / "data-schema"
        / "processed_results"
        / "results.schema.json"
    )
    with open(path_to_schema_results) as _:
        results_schema = json.load(_)

    return meta_schema, results_schema


@app.cell
def validate_metadata(jsonschema, meta_schema, results_df):
    metadata_failed_attempts = 0
    for _ in results_df["metadata"]:
        try:
            jsonschema.validate(instance=_, schema=meta_schema)
        except jsonschema.ValidationError as e:
            print(e)
            metadata_failed_attempts = metadata_failed_attempts + 1
    print(f"{metadata_failed_attempts=}")
    return


@app.cell
def validate_results(jsonschema, results_df, results_schema):
    results_failed_attempts = 0
    for _ in results_df["results"]:
        try:
            jsonschema.validate(instance=_, schema=results_schema)
        except jsonschema.ValidationError as e:
            print(e)
            results_failed_attempts = results_failed_attempts + 1
    print(f"{results_failed_attempts=}")
    return


if __name__ == "__main__":
    app.run()
