"""
Validate output with schema passing
"""
import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Init""")
    return


@app.cell
def init():
    import json
    from pprint import pprint

    import jsonschema

    from local_funcs import parsers
    from yiutils.project_utils import find_project_root

    project_root = find_project_root("justfile")
    return json, jsonschema, parsers, pprint, project_root


@app.cell
def _(project_root):
    path_to_script = (
        project_root / "scripts" / "python" / "process-llm-aggregated-results.py"
    )
    assert path_to_script.exists()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "process_results", str(path_to_script)
    )
    proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc)

    process_metadata = proc.process_metadata
    process_results = proc.process_results
    return process_metadata, process_results


@app.cell
def setup_data(json, pprint, project_root):
    result_path = project_root / "output" / "openai_output.json"
    with result_path.open() as _:
        result = json.load(_)
    pprint(result)
    return (result,)


@app.cell
def parse_extracted_data(parsers, pprint, process_metadata, result):
    metadata_raw = result["completion_metadata"]
    metadata = parsers.parse_json(metadata_raw)
    metadata = process_metadata(metadata)
    pprint(metadata)
    return (metadata,)


@app.cell
def _(parsers, pprint, process_results, result):
    results_raw = result["completion_results"]
    results = parsers.parse_json(results_raw)
    results = process_results(results)
    pprint(results)
    return (results,)


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
def validate_metadata(jsonschema, meta_schema, metadata):
    jsonschema.validate(instance=metadata, schema=meta_schema)
    return


@app.cell
def validate_results(jsonschema, results, results_schema):
    jsonschema.validate(instance=results, schema=results_schema)
    return


if __name__ == "__main__":
    app.run()
