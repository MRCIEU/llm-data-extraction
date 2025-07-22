"""
Check post-batch-processing logics
"""
import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def module_init():
    # Import the extract_data module
    from yiutils.project_utils import find_project_root

    project_root = find_project_root("justfile")
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

    return proc, project_root


@app.cell
def init(proc, project_root):
    import jsonschema
    from local_funcs import parsers

    load_raw_results = proc.load_raw_results
    load_schema_files = proc.load_schema_files
    process_gpt_4o = proc.process_gpt_4o
    process_o4_mini = proc.process_o4_mini
    process_metadata = proc.process_metadata
    process_results = proc.process_results

    data_dir = project_root / "data"
    agg_data_dir = data_dir / "intermediate" / "llm-results-aggregated"
    assert agg_data_dir.exists()

    global_config = {
        "schema": {
            "metadata": data_dir
            / "assets"
            / "data-schema"
            / "processed_results"
            / "metadata.schema.json",
            "results": data_dir
            / "assets"
            / "data-schema"
            / "processed_results"
            / "results.schema.json",
        }
    }

    model_configs = {
        "o4-mini": {
            "name": "o4-mini",
            "data_dir": agg_data_dir / "o4-mini",
            "error_log": agg_data_dir / "logs" / "o4-mini_schema_validation_errors.log",
            "func": process_o4_mini,
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "data_dir": agg_data_dir / "gpt-4o",
            "error_log": agg_data_dir / "logs" / "gpt-4o_schema_validation_errors.log",
            "func": process_gpt_4o,
        },
    }
    for k, v in model_configs.items():
        assert v["data_dir"].exists(), (
            f"Input path for {k} does not exist: {v['data_dir']}"
        )
    return (
        global_config,
        jsonschema,
        load_raw_results,
        load_schema_files,
        model_configs,
        parsers,
        process_metadata,
        process_results,
    )


@app.cell
def o4_mini(
    global_config,
    load_raw_results,
    load_schema_files,
    model_configs,
    parsers,
    process_metadata,
    process_results,
):
    model_config = model_configs["o4-mini"]
    raw_results_df = load_raw_results(model_config)
    raw_results_df.info()
    meta_schema, results_schema = load_schema_files(global_config)

    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )

    results_df = results_df[["pmid", "metadata", "results"]]
    results_df = results_df.dropna(subset=["metadata", "results"]).assign(
        metadata=lambda df: df["metadata"].apply(process_metadata),
        results=lambda df: df["results"].apply(process_results),
    )
    return meta_schema, results_df, results_schema


@app.cell
def validate(meta_schema, proc, results_df):
    metadata = results_df["metadata"][0]
    validate_item_with_schema = proc.validate_item_with_schema
    validate_item_with_schema(
        metadata,
        meta_schema,
        "/dev/null"
        )
    return (validate_item_with_schema,)


@app.cell
def _(meta_schema, results_df, validate_item_with_schema):
    metadata_s = results_df["metadata"]
    valid_s = metadata_s.apply(validate_item_with_schema, schema=meta_schema, log_file="/dev/null")
    return


@app.cell
def _(jsonschema, results_df, results_schema, validate_item_with_schema):
    results = results_df["results"][0]
    _ = validate_item_with_schema(
        results,
        results_schema,
        "/dev/null"
        )
    print(_)

    jsonschema.validate(instance=results, schema=results_schema)
    return


@app.cell
def _(results_df, results_schema, validate_item_with_schema):
    results_s = results_df["results"]
    valid_results_s = results_s.apply(validate_item_with_schema, schema=results_schema, log_file="/dev/null")
    return


if __name__ == "__main__":
    app.run()
