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
    process_results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_results)

    return (
        process_results,
        project_root,
    )


@app.cell
def init(process_results, process_o4_mini, project_root):
    from local_funcs import parsers

    load_raw_results = process_results.load_raw_results
    load_schema_files = process_results.load_schema_files
    process_gpt_4o = process_results.process_gpt_4o
    process_o4_mini = process_results.process_o4_mini

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
    return global_config, model_configs, parsers


@app.cell
def o4_mini(
    global_config,
    load_raw_results,
    load_schema_files,
    model_configs,
    parsers,
):
    model_config = model_configs["o4-mini"]
    raw_results_df = load_raw_results(model_config)
    raw_results_df.info()
    meta_schema, results_schema = load_schema_files(global_config)

    results_df = raw_results_df.assign(
        metadata=lambda df: df["completion_metadata"].apply(parsers.parse_json),
        results=lambda df: df["completion_results"].apply(parsers.parse_json),
    )
    return


if __name__ == "__main__":
    app.run()
