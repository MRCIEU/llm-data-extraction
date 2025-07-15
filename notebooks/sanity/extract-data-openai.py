import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Extract mr pubmed abstracts data using OpenAI models

    NOTE: this should be part of a slurm job submission

    Logics
    - Processes abstracts in batches based on SLURM array task ID.
    """
    )
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from environs import env
    from openai import OpenAI
    from tqdm import tqdm

    from local_funcs import openai_funcs, prompt_funcs
    from yiutils.chunking import calculate_chunk_start_end
    from yiutils.project_utils import find_project_root

    return (
        OpenAI,
        Path,
        calculate_chunk_start_end,
        env,
        find_project_root,
        json,
        openai_funcs,
        prompt_funcs,
        tqdm,
    )


@app.cell
def _(find_project_root, openai_funcs):
    # ==== params ====
    PROJECT_ROOT = find_project_root("justfile")
    DATA_DIR = PROJECT_ROOT / "data"
    PATH_DATA = (
        DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data-sample.json"
    )
    PATH_SCHEMA_DIR = DATA_DIR / "assets" / "data-schema" / "example-data"

    MODEL_CONFIGS = {
        "o4-mini": {
            "model_id": "o4-mini",
            "chat_func": openai_funcs.get_o4_mini_result,
        },
        "gpt-4o": {"model_id": "gpt-4o", "chat_func": openai_funcs.get_gpt_4o_result},
    }
    PILOT_NUM_DOCS = 20
    ARRAY_LENGTH = 30
    return (
        ARRAY_LENGTH,
        MODEL_CONFIGS,
        PATH_DATA,
        PATH_SCHEMA_DIR,
        PILOT_NUM_DOCS,
        PROJECT_ROOT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Argument Parsing (for notebook, set variables directly)""")
    return


@app.cell
def params(PATH_DATA, PROJECT_ROOT):
    # Set these variables directly in the notebook
    output_dir = PROJECT_ROOT / "output"
    path_data = PATH_DATA
    array_id = 0
    pilot = True
    model = "o4-mini"  # or "gpt-4o"
    return array_id, model, output_dir, path_data, pilot


@app.cell
def config(
    ARRAY_LENGTH,
    MODEL_CONFIGS,
    PILOT_NUM_DOCS,
    PROJECT_ROOT,
    Path,
    array_id,
    calculate_chunk_start_end,
    env,
    json,
    model,
    output_dir,
    path_data,
    pilot,
):
    env.read_env(path=PROJECT_ROOT / ".env")
    array_task_id = array_id
    openai_api_key = env("PROJECT_OPENAI_API_KEY")
    print(f"{openai_api_key[:10]=}")
    pilot_num_docs = PILOT_NUM_DOCS
    array_length = ARRAY_LENGTH

    path_to_pubmed = Path(path_data)
    assert path_data is not None and path_to_pubmed.exists(), "pubmed data not found"

    print(f"Loading data from {path_to_pubmed}")
    with path_to_pubmed.open("r") as _f:
        pubmed_data = json.load(_f)
    data_length = len(pubmed_data)

    startpoint, endpoint = calculate_chunk_start_end(
        chunk_id=array_task_id,
        num_chunks=array_length,
        data_length=data_length,
        pilot_num_docs=pilot_num_docs,
        pilot=pilot,
        verbose=True,
    )
    if startpoint is None or endpoint is None:
        print(f"WARNING: startpoint {startpoint} endpoint {endpoint}")
        raise RuntimeError("Invalid startpoint/endpoint")

    pubmed_data = pubmed_data[startpoint:endpoint]
    model_config_name = model
    assert (
        model_config_name is not None and model_config_name in MODEL_CONFIGS.keys()
    ), "--model must not be empty and must be one of the configs"

    model_config = MODEL_CONFIGS[model_config_name]
    output_dir_1 = Path(output_dir) / model_config_name
    output_dir_1.mkdir(parents=True, exist_ok=True)
    pilot_suffix = "_pilot" if pilot else ""
    out_file = (
        output_dir_1 / f"mr_extract_openai_array_{array_task_id}{pilot_suffix}.json"
    )
    print(f"Loaded {len(pubmed_data)} abstracts from {path_to_pubmed}")
    return model_config, openai_api_key, out_file, pubmed_data


@app.cell
def _(OpenAI, openai_api_key):
    def setup_openai_client(api_key):
        client = OpenAI(api_key=api_key)
        print("Loaded OpenAI client")
        return client

    client = setup_openai_client(api_key=openai_api_key)
    return (client,)


@app.cell
def _(PATH_SCHEMA_DIR, json):
    def load_schema_data():
        schema_config = {
            "metadata": {
                "example": PATH_SCHEMA_DIR / "metadata.json",
                "schema": PATH_SCHEMA_DIR / "metadata.schema.json",
            },
            "results": {
                "example": PATH_SCHEMA_DIR / "results.json",
                "schema": PATH_SCHEMA_DIR / "results.schema.json",
            },
        }
        missing_files = []
        schema_data = {}
        for section_name, section in schema_config.items():
            schema_data[section_name] = {}
            for key, path in section.items():
                if not path.exists():
                    missing_files.append(str(path))
                    schema_data[section_name][key] = None
                else:
                    with path.open("r") as _f:
                        try:
                            schema_data[section_name][key] = json.load(_f)
                        except Exception as e:
                            print(f"ERROR loading {path}: {e}")
                            schema_data[section_name][key] = None
        if missing_files:
            print(f"WARNING: The following schema files do not exist: {missing_files}")
        else:
            print("All schema files found.")
        return schema_data

    schema_data = load_schema_data()
    return (schema_data,)


@app.cell
def process_abstract(prompt_funcs):
    def process_abstract(article_data, schema_data, client, model_config):
        try:
            chat_func = model_config["chat_func"]
            input_prompt_metadata = prompt_funcs.make_message_metadata_new(
                abstract=article_data["ab"],
                json_example=schema_data["metadata"]["example"],
                json_schema=schema_data["metadata"]["schema"],
            )
            input_prompt_results = prompt_funcs.make_message_metadata_new(
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
            print(f"\n\n=========== {article_data.get('pmid', 'NO PMID')} ===========")
            print("\n=========== FAILED! ===========")
            print(e)
            result1 = {"metadata": {}, "metainformation": {"error": f"Failed {e}"}}
            output = dict(article_data, **result1)
            print(f"Output: {output}")
            return output

    return (process_abstract,)


@app.cell
def single_item_debug(mo):
    mo.md(r"""# Single item""")
    return


@app.cell
def _(client, model_config, prompt_funcs, pubmed_data, schema_data):
    def _single_item_debug():
        chat_func = model_config["chat_func"]
        article_data = pubmed_data[0]
        input_prompt_metadata = prompt_funcs.make_message_metadata_new(
            abstract=article_data["ab"],
            json_example=schema_data["metadata"]["example"],
            json_schema=schema_data["metadata"]["schema"],
        )
        completion_metadata = chat_func(client, input_prompt_metadata)
        print(completion_metadata)

    _single_item_debug()
    return


@app.cell
def _(mo):
    mo.md(r"""# Full batch""")
    return


@app.cell
def full_batch(
    client,
    model_config,
    process_abstract,
    pubmed_data,
    schema_data,
    tqdm,
):
    # ==== process abstracts ====
    fulldata = []
    for article_data in tqdm(pubmed_data):
        output = process_abstract(
            article_data=article_data,
            schema_data=schema_data,
            client=client,
            model_config=model_config,
        )
        fulldata.append(output)
    return (fulldata,)


@app.cell
def _(fulldata, json, out_file):
    print(f"Wrote results to {out_file}")
    with out_file.open("w") as _f:
        json.dump(fulldata, _f, indent=4)
    return


if __name__ == "__main__":
    app.run()
