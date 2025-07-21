import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def module_init():
    # Import the extract_data module
    from yiutils.project_utils import find_project_root

    project_root = find_project_root("justfile")
    path_to_script = project_root / "scripts" / "python" / "extract-data-openai.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_data", str(path_to_script))
    extract_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_data)

    return extract_data, project_root


@app.cell
def _(mo):
    mo.md(r"""# Init""")
    return


@app.cell
def init(extract_data):
    # Import required modules for data processing
    import json

    # Access functions from the extracted module
    get_config = extract_data.get_config
    load_schema_data = extract_data.load_schema_data
    setup_openai_client = extract_data.setup_openai_client

    return get_config, json, load_schema_data, setup_openai_client


@app.cell
def mock_args(extract_data, project_root):
    # Create mock arguments (equivalent to command line args)
    class MockArgs:
        def __init__(self):
            self.output_dir = project_root / "output"
            self.path_data = extract_data.PATH_DATA
            self.array_id = 0
            self.array_length = 30
            self.pilot = True  # Enable pilot mode for testing
            # self.model = "o4-mini"  # or "gpt-4o"
            self.model = "gpt-4o"
            self.dry_run = False

    # Create mock args
    mock_args = MockArgs()
    print(f"Mock args created: {vars(mock_args)}")

    return (mock_args,)


@app.cell
def config(get_config, mock_args):
    # Get configuration and data
    config, pubmed_data = get_config(args=mock_args)
    print(f"Loaded {len(pubmed_data)} abstracts")
    print(f"Config keys: {list(config.keys())}")

    return config, pubmed_data


@app.cell
def client(config, setup_openai_client):
    # Setup OpenAI client
    client = setup_openai_client(api_key=config["openai_api_key"])
    print("OpenAI client initialized")

    return


@app.cell
def schema(load_schema_data):
    # Load schema data
    schema_data = load_schema_data()
    print("Schema data loaded")
    print(f"Schema sections: {list(schema_data.keys())}")

    return (schema_data,)


@app.cell
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Process abstract""")
    return


@app.cell
def process_abstract(pubmed_data):
    article_data = pubmed_data[0]
    print(article_data.keys())

    # process_abstract = extract_data.process_abstract
    # output = process_abstract(
    #     article_data=article_data,
    #     schema_data=schema_data,
    #     client=client,
    #     model_config=config["model_config"],
    # )

    # print(output, article_data)
    return (article_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Process abstract logics""")
    return


@app.cell
def _(mo):
    mo.md(r"""## prompts""")
    return


@app.cell
def prompt_metadata(article_data, json, project_root, schema_data):
    from pprint import pprint

    from local_funcs import prompt_funcs

    # model_config = config["model_config"]

    input_prompt_metadata = prompt_funcs.make_message_metadata_new(
        abstract=article_data["ab"],
        json_example=schema_data["metadata"]["example"],
        json_schema=schema_data["metadata"]["schema"],
    )
    pprint(input_prompt_metadata)

    with (project_root / "output" / "input_prompt_metadata.json").open("w") as _:
        json.dump(input_prompt_metadata, _, indent=2)
    return pprint, prompt_funcs


@app.cell
def _(article_data, json, pprint, project_root, prompt_funcs, schema_data):
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
