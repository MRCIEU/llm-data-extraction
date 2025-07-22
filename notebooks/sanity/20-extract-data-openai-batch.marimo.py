"""
Lite batch extraction by openai models
"""

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def module_init():
    from yiutils.project_utils import find_project_root

    project_root = find_project_root("justfile")
    path_to_script = project_root / "scripts" / "python" / "extract-data-openai.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_data", str(path_to_script))
    extract_data = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(extract_data)  # type: ignore

    return extract_data, project_root


@app.cell
def init(extract_data):
    # Import required modules for data processing
    import json
    from tqdm import tqdm

    # Access functions from the extracted module
    get_config = extract_data.get_config
    load_schema_data = extract_data.load_schema_data
    setup_openai_client = extract_data.setup_openai_client
    process_abstract = extract_data.process_abstract

    return (
        get_config,
        json,
        load_schema_data,
        process_abstract,
        setup_openai_client,
        tqdm,
    )


@app.cell
def config(extract_data, get_config, project_root):
    class MockArgs:
        def __init__(self):
            self.output_dir = project_root / "output"
            self.path_data = extract_data.PATH_DATA
            self.array_id = 0
            self.array_length = 30
            self.pilot = True  # Enable pilot mode for testing
            self.model = "o4-mini"  # or "gpt-4o"
            # self.model = "gpt-4o"
            self.dry_run = False

    # Create mock args
    mock_args = MockArgs()
    print(f"Mock args created: {vars(mock_args)}")
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

    return (client,)


@app.cell
def schema(load_schema_data):
    # Load schema data
    schema_data = load_schema_data()
    print("Schema data loaded")
    print(f"Schema sections: {list(schema_data.keys())}")

    return (schema_data,)


@app.cell
def process_abstract(
    client,
    config,
    json,
    process_abstract,
    project_root,
    pubmed_data,
    schema_data,
    tqdm,
):
    fulldata = []
    for article_data in tqdm(pubmed_data):
        output = process_abstract(
            article_data=article_data,
            schema_data=schema_data,
            client=client,
            model_config=config["model_config"],
        )
        fulldata.append(output)

    path_output = project_root / "output" / "openai_batch.json"
    with path_output.open("w") as _:
        json.dump(fulldata, _, indent=2)

    return


if __name__ == "__main__":
    app.run()
