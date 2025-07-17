import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


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
def init(extract_data):
    # Import required modules for data processing
    import json
    import sys
    from pathlib import Path
    from environs import env
    from loguru import logger
    from openai import OpenAI
    from tqdm import tqdm

    # Access functions from the extracted module
    parse_args = extract_data.parse_args
    get_config = extract_data.get_config
    load_schema_data = extract_data.load_schema_data
    setup_openai_client = extract_data.setup_openai_client
    process_abstract = extract_data.process_abstract

    return (json, sys, Path, env, logger, OpenAI, tqdm,
            parse_args, get_config, load_schema_data,
            setup_openai_client, process_abstract)


@app.cell
def mock_args(parse_args):
    # Create mock arguments (equivalent to command line args)
    class MockArgs:
        def __init__(self):
            self.output_dir = "/user/home/ik18445/projects/llm-data-extraction/output"
            self.path_data = "/user/home/ik18445/projects/llm-data-extraction/data/intermediate/mr-pubmed-data/mr-pubmed-data-sample.json"
            self.array_id = 0
            self.array_length = 30
            self.pilot = True  # Enable pilot mode for testing
            # self.model = "o4-mini"  # or "gpt-4o"
            self.model = "gpt-4o"
            self.dry_run = False

    # Create mock args
    mock_args = MockArgs()
    print(f"Mock args created: {vars(mock_args)}")

    return mock_args,


@app.cell
def config(get_config, mock_args):
    # Get configuration and data
    config, pubmed_data = get_config(args=mock_args)
    print(f"Loaded {len(pubmed_data)} abstracts")
    print(f"Config keys: {list(config.keys())}")

    return config, pubmed_data


@app.cell
def client(setup_openai_client, config):
    # Setup OpenAI client
    client = setup_openai_client(api_key=config["openai_api_key"])
    print("OpenAI client initialized")

    return client,


@app.cell
def schema(load_schema_data):
    # Load schema data
    schema_data = load_schema_data()
    print("Schema data loaded")
    print(f"Schema sections: {list(schema_data.keys())}")

    return schema_data,


@app.cell
def _(mock_args, config, json, sys):
    # Handle dry run mode
    if mock_args.dry_run:
        print("Dry run enabled. Printing config and schema_data, then exiting.")
        print("Config:")
        print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))
        print("Would exit here in script mode")
    else:
        print("Proceeding with data processing...")

    return


if __name__ == "__main__":
    app.run()
