import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import json

    from local_funcs.prompt_funcs import make_message_metadata_new
    from yiutils.project_utils import find_project_root
    return find_project_root, json, make_message_metadata_new


@app.cell
def _(find_project_root):
    proj_root = find_project_root("justfile")
    print(f"Project root: {proj_root}")
    return (proj_root,)


@app.cell
def _(json, proj_root):
    # 2. Load metadata.json and metadata.schema.json from example-data
    example_data_dir = proj_root / "data" / "assets" / "data-schema" / "example-data"
    with open(example_data_dir / "metadata.json") as f:
        metadata_json = json.load(f)
    with open(example_data_dir / "metadata.schema.json") as f:
        metadata_schema = json.load(f)
    print("Loaded metadata.json and metadata.schema.json")
    return metadata_json, metadata_schema


@app.cell
def _(json, make_message_metadata_new, metadata_json, metadata_schema):
    abstract = "FOOBAR"
    json_example = json.dumps(metadata_json, indent=2)
    json_schema = metadata_schema
    messages = make_message_metadata_new(
        abstract=abstract, json_example=json_example, json_schema=json_schema
    )
    print(json.dumps(messages, indent=2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
