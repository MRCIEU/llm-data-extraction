"""
Check openai API interaction
"""

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def init():
    import json

    from environs import env
    from openai import OpenAI

    from yiutils.project_utils import find_project_root

    proj_root = find_project_root(anchor_file="justfile")

    path_to_env = proj_root / ".env"
    assert path_to_env.exists(), f"File not found: {path_to_env}"

    env.read_env(path_to_env)
    openai_api_key = env("PROJECT_OPENAI_API_KEY")

    path_to_data_dir = proj_root / "data"
    assert path_to_data_dir.exists()
    path_to_data = (
        path_to_data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
    )
    assert path_to_data.exists(), f"File not found: {path_to_data}"

    output_dir = proj_root / "output"
    assert output_dir.exists(), f"Output directory not found: {output_dir}"

    return OpenAI, json, openai_api_key, path_to_data


@app.cell
def _(json, path_to_data):
    with path_to_data.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(data[0])
    return


@app.cell
def _(OpenAI, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    return (client,)


@app.cell
def _(client):
    response = client.responses.create(
        model="o4-mini",
        input="What is the capital of France?",
        reasoning={"effort": "low", "summary": "auto"},
    )

    print(response.output)
    return


if __name__ == "__main__":
    app.run()
