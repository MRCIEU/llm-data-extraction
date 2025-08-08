"""
Check openai API interaction, gpt-5
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

    return OpenAI, json, openai_api_key, path_to_data


@app.cell
def _(OpenAI, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    return (client,)


@app.cell
def _(client):
    response = client.responses.create(
        model="gpt-5",
        input="What is the capital of France?",
    )

    print(response.output)
    return


if __name__ == "__main__":
    app.run()
