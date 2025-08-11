"""
Aggregate processed OpenAI SP LLM results and produce all results (no sampling),
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from yiutils.project_utils import find_project_root

# OpenAI SP models available
MODEL_RESULTS = ["o4-mini", "gpt-4o", "gpt-4-1", "gpt-5", "gpt-5-mini"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate processed OpenAI SP LLM results and produce all results."
    )
    return parser.parse_args()


def load_model_results(model_name: str, data_dir: Path):
    print(f"Loading results for model: {model_name}")
    results_file = "processed_results_valid.json"
    path_to_results = (
        data_dir
        / "intermediate"
        / "llm-results-aggregated-sp"
        / model_name
        / results_file
    )

    if not path_to_results.exists():
        print(f"Warning: Results file {path_to_results} does not exist for model {model_name}")
        return None

    data = (
        pd.read_json(path_to_results, orient="records")
        .dropna(subset=["metadata", "results"])
        .assign(pmid=lambda x: x["pmid"].astype(str))
    )
    id_set = set(data["pmid"].unique())
    res = {
        "data": data,
        "id_set": id_set,
    }
    return res


def make_model_results_all(pmid: str, model_results: dict):
    def _make(model_name: str):
        if model_results[model_name] is None:
            return None
        df = model_results[model_name]["data"]
        matching_rows = df[df["pmid"] == pmid]
        if matching_rows.empty:
            return None
        item = matching_rows.iloc[0].to_dict()
        return item

    res = {_: _make(_) for _ in model_results.keys()}
    # Filter out None values
    res = {k: v for k, v in res.items() if v is not None}
    return res


def main():
    # ==== init ====
    args = parse_args()
    proj_root = find_project_root(anchor_file="justfile")
    data_dir = proj_root / "data"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    path_to_output_dir = data_dir / "intermediate" / "analysis-sample-sp"
    path_to_output_dir.mkdir(parents=True, exist_ok=True)

    path_to_output = path_to_output_dir / f"all-results.json"

    # ==== load data ====
    # ---- mr data ----
    path_to_processed_mr_pubmed_data = (
        data_dir / "intermediate" / "mr-pubmed-data" / "special-sample.json"
    )
    assert path_to_processed_mr_pubmed_data.exists()
    mr_pubmed_df = pd.read_json(
        path_to_processed_mr_pubmed_data, orient="records"
    ).assign(pmid=lambda x: x["pmid"].astype(str))
    mr_pubmed_id_set = set(mr_pubmed_df["pmid"].unique())

    # ---- model results ----
    # results look like
    # {"data", "id_set"} or None if model doesn't exist
    model_results = {}
    for model in MODEL_RESULTS:
        result = load_model_results(model, data_dir=data_dir)
        model_results[model] = result

    # Filter out models that don't have results
    available_models = {k: v for k, v in model_results.items() if v is not None}

    if not available_models:
        print("No model results found. Exiting.")
        return

    # ---- get commonly available ids ----
    common_id_set = set.intersection(*(v["id_set"] for v in available_models.values()))
    common_id_set = common_id_set.intersection(mr_pubmed_id_set)
    common_id_series = pd.Series(list(common_id_set), name="pmid")

    print(f"Found {len(common_id_series)} common PMIDs across all available models")
    print(f"Available models: {list(available_models.keys())}")

    # ==== process all results ====
    output_data = []
    for pmid in common_id_series:
        pubmed_data = mr_pubmed_df[mr_pubmed_df["pmid"] == pmid].iloc[0].to_dict()
        model_results_for_pmid = make_model_results_all(pmid, model_results)

        # Only include if we have at least one model result
        if model_results_for_pmid:
            item = {
                "pubmed_data": pubmed_data,
                "model_results": model_results_for_pmid,
            }
            output_data.append(item)

    print(f"Processed {len(output_data)} records")
    print(f"Save to {path_to_output}")
    with path_to_output.open("w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
