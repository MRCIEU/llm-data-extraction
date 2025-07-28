"""
Aggregate processed LLM results and produce a sample,
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from yiutils.project_utils import find_project_root

# They should look like this in the filesystem
# data_dir / "intermediate" / "llm-results-aggregated" / <MODEL-NAME> / "processed_results.json"
MODEL_RESULTS = ["llama3", "llama3-2", "deepseek-r1-distilled", "o4-mini"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate processed LLM results and produce a sample."
    )
    # ---- --size ----
    parser.add_argument(
        "--size",
        type=int,
        default=20,
        help="Number of samples to produce (default: 20)",
    )
    # ---- --seed ----
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def load_model_results(model_name: str, data_dir: Path):
    print(f"Loading results for model: {model_name}")
    path_to_results = (
        data_dir
        / "intermediate"
        / "llm-results-aggregated"
        / model_name
        / "processed_results.json"
    )
    assert path_to_results.exists(), f"Results file {path_to_results} does not exist."
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


def make_model_results_sample(pmid: str, model_results: dict):
    def _make(model_name: str):
        df = model_results[model_name]["data"]
        item = df[df["pmid"] == pmid].iloc[0].to_dict()
        return item

    res = {_: _make(_) for _ in model_results.keys()}
    return res


def main():
    # ==== init ====
    args = parse_args()
    proj_root = find_project_root(anchor_file="justfile")
    data_dir = proj_root / "data"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    path_to_output_dir = data_dir / "intermediate" / "analysis-sample"
    path_to_output_dir.mkdir(parents=True, exist_ok=True)
    path_to_output = path_to_output_dir / f"sample-{args.seed}-{args.size}.json"

    # ==== load data ====
    # ---- mr data ----
    path_to_processed_mr_pubmed_data = (
        data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
    )
    assert path_to_processed_mr_pubmed_data.exists()
    mr_pubmed_df = pd.read_json(
        path_to_processed_mr_pubmed_data, orient="records"
    ).assign(pmid=lambda x: x["pmid"].astype(str))
    mr_pubmed_id_set = set(mr_pubmed_df["pmid"].unique())

    # ---- model results ----
    # results look like
    # {"data", "id_set"}
    model_results = {_: load_model_results(_, data_dir=data_dir) for _ in MODEL_RESULTS}

    # ---- get commonly available ids ----
    common_id_set = set.intersection(*(v["id_set"] for v in model_results.values()))
    common_id_set = common_id_set.intersection(mr_pubmed_id_set)
    common_id_series = pd.Series(list(common_id_set), name="pmid")

    # ==== make sample ====
    sample_id_series = common_id_series.sample(
        n=args.size, random_state=args.seed
    ).reset_index(drop=True)
    print(f"Sampled {len(sample_id_series)} ids: {sample_id_series.tolist()}")
    output_data = []
    for pmid in sample_id_series:
        pubmed_data = mr_pubmed_df[mr_pubmed_df["pmid"] == pmid].iloc[0].to_dict()
        item = {
            "pubmed_data": pubmed_data,
            "model_results": make_model_results_sample(pmid, model_results),
        }
        output_data.append(item)

    print(f"Save to {path_to_output}")
    with path_to_output.open("w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
