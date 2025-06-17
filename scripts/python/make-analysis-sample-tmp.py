"""
Aggregate processed LLM results and produce a sample,
"""

import json

import pandas as pd

from yiutils.project_utils import find_project_root


def main():
    # Set up paths
    proj_root = find_project_root(anchor_file="justfile")
    data_dir = proj_root / "data"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    # Load processed MR PubMed data
    path_to_processed_mr_pubmed_data = (
        data_dir / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
    )
    assert path_to_processed_mr_pubmed_data.exists(), (
        f"Processed MR PubMed data file {path_to_processed_mr_pubmed_data} does not exist."
    )

    with path_to_processed_mr_pubmed_data.open("r") as f:
        mr_pubmed_json = json.load(f)
        mr_pubmed_df = pd.DataFrame(mr_pubmed_json).assign(
            pmid=lambda x: x["pmid"].astype(str)
        )

    # Load DeepSeek R1 results
    path_to_ds_r1_results = (
        data_dir
        / "intermediate"
        / "llm-results-aggregated"
        / "deepseek-r1-distilled"
        / "processed_results.json"
    )
    assert path_to_ds_r1_results.exists(), f"{path_to_ds_r1_results} does not exist."

    df_ds_r1 = (
        pd.read_json(path_to_ds_r1_results, orient="records")
        .dropna(subset=["metadata", "results"])
        .assign(pmid=lambda x: x["pmid"].astype(str))
    )

    # Load Llama3 results
    path_to_llama3_results = (
        data_dir
        / "intermediate"
        / "llm-results-aggregated"
        / "llama3"
        / "processed_results.json"
    )
    assert path_to_llama3_results.exists(), f"{path_to_llama3_results} does not exist."

    df_llama3 = (
        pd.read_json(path_to_llama3_results, orient="records")
        .dropna(subset=["metadata", "results"])
        .assign(pmid=lambda x: x["pmid"].astype(str))
    )

    # Load Llama3-2 results
    path_to_llama3_2_results = (
        data_dir
        / "intermediate"
        / "llm-results-aggregated"
        / "llama3-2"
        / "processed_results.json"
    )
    assert path_to_llama3_2_results.exists(), (
        f"{path_to_llama3_2_results} does not exist."
    )

    df_llama3_2 = (
        pd.read_json(path_to_llama3_2_results, orient="records")
        .dropna(subset=["metadata", "results"])
        .assign(pmid=lambda x: x["pmid"].astype(str))
    )

    # Aggregate analysis
    ids_ds_r1 = set(df_ds_r1["pmid"].unique())
    ids_llama3 = set(df_llama3["pmid"].unique())
    ids_llama3_2 = set(df_llama3_2["pmid"].unique())

    intersection = ids_ds_r1.intersection(ids_llama3).intersection(ids_llama3_2)
    print(f"Number of intersecting IDs: {len(intersection)}")

    # Sample output
    sample_ids = pd.Series(list(intersection)).sample(n=5, random_state=42)
    print(f"Sampled IDs: {sample_ids.tolist()}")

    pubmed_data = mr_pubmed_df[mr_pubmed_df["pmid"].isin(sample_ids)]
    sample_ds_r1 = df_ds_r1[df_ds_r1["pmid"].isin(sample_ids)]
    sample_llama3 = df_llama3[df_llama3["pmid"].isin(sample_ids)]
    sample_llama3_2 = df_llama3_2[df_llama3_2["pmid"].isin(sample_ids)]

    # Create output_data as a list of dicts, one per pmid, removing "pmid", "ab", "title" from model results
    output_data = []
    for pmid in sample_ids:
        pubmed_row = pubmed_data[pubmed_data["pmid"] == pmid].iloc[0].to_dict()
        ds_r1_row = sample_ds_r1[sample_ds_r1["pmid"] == pmid].iloc[0].to_dict()
        llama3_row = sample_llama3[sample_llama3["pmid"] == pmid].iloc[0].to_dict()
        llama3_2_row = (
            sample_llama3_2[sample_llama3_2["pmid"] == pmid].iloc[0].to_dict()
        )
        for d in (ds_r1_row, llama3_row, llama3_2_row):
            for k in ["pmid", "ab", "title"]:
                d.pop(k, None)
        output_data.append(
            {
                "pubmed_data": pubmed_row,
                "result": {
                    "ds_r1": ds_r1_row,
                    "llama3": llama3_row,
                    "llama3_2": llama3_2_row,
                },
            }
        )

    # Write output files
    output_dir = proj_root / "data" / "artifacts" / "sample_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    path_to_output = output_dir / "data_sample.json"
    with path_to_output.open("w") as f:
        json.dump(output_data, f, indent=2)

    path_to_output_base = output_dir / "data_sample_base.json"
    with path_to_output_base.open("w") as f:
        json.dump(output_data[0], f, indent=2)

    print(f"Wrote {path_to_output} and {path_to_output_base}")


if __name__ == "__main__":
    main()
