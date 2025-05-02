"""
Combine pubmed.json and pubmed_new.json into a single mr-data.json file
for LLM data extraction use.
"""

import json
from pathlib import Path

import pandas as pd

from yiutils.project_utils import find_project_root


def main():
    # init
    # {{{
    proj_root = find_project_root()

    data_dir = proj_root / "data"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    path_to_pubmed_json = (
        data_dir / "raw" / "mr-pubmed-abstracts" / "data" / "pubmed.json"
    )
    assert path_to_pubmed_json.exists()
    path_to_pubmed_new_json = (
        data_dir / "raw" / "mr-pubmed-abstracts" / "data" / "pubmed_new.json"
    )
    assert path_to_pubmed_new_json.exists()

    # path to output
    path_to_output = data_dir / "intermediate" / "mr-data" / "mr-data.json"
    path_to_output.parent.mkdir(parents=True, exist_ok=True)
    # }}}

    # load in json files
    with Path(path_to_pubmed_json).open("r") as f:
        pubmed_json = json.load(f)
        pubmed_df = pd.DataFrame(pubmed_json)

    with Path(path_to_pubmed_new_json).open("r") as f:
        pubmed_new_json = json.load(f)
        pubmed_new_df = pd.DataFrame(pubmed_new_json)

    # data process
    # - combine
    # - drop recs with missing abstracts
    # {{{
    mr_data_df = (
        pd.concat([pubmed_df, pubmed_new_df], ignore_index=True)
        .drop_duplicates()
        .dropna(subset=["ab"])
        .reset_index(drop=True)
    )
    mr_data_df.info()
    # }}}

    # write
    print(f"Writing to {path_to_output}")
    mr_data_df.to_json(
        path_to_output,
        orient="records",
        lines=True,
        indent=4,
    )


if __name__ == "__main__":
    main()
