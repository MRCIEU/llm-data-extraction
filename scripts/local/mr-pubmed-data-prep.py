"""
Combine pubmed.json and pubmed_new.json into a single mr-pubmed-data.json file
for LLM data extraction use.
"""

import json
from pathlib import Path

import pandas as pd

from yiutils.project_utils import find_project_root

# config {{{
PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT / "data"

MR_RAW_DATA_DIR = DATA_DIR / "raw" / "mr-pubmed-abstracts" / "data"
RAW_DATA_PATHS = [
    MR_RAW_DATA_DIR / "pubmed.json",
    MR_RAW_DATA_DIR / "pubmed_new.json",
    MR_RAW_DATA_DIR / "pubmed_abstracts_20250502.json",
    MR_RAW_DATA_DIR / "pubmed_abstracts_20250519.json",
]

PATH_TO_OUTPUT = DATA_DIR / "intermediate" / "mr-pubmed-data" / "mr-pubmed-data.json"
# }}}


def main():
    # init
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist."
    PATH_TO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # load raw data {{{
    mr_raw_data = []
    for path in RAW_DATA_PATHS:
        print(f"Loading {path}")
        assert path.exists(), f"Path {path} does not exist."
        with Path(path).open("r") as f:
            raw_data = json.load(f)
            raw_df = pd.DataFrame(raw_data)
            raw_df.info()
            mr_raw_data.append(raw_df)
    # }}}

    # Convert to DataFrame for processing
    mr_data_df = (
        pd.concat(mr_raw_data)
        .dropna(subset=["ab", "pmid"])
        .drop_duplicates(subset=["pmid"])
        .reset_index(drop=True)
    )
    mr_data_df.info()

    # write
    print(f"Writing to {PATH_TO_OUTPUT}")
    mr_data_df.to_json(
        PATH_TO_OUTPUT,
        orient="records",
    )


if __name__ == "__main__":
    main()
