"""
Combine pubmed.json and pubmed_new.json into a single mr-pubmed-data.json file
for LLM data extraction use.
"""

import json
from pathlib import Path

import pandas as pd

from yiutils.project_utils import find_project_root

# ==== config ====
PROJ_ROOT = find_project_root()
DATA_DIR = PROJ_ROOT / "data"

MR_RAW_DATA_DIR = DATA_DIR / "raw" / "mr-pubmed-abstracts" / "data"
RAW_DATA_PATHS = [
    MR_RAW_DATA_DIR / "pubmed.json",
    MR_RAW_DATA_DIR / "pubmed_new.json",
    MR_RAW_DATA_DIR / "pubmed_abstracts_20250502.json",
    MR_RAW_DATA_DIR / "pubmed_abstracts_20250519.json",
]

OUTPUT_DIR = DATA_DIR / "intermediate" / "mr-pubmed-data"

# NOTE: this is the length used in llm batch results
SAMPLE_SIZE = 69 * 100 + 100
SAMPLE_LITE_SIZE = 20


def main():
    # ==== init ====
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist."
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==== raw data ====
    mr_raw_data = []
    for path in RAW_DATA_PATHS:
        print(f"Loading {path}")
        assert path.exists(), f"Path {path} does not exist."
        with Path(path).open("r") as f:
            raw_data = json.load(f)
            raw_df = pd.DataFrame(raw_data)
            raw_df.info()
            mr_raw_data.append(raw_df)

    # ==== full data prep ====

    # Convert to DataFrame for processing
    mr_data_df = (
        pd.concat(mr_raw_data)
        .dropna(subset=["ab", "pmid"])
        .drop_duplicates(subset=["pmid"])
        .reset_index(drop=True)
    )
    mr_data_df.info()

    # write
    output_path = OUTPUT_DIR / "mr-pubmed-data.json"
    print(f"Writing to {output_path}")
    mr_data_df.to_json(output_path, orient="records")

    # ==== sample ====
    mr_data_sample = mr_data_df[:SAMPLE_SIZE]

    output_path = OUTPUT_DIR / "mr-pubmed-data-sample.json"
    print(f"Writing sample to {output_path}")
    mr_data_sample.to_json(output_path, orient="records")

    # ==== sample lite ====
    mr_data_sample_lite = mr_data_df[:SAMPLE_LITE_SIZE]

    output_path = OUTPUT_DIR / "mr-pubmed-data-sample-lite.json"
    print(f"Writing sample to {output_path}")
    mr_data_sample_lite.to_json(output_path, orient="records")


if __name__ == "__main__":
    main()
