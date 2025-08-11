#!/usr/bin/env python3
"""
Prepare special sample by filtering mr-pubmed-data.json to only include PMIDs from sample-42-100.json.

This script:
1. Reads the analysis sample file to extract PMIDs
2. Loads the full mr-pubmed-data.json file
3. Filters the mr-pubmed-data to only include papers with PMIDs present in the sample
4. Outputs the filtered data to special-sample.json
"""

import json
import sys
from pathlib import Path


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def save_json(data, file_path):
    """Save data to JSON file."""
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(data)} entries to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        sys.exit(1)


def extract_pmids_from_sample(sample_data):
    """Extract PMIDs from sample-42-100.json."""
    pmids = set()
    for entry in sample_data:
        if "pubmed_data" in entry and "pmid" in entry["pubmed_data"]:
            pmids.add(entry["pubmed_data"]["pmid"])
    return pmids


def filter_mr_pubmed_data(mr_pubmed_data, target_pmids):
    """Filter mr-pubmed-data to only include entries with PMIDs in target set."""
    filtered_data = []
    found_pmids = set()

    for entry in mr_pubmed_data:
        if "pmid" in entry and entry["pmid"] in target_pmids:
            filtered_data.append(entry)
            found_pmids.add(entry["pmid"])

    return filtered_data, found_pmids


def main():
    # Define file paths
    sample_file = "./data/intermediate/analysis-sample/sample-42-100.json"
    mr_pubmed_file = "./data/intermediate/mr-pubmed-data/mr-pubmed-data.json"
    output_file = "./data/intermediate/mr-pubmed-data/special-sample.json"

    print("Loading analysis sample...")
    sample_data = load_json(sample_file)
    print(f"Loaded {len(sample_data)} entries from sample file")

    print("Extracting PMIDs from sample...")
    target_pmids = extract_pmids_from_sample(sample_data)
    print(f"Found {len(target_pmids)} unique PMIDs in sample")

    print("Loading mr-pubmed-data...")
    mr_pubmed_data = load_json(mr_pubmed_file)
    print(f"Loaded {len(mr_pubmed_data)} entries from mr-pubmed-data")

    print("Filtering mr-pubmed-data...")
    filtered_data, found_pmids = filter_mr_pubmed_data(mr_pubmed_data, target_pmids)

    # Report matching statistics
    print("\nMatching statistics:")
    print(f"  Target PMIDs from sample: {len(target_pmids)}")
    print(f"  PMIDs found in mr-pubmed-data: {len(found_pmids)}")
    print(f"  PMIDs missing from mr-pubmed-data: {len(target_pmids - found_pmids)}")

    if target_pmids - found_pmids:
        print(
            f"  Missing PMIDs: {sorted(target_pmids - found_pmids)[:10]}{'...' if len(target_pmids - found_pmids) > 10 else ''}"
        )

    print("\nSaving filtered data...")
    save_json(filtered_data, output_file)

    print(f"\nDone! Created special sample with {len(filtered_data)} entries.")


if __name__ == "__main__":
    main()
