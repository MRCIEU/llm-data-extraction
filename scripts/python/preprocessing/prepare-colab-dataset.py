#!/usr/bin/env python3
"""
Prepare dataset for Gemma3 270M fine-tuning on Google Colab.

This script processes existing OpenAI extraction results and creates
a single consolidated dataset file that can be easily uploaded to
Google Colab for fine-tuning.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_openai_results(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all OpenAI extraction results from the data directory."""
    logging.info(f"Loading OpenAI results from {data_dir}")

    all_results = []
    processed_files = []

    # Look for processed results files
    result_patterns = ["**/*results*.json", "**/*openai*.json", "**/*extract*.json"]

    for pattern in result_patterns:
        for result_file in data_dir.glob(pattern):
            if result_file.name.startswith("."):
                continue

            try:
                logging.info(f"Processing {result_file}")
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle different data structures
                if isinstance(data, list):
                    for item in data:
                        if validate_extraction_item(item):
                            all_results.append(item)
                elif isinstance(data, dict):
                    if validate_extraction_item(data):
                        all_results.append(data)

                processed_files.append(str(result_file))

            except Exception as e:
                logging.warning(f"Failed to load {result_file}: {e}")
                continue

    logging.info(
        f"Loaded {len(all_results)} extraction results from {len(processed_files)} files"
    )
    return all_results


def validate_extraction_item(item: Dict[str, Any]) -> bool:
    """Validate that an extraction item has required fields."""
    required_fields = ["abstract"]

    # Check for basic required fields
    if not all(field in item for field in required_fields):
        return False

    # Check for at least one extraction type
    has_metadata = "extracted_metadata" in item or "metadata" in item
    has_results = "extracted_results" in item or "results" in item

    return has_metadata or has_results


def normalize_extraction_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize extraction item to consistent format."""
    normalized = {
        "abstract": item.get("abstract", ""),
        "pmid": item.get("pmid", ""),
        "doi": item.get("doi", ""),
        "title": item.get("title", ""),
    }

    # Normalize metadata
    metadata = item.get("extracted_metadata") or item.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}
    normalized["extracted_metadata"] = metadata

    # Normalize results
    results = item.get("extracted_results") or item.get("results") or {}
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except:
            results = {}
    normalized["extracted_results"] = results

    return normalized


def create_training_examples(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Create training examples in instruction-response format."""
    examples = []

    for item in data:
        abstract = item["abstract"]

        # Skip if abstract is too short
        if len(abstract.strip()) < 100:
            continue

        # Create metadata extraction example
        metadata = item.get("extracted_metadata", {})
        if metadata:
            metadata_instruction = create_metadata_instruction(abstract)
            metadata_response = json.dumps(metadata, ensure_ascii=False, indent=None)

            examples.append(
                {
                    "task_type": "metadata",
                    "instruction": metadata_instruction,
                    "response": metadata_response,
                    "abstract": abstract[:200] + "...",  # For reference
                    "source_pmid": item.get("pmid", ""),
                }
            )

        # Create results extraction example
        results = item.get("extracted_results", {})
        if results:
            results_instruction = create_results_instruction(abstract)
            results_response = json.dumps(results, ensure_ascii=False, indent=None)

            examples.append(
                {
                    "task_type": "results",
                    "instruction": results_instruction,
                    "response": results_response,
                    "abstract": abstract[:200] + "...",  # For reference
                    "source_pmid": item.get("pmid", ""),
                }
            )

    return examples


def create_metadata_instruction(abstract: str) -> str:
    """Create instruction for metadata extraction."""
    return (
        "Extract metadata from the following Mendelian randomization abstract. "
        "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
        f"Abstract: {abstract}\n\n"
        "Metadata JSON:"
    )


def create_results_instruction(abstract: str) -> str:
    """Create instruction for results extraction."""
    return (
        "Extract MR results from the following Mendelian randomization abstract. "
        "Return valid JSON with fields: exposures, outcomes, mr_methods, effect_estimates, "
        "pvalues, confidence_intervals, sample_sizes, populations, study_design.\n\n"
        f"Abstract: {abstract}\n\n"
        "Results JSON:"
    )


def format_for_gemma(examples: List[Dict[str, str]]) -> List[str]:
    """Format examples for Gemma conversation format."""
    formatted_examples = []

    for example in examples:
        # Use Gemma conversation format
        formatted = (
            f"<bos><start_of_turn>user\n"
            f"{example['instruction']}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{example['response']}"
            f"<eos>"
        )
        formatted_examples.append(formatted)

    return formatted_examples


def create_dataset_splits(
    examples: List[Dict[str, str]], train_ratio: float = 0.85
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split examples into train and validation sets."""
    import random

    random.seed(42)

    # Shuffle examples
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_examples = shuffled[:split_idx]
    val_examples = shuffled[split_idx:]

    return train_examples, val_examples


def save_dataset(
    examples: List[Dict[str, str]],
    output_file: Path,
    dataset_name: str = "mr_extraction_dataset",
):
    """Save dataset in multiple formats for Colab usage."""
    logging.info(f"Saving dataset to {output_file}")

    # Create formatted prompts for training
    formatted_prompts = format_for_gemma(examples)

    # Create dataset structure
    dataset = {
        "dataset_info": {
            "name": dataset_name,
            "description": "Mendelian randomization abstract extraction dataset",
            "created_at": datetime.now().isoformat(),
            "num_examples": len(examples),
            "num_metadata_examples": len(
                [e for e in examples if e["task_type"] == "metadata"]
            ),
            "num_results_examples": len(
                [e for e in examples if e["task_type"] == "results"]
            ),
        },
        "examples": examples,
        "formatted_prompts": formatted_prompts,
    }

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Also save as a simple text file for easy loading
    text_file = output_file.with_suffix(".txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for prompt in formatted_prompts:
            f.write(prompt + "\n\n")

    # Save as CSV for inspection
    csv_file = output_file.with_suffix(".csv")
    df = pd.DataFrame(examples)
    df.to_csv(csv_file, index=False, encoding="utf-8")

    logging.info(f"Dataset saved in multiple formats:")
    logging.info(f"  JSON: {output_file}")
    logging.info(f"  Text: {text_file}")
    logging.info(f"  CSV: {csv_file}")


def create_colab_upload_package(dataset_file: Path, output_dir: Path):
    """Create a package with all files needed for Colab."""
    logging.info(f"Creating Colab upload package in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset files
    for ext in [".json", ".txt", ".csv"]:
        src_file = dataset_file.with_suffix(ext)
        if src_file.exists():
            dst_file = output_dir / src_file.name
            dst_file.write_bytes(src_file.read_bytes())

    # Create a README for Colab usage
    readme_content = f"""# MR Extraction Dataset for Google Colab

## Files in this package:

- `{dataset_file.name}` - Main dataset in JSON format
- `{dataset_file.with_suffix(".txt").name}` - Formatted prompts for training
- `{dataset_file.with_suffix(".csv").name}` - Dataset in CSV format for inspection

## Usage in Google Colab:

1. Upload all files to your Colab session
2. Load the dataset:
   ```python
   import json
   with open('{dataset_file.name}', 'r') as f:
       dataset = json.load(f)
   
   examples = dataset['examples']
   formatted_prompts = dataset['formatted_prompts']
   ```

3. Or load the text file directly:
   ```python
   with open('{dataset_file.with_suffix(".txt").name}', 'r') as f:
       content = f.read()
   prompts = [p.strip() for p in content.split('\\n\\n') if p.strip()]
   ```

## Dataset Info:
- Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- For use with Gemma3 270M fine-tuning
- Compatible with Unsloth training format
"""

    readme_file = output_dir / "README_COLAB.md"
    readme_file.write_text(readme_content, encoding="utf-8")

    logging.info(f"Colab package created with {len(list(output_dir.iterdir()))} files")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --data-dir ----
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/intermediate"),
        help="Directory containing OpenAI extraction results",
    )

    # ---- --output-file ----
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/assets/colab/mr_extraction_dataset.json"),
        help="Output dataset file",
    )

    # ---- --train-ratio ----
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Ratio of data to use for training (rest for validation)",
    )

    # ---- --create-splits ----
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create separate train/validation files",
    )

    # ---- --colab-package ----
    parser.add_argument(
        "--colab-package",
        action="store_true",
        help="Create a complete package for Colab upload",
    )

    # ---- --min-examples ----
    parser.add_argument(
        "--min-examples",
        type=int,
        default=10,
        help="Minimum number of examples required",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging()

    if args.dry_run:
        logging.info("DRY RUN MODE - No files will be created")

    # Load OpenAI results
    if not args.data_dir.exists():
        logging.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    raw_data = load_openai_results(args.data_dir)

    if len(raw_data) == 0:
        logging.error("No OpenAI extraction results found")
        sys.exit(1)

    # Normalize and validate data
    normalized_data = []
    for item in raw_data:
        try:
            normalized = normalize_extraction_item(item)
            normalized_data.append(normalized)
        except Exception as e:
            logging.warning(f"Failed to normalize item: {e}")
            continue

    logging.info(f"Normalized {len(normalized_data)} items")

    # Create training examples
    examples = create_training_examples(normalized_data)

    if len(examples) < args.min_examples:
        logging.error(f"Insufficient examples: {len(examples)} < {args.min_examples}")
        sys.exit(1)

    logging.info(f"Created {len(examples)} training examples")
    logging.info(
        f"  Metadata examples: {len([e for e in examples if e['task_type'] == 'metadata'])}"
    )
    logging.info(
        f"  Results examples: {len([e for e in examples if e['task_type'] == 'results'])}"
    )

    if args.dry_run:
        logging.info("DRY RUN: Would save dataset and create package")
        logging.info(f"Example instruction: {examples[0]['instruction'][:100]}...")
        logging.info(f"Example response: {examples[0]['response'][:100]}...")
        return

    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.create_splits:
        # Create train/validation splits
        train_examples, val_examples = create_dataset_splits(examples, args.train_ratio)

        train_file = args.output_file.with_name(args.output_file.stem + "_train.json")
        val_file = args.output_file.with_name(args.output_file.stem + "_val.json")

        save_dataset(train_examples, train_file, "mr_extraction_train")
        save_dataset(val_examples, val_file, "mr_extraction_val")

        logging.info(f"Created training split: {len(train_examples)} examples")
        logging.info(f"Created validation split: {len(val_examples)} examples")
    else:
        # Save full dataset
        save_dataset(examples, args.output_file)

    # Create Colab package if requested
    if args.colab_package:
        package_dir = args.output_file.parent / "colab_package"
        create_colab_upload_package(args.output_file, package_dir)

    logging.info("Dataset preparation completed successfully!")
    logging.info(f"Main dataset: {args.output_file}")

    if args.colab_package:
        logging.info(f"Colab package: {args.output_file.parent / 'colab_package'}")


if __name__ == "__main__":
    main()
