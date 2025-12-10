"""Process reviewer assessment data for LLM extraction performance analysis.

This script consolidates assessment results from two independent reviewers
evaluating the performance of various LLMs on data extraction tasks. The script
reads raw assessment data, harmonizes model names and column names, and outputs
consolidated datasets for downstream analysis.
"""

from pathlib import Path

import pandas as pd


def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments containing input and output paths
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # ---- Input paths ----
    parser.add_argument(
        "--reviewer-1-xlsx",
        type=Path,
        default=Path("data/assets/assessment-reviews/assessment-reviewer-1.xlsx"),
        help="Path to reviewer 1 Excel file",
    )

    # ---- --reviewer-1-sheets ----
    parser.add_argument(
        "--reviewer-1-sheets",
        type=Path,
        default=Path("data/assets/assessment-reviews/reviewer-1-sheets.txt"),
        help="Path to file containing reviewer 1 sheet names",
    )

    # ---- --reviewer-1-columns ----
    parser.add_argument(
        "--reviewer-1-columns",
        type=Path,
        default=Path("data/assets/assessment-reviews/reviewer-1-columns.txt"),
        help="Path to file containing reviewer 1 column names",
    )

    # ---- --reviewer-2-csv ----
    parser.add_argument(
        "--reviewer-2-csv",
        type=Path,
        default=Path("data/assets/assessment-reviews/assessment-reviewer-2.csv"),
        help="Path to reviewer 2 CSV file",
    )

    # ---- Output paths ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/assessment-results"),
        help="Directory for output files",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    res = parser.parse_args()
    return res


def load_reviewer_2_data(csv_path: Path) -> pd.DataFrame:
    """Load and process reviewer 2 data.

    Args:
        csv_path: Path to reviewer 2 CSV file

    Returns:
        Processed dataframe with renamed columns and harmonized model names
    """
    df = pd.read_csv(csv_path, na_values=["None", ""])

    # ---- Rename columns ----
    df = df.rename(columns={"Column1": "pmid", "Column2": "model", "Other": "comment"})

    # ---- Convert pmid to string ----
    df["pmid"] = df["pmid"].astype(str)

    # ---- Harmonize model names ----
    model_name_map = {
        "gpt_4o": "gpt-4o",
        "gpt_5": "gpt-5",
        "gpt_5mini": "gpt-5-mini",
        "deepseek": "deepseek-r1",
    }
    df["model"] = df["model"].replace(model_name_map)

    # ---- Apply score correction for reviewer 2 ----
    # Reviewer 2 scored these columns on an inverted scale; transform as 10 - X
    cols_to_transform = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
    for col in cols_to_transform:
        if col in df.columns:
            df[col] = 10 - df[col]

    res = df
    return res


def load_reviewer_1_data(
    xlsx_path: Path, sheets_path: Path, columns_path: Path
) -> pd.DataFrame:
    """Load and process reviewer 1 data.

    Args:
        xlsx_path: Path to reviewer 1 Excel file
        sheets_path: Path to file containing sheet names
        columns_path: Path to file containing column names

    Returns:
        Processed dataframe with standardized column names and model info
    """
    # ---- Read sheet names ----
    with open(sheets_path, "r") as f:
        sheet_names = [line.strip() for line in f if line.strip()]

    # ---- Read column names ----
    with open(columns_path, "r") as f:
        column_names = [line.strip() for line in f if line.strip()]

    # ---- Read all sheets ----
    dfs = []
    for sheet in sheet_names:
        df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet)

        # ---- Drop unnamed columns (empty columns at the end) ----
        df_sheet = df_sheet.loc[:, ~df_sheet.columns.str.contains("^Unnamed")]

        # ---- Rename columns ----
        if len(df_sheet.columns) == len(column_names):
            df_sheet.columns = column_names
        else:
            raise ValueError(
                f"Column count mismatch for sheet {sheet}: "
                f"expected {len(column_names)}, got {len(df_sheet.columns)}"
            )

        # ---- Add model column ----
        df_sheet["model"] = sheet

        # ---- Convert pmid to string ----
        df_sheet["pmid"] = df_sheet["pmid"].astype(str)

        # ---- Add comment column (empty for reviewer 1) ----
        df_sheet["comment"] = pd.NA

        dfs.append(df_sheet)

    # ---- Concatenate all sheets ----
    res = pd.concat(dfs, ignore_index=True)
    return res


def harmonize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize model names across both reviewers.

    Args:
        df: Dataframe with model column

    Returns:
        Dataframe with harmonized model names
    """
    # ---- Model name mapping ----
    model_name_map = {
        "deepseek-r1-distilled": "deepseek-r1",
        "gpt4-1": "gpt-4-1",
    }

    df["model"] = df["model"].replace(model_name_map)
    res = df
    return res


def set_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set appropriate data types for columns.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with correct column data types
    """
    # ---- String columns ----
    string_cols = ["pmid", "model", "comment"]

    # ---- Add Q-*-b-3 and Q-*-c-3 columns (free text fields) ----
    for col in df.columns:
        if col.endswith("-b-3") or col.endswith("-c-3"):
            string_cols.append(col)

    # ---- Convert string columns ----
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ---- Convert numeric columns ----
    numeric_cols = [col for col in df.columns if col not in string_cols]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    res = df
    return res


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print diagnostic information about the processed data.

    Args:
        df: Processed dataframe
    """
    print("\n==== Diagnostics ====\n")

    # ---- Model value counts ----
    print("Value counts by model:")
    print(df["model"].value_counts().sort_index())
    print()

    # ---- Column data types ----
    print("Column data types:")
    print(df.dtypes)
    print()


def save_outputs(df: pd.DataFrame, output_dir: Path, dry_run: bool) -> None:
    """Save processed data to output files.

    Args:
        df: Processed dataframe
        output_dir: Directory for output files
        dry_run: If True, don't actually write files
    """
    if dry_run:
        print("\n[DRY RUN] Would save outputs to:", output_dir)
        return

    # ---- Create output directory ----
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Define column groups ----
    string_cols = ["pmid", "model", "comment"]

    # ---- Add Q-*-b-3 and Q-*-c-3 columns (free text fields) ----
    for col in df.columns:
        if col.endswith("-b-3") or col.endswith("-c-3"):
            string_cols.append(col)

    numeric_cols = [col for col in df.columns if col not in string_cols]

    # ---- Numeric output ----
    numeric_output_cols = ["pmid", "model"] + numeric_cols
    df_numeric = df[numeric_output_cols]
    numeric_output_path = output_dir / "assessment-results-numeric.csv"
    df_numeric.to_csv(numeric_output_path, index=False)
    print(f"\nSaved numeric results to: {numeric_output_path}")

    # ---- String output ----
    df_strings = df[string_cols]
    strings_output_path = output_dir / "assessment-results-strings.csv"
    df_strings.to_csv(strings_output_path, index=False)
    print(f"Saved string results to: {strings_output_path}")


def main():
    """Main processing workflow."""
    args = parse_args()

    print("==== Processing reviewer assessment data ====\n")

    # ---- Load reviewer 2 data ----
    print(f"Loading reviewer 2 data from: {args.reviewer_2_csv}")
    df_reviewer_2 = load_reviewer_2_data(args.reviewer_2_csv)
    print(f"Loaded {len(df_reviewer_2)} rows from reviewer 2")

    # ---- Load reviewer 1 data ----
    print(f"\nLoading reviewer 1 data from: {args.reviewer_1_xlsx}")
    df_reviewer_1 = load_reviewer_1_data(
        args.reviewer_1_xlsx, args.reviewer_1_sheets, args.reviewer_1_columns
    )
    print(f"Loaded {len(df_reviewer_1)} rows from reviewer 1")

    # ---- Consolidate data ----
    print("\nConsolidating data from both reviewers...")
    df_combined = pd.concat([df_reviewer_1, df_reviewer_2], ignore_index=True)

    # ---- Harmonize model names ----
    print("Harmonizing model names...")
    df_combined = harmonize_model_names(df_combined)

    # ---- Set column data types ----
    print("Setting column data types...")
    df_combined = set_column_dtypes(df_combined)

    # ---- Print diagnostics ----
    print_diagnostics(df_combined)

    # ---- Save outputs ----
    save_outputs(df_combined, args.output_dir, args.dry_run)

    print("\n==== Processing complete ====")


if __name__ == "__main__":
    main()
