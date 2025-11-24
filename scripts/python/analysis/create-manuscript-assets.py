"""Create manuscript assets (figures and tables) for LLM data extraction paper.

This script generates:
- Figure 1: Combined heatmap and validation metrics visualization
- Table: Comprehensive model performance by extraction group
- Table: Validation and data quality metrics
"""

# ==== Imports ====

# ---- Standard library ----
import argparse
import json
from pathlib import Path

# ---- Visualization ----
import matplotlib.pyplot as plt
import numpy as np

# ---- Data and computation ----
import pandas as pd
import seaborn as sns

# ==== Constants ====

SAMPLE_SIZE_SMALL = 7000
SAMPLE_SIZE_FULL = 15635


# ==== Argument Parsing ====


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --assessment-file ----
    parser.add_argument(
        "--assessment-file",
        type=Path,
        default=Path(
            "data/artifacts/assessment-results/assessment-results-numeric.csv"
        ),
        help="Path to numeric assessment results CSV",
    )

    # ---- --aggregated-dir ----
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=Path("data/intermediate/llm-results-aggregated"),
        help="Directory containing aggregated LLM results",
    )

    # ---- --output-figures ----
    parser.add_argument(
        "--output-figures",
        type=Path,
        default=Path("data/artifacts/manuscript-figures"),
        help="Directory for output figures",
    )

    # ---- --output-tables ----
    parser.add_argument(
        "--output-tables",
        type=Path,
        default=Path("data/artifacts/manuscript-tables"),
        help="Directory for output tables",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview without generating outputs",
    )

    # ---- --verbose ----
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    return parser.parse_args()


# ==== Data Loading ====


def load_assessment_data(assessment_file: Path, verbose: bool = False) -> pd.DataFrame:
    """Load assessment results data.

    Args:
        assessment_file: Path to assessment CSV
        verbose: Print detailed information if True

    Returns:
        DataFrame with assessment data
    """
    if verbose:
        print(f"Loading assessment data from: {assessment_file}")

    df = pd.read_csv(assessment_file)

    if verbose:
        print(f"Loaded {len(df)} rows, {df['model'].nunique()} models")

    return df


def get_sample_size(model: str) -> int:
    """Get sample size for a model.

    Args:
        model: Model name

    Returns:
        Sample size (number of input documents)
    """
    if model in ["gpt-4-1", "gpt-5"]:
        return SAMPLE_SIZE_FULL
    else:
        return SAMPLE_SIZE_SMALL


def load_validation_metrics(
    aggregated_dir: Path, verbose: bool = False
) -> pd.DataFrame:
    """Load validation metrics from processed results.

    Args:
        aggregated_dir: Directory with aggregated results
        verbose: Print detailed information if True

    Returns:
        DataFrame with validation metrics per model
    """
    if verbose:
        print(f"Loading validation metrics from: {aggregated_dir}")

    models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-4-1",
        "o4-mini",
        "gpt-4o",
        "llama3-2",
        "deepseek-r1-distilled",
        "llama3",
    ]

    metrics_list = []

    for model in models:
        model_dir = aggregated_dir / model
        if not model_dir.exists():
            if verbose:
                print(f"Skipping {model}: directory not found")
            continue

        valid_file = model_dir / "processed_results_valid.json"
        invalid_file = model_dir / "processed_results_invalid.json"
        all_file = model_dir / "processed_results.json"

        if not all_file.exists():
            if verbose:
                print(f"Skipping {model}: no processed results")
            continue

        valid_count = 0
        invalid_count = 0
        total_count = 0

        with open(all_file, "r") as f:
            total_count = len(json.load(f))

        if valid_file.exists():
            with open(valid_file, "r") as f:
                valid_count = len(json.load(f))

        if invalid_file.exists():
            with open(invalid_file, "r") as f:
                invalid_count = len(json.load(f))

        validation_rate = (invalid_count / total_count * 100) if total_count > 0 else 0

        sample_size = get_sample_size(model)

        metrics_list.append(
            {
                "model": model,
                "sample_size": sample_size,
                "total_extractions": total_count,
                "valid_extractions": valid_count,
                "invalid_extractions": invalid_count,
                "validation_issue_rate": validation_rate,
            }
        )

        if verbose:
            print(
                f"{model}: sample={sample_size}, total={total_count}, "
                f"invalid={invalid_count} ({validation_rate:.1f}%)"
            )

    res = pd.DataFrame(metrics_list)
    return res


# ==== Data Processing ====


def compute_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics by extraction group and dimension.

    Args:
        df: Assessment data

    Returns:
        DataFrame with mean scores by model, group, and dimension
    """
    groups = {
        "Q-1": "Exposure Traits",
        "Q-2": "Outcome Traits",
        "Q-3": "Methods",
        "Q-4": "Populations",
        "Q-5": "Results",
    }

    dimensions = {
        "Accuracy": "a-3",
        "Detail": "b-2",
        "Completeness": "c-2",
    }

    records = []

    for model in df["model"].unique():
        model_data = df[df["model"] == model]

        for group_code, group_name in groups.items():
            for dim_name, dim_suffix in dimensions.items():
                col = f"{group_code}-{dim_suffix}"

                if col in df.columns:
                    scores = model_data[col]
                    valid_scores = scores[scores > 0]

                    if len(valid_scores) > 0:
                        mean_score = valid_scores.mean()
                        records.append(
                            {
                                "model": model,
                                "group_code": group_code,
                                "group_name": group_name,
                                "dimension": dim_name,
                                "metric": f"{group_code}-{dim_name}",
                                "mean_score": mean_score,
                            }
                        )

    res = pd.DataFrame(records)
    return res


def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall performance metrics per model.

    Args:
        df: Assessment data

    Returns:
        DataFrame with overall metrics by model
    """
    score_cols = [col for col in df.columns if col.endswith(("a-3", "b-2", "c-2"))]

    records = []

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        all_scores = model_data[score_cols].values.flatten()
        valid_scores = all_scores[all_scores > 0]

        if len(valid_scores) > 0:
            records.append(
                {
                    "model": model,
                    "overall_mean": valid_scores.mean(),
                    "overall_sd": valid_scores.std(),
                }
            )

    res = pd.DataFrame(records)
    return res


# ==== Figure Generation ====


def create_combined_figure(
    group_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    output_file: Path,
    verbose: bool = False,
):
    """Create heatmap figure with gruvbox dark colorscheme.

    Args:
        group_metrics: Metrics by group and dimension
        overall_metrics: Overall performance metrics
        validation_metrics: Validation metrics
        output_file: Path to save figure
        verbose: Print detailed information if True
    """
    if verbose:
        print("Creating figure...")

    model_order = overall_metrics.sort_values("overall_mean", ascending=False)[
        "model"
    ].tolist()

    gruvbox_colors = [
        # "#282828",
        # "#3c3836",
        "#504945",
        "#665c54",
        "#7c6f64",
        "#928374",
        "#a89984",
        "#bdae93",
        "#d5c4a1",
        "#ebdbb2",
        "#fbf1c7",
    ]
    gruvbox_colors.reverse()

    from matplotlib.colors import LinearSegmentedColormap

    gruvbox_cmap = LinearSegmentedColormap.from_list("gruvbox_dark", gruvbox_colors)

    fig = plt.figure(figsize=(14, 7))

    pivot_data = group_metrics.pivot_table(
        index="model", columns="metric", values="mean_score"
    )
    pivot_data = pivot_data.reindex(model_order)

    metric_order = [
        f"{g}-{d}"
        for g in ["Q-1", "Q-2", "Q-3", "Q-4", "Q-5"]
        for d in ["Accuracy", "Detail", "Completeness"]
    ]
    pivot_data = pivot_data[metric_order]

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".1f",
        cmap=gruvbox_cmap,
        vmin=0,
        vmax=10,
        cbar_kws={"label": "Score (0-10)"},
        linewidths=0.5,
        linecolor="white",
    )

    plt.xlabel("Extraction Group and Metric", fontsize=11)
    plt.ylabel("Model", fontsize=11)
    plt.title("LLM Performance Across Extraction Tasks", fontsize=13, pad=10)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"Saved figure to: {output_file}")


# ==== Table Generation ====


def create_table_performance(
    group_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    output_file: Path,
    verbose: bool = False,
):
    """Create performance table in CSV and LaTeX formats.

    Args:
        group_metrics: Metrics by group and dimension
        overall_metrics: Overall performance metrics
        output_file: Path to save table (CSV)
        verbose: Print detailed information if True
    """
    if verbose:
        print("Creating performance table...")

    pivot_data = group_metrics.pivot_table(
        index="model", columns="metric", values="mean_score"
    )

    table = overall_metrics.merge(pivot_data, on="model")

    table["family"] = table["model"].apply(
        lambda x: "OpenAI"
        if x in ["gpt-5", "gpt-4-1", "gpt-5-mini", "gpt-4o", "o4-mini"]
        else "Local"
    )

    table = table.sort_values("overall_mean", ascending=False)

    cols = ["model", "family", "overall_mean", "overall_sd"] + [
        col for col in table.columns if col.startswith("Q-")
    ]
    table = table[cols]

    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(2)

    table.to_csv(output_file, index=False)

    tex_file = output_file.with_suffix(".tex")
    latex_str = table.to_latex(index=False, float_format="%.2f", escape=False)
    tex_file.write_text(latex_str)

    if verbose:
        print(f"Saved performance table to: {output_file}")
        print(f"Saved performance table (LaTeX) to: {tex_file}")


def create_table_validation(
    validation_metrics: pd.DataFrame,
    output_file: Path,
    verbose: bool = False,
):
    """Create validation metrics table in CSV and LaTeX formats.

    Args:
        validation_metrics: Validation metrics by model
        output_file: Path to save table (CSV)
        verbose: Print detailed information if True
    """
    if verbose:
        print("Creating validation table...")

    table = validation_metrics.copy()

    table = table.sort_values("validation_issue_rate", ascending=True)

    cols = [
        "model",
        "sample_size",
        "total_extractions",
        "valid_extractions",
        "invalid_extractions",
        "validation_issue_rate",
    ]
    table = table[cols]

    table["validation_issue_rate"] = table["validation_issue_rate"].round(1)

    table.to_csv(output_file, index=False)

    tex_file = output_file.with_suffix(".tex")
    latex_str = table.to_latex(index=False, float_format="%.1f", escape=False)
    tex_file.write_text(latex_str)

    if verbose:
        print(f"Saved validation table to: {output_file}")
        print(f"Saved validation table (LaTeX) to: {tex_file}")


# ==== Main ====


def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        print("Starting manuscript asset generation...")
        print(f"Assessment file: {args.assessment_file}")
        print(f"Aggregated dir: {args.aggregated_dir}")
        print(f"Output figures: {args.output_figures}")
        print(f"Output tables: {args.output_tables}")

    assessment_df = load_assessment_data(args.assessment_file, args.verbose)

    validation_metrics = load_validation_metrics(args.aggregated_dir, args.verbose)

    group_metrics = compute_group_metrics(assessment_df)
    overall_metrics = compute_overall_metrics(assessment_df)

    if args.dry_run:
        print("\nDry run mode - would generate:")
        print(f"  - Figure: {args.output_figures / 'llm-performance.png'}")
        print(
            f"  - Table: {args.output_tables / 'llm-extraction-performance-detail.csv'}"
        )
        print(
            f"  - Table: {args.output_tables / 'llm-extraction-performance-detail.tex'}"
        )
        print(f"  - Table: {args.output_tables / 'llm-validation-metrics.csv'}")
        print(f"  - Table: {args.output_tables / 'llm-validation-metrics.tex'}")
        return

    args.output_figures.mkdir(parents=True, exist_ok=True)
    args.output_tables.mkdir(parents=True, exist_ok=True)

    create_combined_figure(
        group_metrics,
        overall_metrics,
        validation_metrics,
        args.output_figures / "llm-performance.png",
        args.verbose,
    )

    create_table_performance(
        group_metrics,
        overall_metrics,
        args.output_tables / "llm-extraction-performance-detail.csv",
        args.verbose,
    )

    create_table_validation(
        validation_metrics,
        args.output_tables / "llm-validation-metrics.csv",
        args.verbose,
    )

    if args.verbose:
        print("\nManuscript asset generation complete!")


if __name__ == "__main__":
    main()
