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

    # ---- --assessment-only ----
    parser.add_argument(
        "--assessment-only",
        action="store_true",
        help="Generate assessment figure/table without validation-data access",
    )

    # ---- --heatmap-only ----
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Render only the heatmap; do not access validation or tables",
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
    """Render unchanged assessment means at the manuscript display width.

    Three dimension palettes and their strips share 6-10 normalization.
    The original assessment response scale remains 1-10.
    Validation metrics are retained for call compatibility but are unused.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import (
        LinearSegmentedColormap,
        Normalize,
        to_rgb,
    )

    model_order = overall_metrics.sort_values(
        "overall_mean", ascending=False
    )["model"].tolist()
    groups = [
        ("Q-1", "Q1\nExposure\ntraits"),
        ("Q-2", "Q2\nOutcome\ntraits"),
        ("Q-3", "Q3\nAnalytical\nmethods"),
        ("Q-4", "Q4\nPopulations"),
        ("Q-5", "Q5\nMR findings"),
    ]
    dimensions = ["Accuracy", "Detail", "Completeness"]
    metric_order = [
        f"{code}-{dimension}"
        for code, _ in groups
        for dimension in dimensions
    ]
    pivot_data = group_metrics.pivot_table(
        index="model", columns="metric", values="mean_score"
    ).reindex(index=model_order, columns=metric_order)

    # ---- Fixed physical size and shared score encoding ----
    fig, ax = plt.subplots(figsize=(6.175, 3.65))
    fig.subplots_adjust(left=0.18, right=0.85, bottom=0.25, top=0.78)
    norm = Normalize(vmin=6, vmax=10)
    # Effective original stops from producer revision bcf09f8.
    # D/C use its final assignments, not the earlier overridden lists.
    original_stops = {
        "Accuracy": [
            (0.0, "#282828"), (0.7, "#1d4540"), (0.8, "#2a5a4a"),
            (0.85, "#3a7058"), (0.9, "#4a8568"), (0.93, "#5a9a70"),
            (0.96, "#6aaf78"), (0.98, "#7ac480"), (1.0, "#8ec07c"),
        ],
        "Detail": [
            (0.0, "#282828"), (0.5, "#4a2845"), (0.7, "#8f3f71"),
            (0.8, "#9a4a7c"), (0.85, "#a55687"), (0.9, "#b16286"),
            (0.95, "#c275a0"), (1.0, "#d3869b"),
        ],
        "Completeness": [
            (0.0, "#282828"), (0.5, "#5a3c0a"), (0.7, "#b57614"),
            (0.8, "#c4851a"), (0.85, "#cc8f1d"), (0.9, "#d79921"),
            (0.95, "#e8b528"), (1.0, "#fabd2f"),
        ],
    }
    cmaps = {}
    for dimension, stops in original_stops.items():
        positions = np.array([position for position, _ in stops])
        colors = np.array([to_rgb(color) for _, color in stops])
        # Restrict the original 0-10 ramp to scores 6-10.
        # Interpolate its score-6 endpoint; retain every higher anchor.
        lower_color = tuple(
            np.interp(0.6, positions, colors[:, channel])
            for channel in range(3)
        )
        displayed_stops = [(0.0, lower_color)] + [
            ((position - 0.6) / 0.4, color)
            for position, color in stops if position > 0.6
        ]
        cmaps[dimension] = LinearSegmentedColormap.from_list(
            f"gruvbox_{dimension.lower()}", displayed_stops
        )
    n_models = len(model_order)
    ax.set_xlim(0, len(metric_order))
    ax.set_ylim(0, n_models)

    for i, model in enumerate(model_order):
        for j, metric in enumerate(metric_order):
            value = pivot_data.loc[model, metric]
            cmap = cmaps[dimensions[j % 3]]
            color = cmap(norm(value))
            rect = plt.Rectangle(
                (j, n_models - i - 1),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.add_patch(rect)

            # Uniform annotation color; no additional meaning is encoded.
            ax.text(
                j + 0.5,
                n_models - i - 0.5,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black",
            )

    # ---- Grouped headings and compact dimension key ----
    for index, (_, heading) in enumerate(groups):
        ax.text(
            index * 3 + 1.5,
            n_models + 1.8,
            heading,
            ha="center",
            va="top",
            fontsize=8.5,
            linespacing=1.3,
        )
    for boundary in range(3, len(metric_order), 3):
        ax.axvline(boundary, color="white", linewidth=1.5)
    ax.set_xticks(np.arange(len(metric_order)) + 0.5)
    ax.set_xticklabels(["A", "D", "C"] * len(groups), fontsize=8.5)
    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels(reversed(model_order), fontsize=8.5)
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "LLM Performance Across Extraction Tasks", fontsize=10, pad=46
    )
    fig.text(
        0.535,
        0.12,
        "A: Accuracy    D: Detail    C: Completeness",
        ha="center",
        fontsize=8.5,
    )

    # ---- Adjacent dimension strips with one shared score axis ----
    for index, dimension in enumerate(dimensions):
        cax = fig.add_axes([0.87 + index * 0.021, 0.25, 0.014, 0.53])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmaps[dimension]),
            cax=cax,
            ticks=[6, 7, 8, 9, 10] if index == 2 else [],
        )
        cax.set_xlabel(["A", "D", "C"][index], fontsize=8.5, labelpad=4)
        cax.tick_params(labelsize=8.5, length=2, pad=3)
        colorbar.outline.set_linewidth(0.4)
    fig.text(
        0.898, 0.805, "Mean score", ha="center", va="bottom", fontsize=8.5
    )
    try:
        # No tight bounding-box rescaling: preserve the 6.175-inch width.
        fig.savefig(output_file, dpi=300)
    finally:
        plt.close(fig)

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

    if args.heatmap_only:
        group_metrics = compute_group_metrics(assessment_df)
        overall_metrics = compute_overall_metrics(assessment_df)
        output_file = args.output_figures / "llm-performance.png"
        if args.dry_run:
            print(f"Dry run mode - would generate heatmap: {output_file}")
            return
        args.output_figures.mkdir(parents=True, exist_ok=True)
        create_combined_figure(
            group_metrics,
            overall_metrics,
            pd.DataFrame(),
            output_file,
            args.verbose,
        )
        return

    validation_metrics = pd.DataFrame()
    if not args.assessment_only:
        validation_metrics = load_validation_metrics(
            args.aggregated_dir, args.verbose
        )

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
        if not args.assessment_only:
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

    if not args.assessment_only:
        create_table_validation(
            validation_metrics,
            args.output_tables / "llm-validation-metrics.csv",
            args.verbose,
        )

    if args.verbose:
        print("\nManuscript asset generation complete!")


if __name__ == "__main__":
    main()
