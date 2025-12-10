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
    """Create heatmap figure with gruvbox colorschemes.

    Uses different gruvbox color scales for each dimension:
    - Accuracy: gruvbox aqua/cyan tones
    - Detail: gruvbox purple tones
    - Completeness: gruvbox yellow/orange tones

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

    from matplotlib.colors import LinearSegmentedColormap

    # ---- Gruvbox color scales for each dimension ----
    # Use position-based color stops to keep 0 as black but spread colors
    # more across the 8-10 range where actual scores fall
    # Position 0.0 = score 0, position 1.0 = score 10
    # Compress 0-8 range and expand 8-10 range for better differentiation

    # Accuracy: aqua/cyan tones (gruvbox aqua)
    accuracy_colors = [
        (0.0, "#282828"),  # 0: black
        (0.7, "#1d4540"),  # 7: very dark aqua (compressed)
        (0.8, "#2a5a4a"),  # 8: dark aqua
        (0.85, "#3a7058"),  # 8.5: medium-dark aqua
        (0.9, "#4a8568"),  # 9: medium aqua
        (0.93, "#5a9a70"),  # 9.3: medium-light aqua
        (0.96, "#6aaf78"),  # 9.6: light aqua
        (0.98, "#7ac480"),  # 9.8: lighter aqua
        (1.0, "#8ec07c"),  # 10: lightest aqua
    ]

    # Detail: purple tones (gruvbox purple)
    detail_colors = [
        (0.0, "#282828"),  # 0: black
        (0.7, "#3a2035"),  # 7: very dark purple (compressed)
        (0.8, "#5a3050"),  # 8: dark purple
        (0.85, "#7a4068"),  # 8.5: medium-dark purple
        (0.9, "#9a5080"),  # 9: medium purple
        (0.93, "#aa6090"),  # 9.3: medium-light purple
        (0.96, "#ba70a0"),  # 9.6: light purple
        (0.98, "#ca80aa"),  # 9.8: lighter purple
        (1.0, "#d3869b"),  # 10: lightest purple
    ]

    # Completeness: yellow/orange tones (gruvbox yellow/orange)
    completeness_colors = [
        (0.0, "#282828"),  # 0: black
        (0.7, "#4a3008"),  # 7: very dark yellow (compressed)
        (0.8, "#7a5010"),  # 8: dark yellow
        (0.85, "#9a6815"),  # 8.5: medium-dark yellow
        (0.9, "#ba8018"),  # 9: medium yellow
        (0.93, "#ca901c"),  # 9.3: medium-light yellow
        (0.96, "#daa020"),  # 9.6: light yellow
        (0.98, "#eab028"),  # 9.8: lighter yellow
        (1.0, "#fabd2f"),  # 10: lightest yellow
    ]

    # Detail: purple tones (gruvbox purple)
    detail_colors = [
        (0.0, "#282828"),  # 0: black
        (0.5, "#4a2845"),  # 5: very dark purple
        (0.7, "#8f3f71"),  # 7: dark purple
        (0.8, "#9a4a7c"),  # 8: medium-dark purple
        (0.85, "#a55687"),  # 8.5: medium purple
        (0.9, "#b16286"),  # 9: medium-light purple
        (0.95, "#c275a0"),  # 9.5: light purple
        (1.0, "#d3869b"),  # 10: lightest purple
    ]

    # Completeness: yellow/orange tones (gruvbox yellow/orange)
    completeness_colors = [
        (0.0, "#282828"),  # 0: black
        (0.5, "#5a3c0a"),  # 5: very dark yellow
        (0.7, "#b57614"),  # 7: dark yellow
        (0.8, "#c4851a"),  # 8: medium-dark yellow
        (0.85, "#cc8f1d"),  # 8.5: medium yellow
        (0.9, "#d79921"),  # 9: medium-light yellow
        (0.95, "#e8b528"),  # 9.5: light yellow
        (1.0, "#fabd2f"),  # 10: lightest yellow
    ]

    # Create colormaps from position-color pairs
    def make_cmap_from_positions(name, pos_colors):
        positions = [p for p, c in pos_colors]
        colors = [c for p, c in pos_colors]
        return LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))

    cmap_accuracy = make_cmap_from_positions("gruvbox_aqua", accuracy_colors)
    cmap_detail = make_cmap_from_positions("gruvbox_purple", detail_colors)
    cmap_completeness = make_cmap_from_positions("gruvbox_yellow", completeness_colors)

    # ---- Prepare data ----
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

    # ---- Create figure with custom coloring ----
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get dimensions for manual plotting
    n_models = len(model_order)
    n_metrics = len(metric_order)

    # Create base heatmap structure (for grid lines and annotations)
    # We'll overlay colored rectangles for each cell
    ax.set_xlim(0, n_metrics)
    ax.set_ylim(0, n_models)

    # Plot each cell with appropriate colormap
    for i, model in enumerate(model_order):
        for j, metric in enumerate(metric_order):
            value = pivot_data.loc[model, metric]

            # Determine which colormap to use based on dimension
            if "Accuracy" in metric:
                cmap = cmap_accuracy
            elif "Detail" in metric:
                cmap = cmap_detail
            else:  # Completeness
                cmap = cmap_completeness

            # Normalize value to [0, 1] for colormap (scores are 0-10)
            norm_value = value / 10.0
            color = cmap(norm_value)

            # Draw rectangle
            rect = plt.Rectangle(
                (j, n_models - i - 1),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Add annotation
            # Use white text for dark backgrounds, black for light
            text_color = "white" if norm_value < 0.6 else "black"
            ax.text(
                j + 0.5,
                n_models - i - 0.5,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    # ---- Set axis labels and ticks ----
    ax.set_xticks([x + 0.5 for x in range(n_metrics)])
    ax.set_xticklabels(metric_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([y + 0.5 for y in range(n_models)])
    ax.set_yticklabels(reversed(model_order), fontsize=10)

    ax.set_xlabel("Extraction Group and Metric", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("LLM Performance Across Extraction Tasks", fontsize=13, pad=10)

    # ---- Add legend for color scales ----
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    divider = make_axes_locatable(ax)

    # Add three small colorbars for each dimension
    cax1 = divider.append_axes("right", size="2%", pad=0.1)
    cax2 = divider.append_axes("right", size="2%", pad=0.3)
    cax3 = divider.append_axes("right", size="2%", pad=0.3)

    norm = Normalize(vmin=0, vmax=10)

    cb1 = ColorbarBase(cax1, cmap=cmap_accuracy, norm=norm)
    cb1.set_label("Accuracy", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    cb2 = ColorbarBase(cax2, cmap=cmap_detail, norm=norm)
    cb2.set_label("Detail", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    cb3 = ColorbarBase(cax3, cmap=cmap_completeness, norm=norm)
    cb3.ax.tick_params(labelsize=8)
    # No label for cb3 to avoid duplicate "Completeness" text

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
