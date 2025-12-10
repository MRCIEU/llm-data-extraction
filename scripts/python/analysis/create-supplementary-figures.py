"""Create supplementary figures for LLM data extraction paper.

This script generates supplementary figures showing score distributions
by question group, comparing reviewer 1 and reviewer 2 assessments.
Each figure has three horizontal facets (Accuracy, Detail, Completeness)
with models on Y-axis and mean scores on X-axis.
"""

# ==== Imports ====

# ---- Standard library ----
import argparse
from pathlib import Path

# ---- Data and computation ----
import pandas as pd

# ---- Visualization ----
import altair as alt


# ==== Constants ====

QUESTION_GROUPS = {
    "Q-1": "Exposure Traits",
    "Q-2": "Outcome Traits",
    "Q-3": "Methods",
    "Q-4": "Populations",
    "Q-5": "Results",
}

DIMENSIONS = {
    "Accuracy": "a-3",
    "Detail": "b-2",
    "Completeness": "c-2",
}


# ==== Argument Parsing ====


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --input-file ----
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "data/artifacts/assessment-results/assessment-results-numeric.csv"
        ),
        help="Path to numeric assessment results CSV",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/manuscript-figures"),
        help="Directory for output figures",
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


def load_assessment_data(input_file: Path, verbose: bool = False) -> pd.DataFrame:
    """Load assessment data and add reviewer column.

    The combined data has reviewer 1 data in the first half (rows 0-799)
    and reviewer 2 data in the second half (rows 800-1599).

    Args:
        input_file: Path to assessment CSV
        verbose: Print progress if True

    Returns:
        DataFrame with reviewer column added
    """
    if verbose:
        print(f"Loading data from: {input_file}")

    df = pd.read_csv(input_file)

    # ---- Add reviewer column ----
    # First half is reviewer 1, second half is reviewer 2
    n_rows = len(df)
    midpoint = n_rows // 2
    df["reviewer"] = ["Reviewer 1"] * midpoint + ["Reviewer 2"] * (n_rows - midpoint)

    if verbose:
        print(f"Loaded {len(df)} rows, {df['model'].nunique()} models")
        print(f"Reviewer distribution: {df['reviewer'].value_counts().to_dict()}")

    return df


# ==== Figure Creation ====


def create_question_group_figure(
    df: pd.DataFrame,
    question_group: str,
    group_name: str,
    verbose: bool = False,
) -> alt.Chart:
    """Create figure for a single question group.

    Args:
        df: Assessment DataFrame with reviewer column
        question_group: Question group code (e.g., "Q-1")
        group_name: Human-readable group name (e.g., "Exposure Traits")
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print(f"Creating figure for {question_group}: {group_name}")

    models = sorted(df["model"].unique())
    reviewers = ["Reviewer 1", "Reviewer 2"]

    data = []

    for model in models:
        for reviewer in reviewers:
            model_reviewer_data = df[
                (df["model"] == model) & (df["reviewer"] == reviewer)
            ]

            for dim_name, dim_suffix in DIMENSIONS.items():
                col = f"{question_group}-{dim_suffix}"

                if col in df.columns:
                    scores = model_reviewer_data[col]
                    valid_scores = scores[scores > 0]

                    if len(valid_scores) > 0:
                        mean_val = valid_scores.mean()
                        sd_val = valid_scores.std(ddof=1)
                        # Clip error bar bounds to valid score range [0, 10]
                        lower_bound = max(0, mean_val - sd_val)
                        upper_bound = min(10, mean_val + sd_val)
                        data.append(
                            {
                                "Model": model,
                                "Reviewer": reviewer,
                                "Dimension": dim_name,
                                "Mean": mean_val,
                                "SD": sd_val,
                                "Lower": lower_bound,
                                "Upper": upper_bound,
                            }
                        )

    plot_df = pd.DataFrame(data)

    # ---- Create points ----
    points = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X(
                "Mean:Q",
                title="Mean Score",
                scale=alt.Scale(domain=[0, 10]),
            ),
            y=alt.Y(
                "Model:N",
                sort=alt.EncodingSortField(field="Mean", op="mean", order="descending"),
                title="Model",
            ),
            color=alt.Color(
                "Reviewer:N",
                title="Reviewer",
                scale=alt.Scale(
                    domain=["Reviewer 1", "Reviewer 2"],
                    # Gruvbox colors: aqua and yellow
                    range=["#689d6a", "#d79921"],
                ),
            ),
            tooltip=[
                "Model",
                "Reviewer",
                "Dimension",
                alt.Tooltip("Mean:Q", format=".2f"),
                alt.Tooltip("SD:Q", format=".2f"),
            ],
        )
        .properties(width=250, height=300)
    )

    # ---- Create error bars (clipped to valid range) ----
    error_bars = (
        alt.Chart(plot_df)
        .mark_rule()
        .encode(
            x=alt.X("Lower:Q"),
            x2=alt.X2("Upper:Q"),
            y=alt.Y(
                "Model:N",
                sort=alt.EncodingSortField(field="Mean", op="mean", order="descending"),
            ),
            color=alt.Color("Reviewer:N"),
        )
    )

    # ---- Combine and facet ----
    res = (
        alt.layer(points, error_bars)
        .facet(column=alt.Column("Dimension:N", title="Assessment Dimension"))
        .properties(title=f"{question_group}: {group_name}")
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )

    return res


def create_all_supplementary_figures(
    df: pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """Create all supplementary figures.

    Args:
        df: Assessment DataFrame with reviewer column
        verbose: Print progress if True

    Returns:
        Dictionary mapping figure names to Altair chart objects
    """
    figures = {}

    for question_group, group_name in QUESTION_GROUPS.items():
        fig_name = f"supplementary-{question_group.lower()}-{group_name.lower().replace(' ', '-')}"
        figures[fig_name] = create_question_group_figure(
            df, question_group, group_name, verbose
        )

    return figures


# ==== Output ====


def save_figures(
    figures: dict,
    output_dir: Path,
    dry_run: bool,
    verbose: bool = False,
) -> None:
    """Save figures to output directory.

    Args:
        figures: Dictionary of figure name to Altair chart
        output_dir: Output directory path
        dry_run: If True, don't actually save
        verbose: Print progress if True
    """
    if dry_run:
        print("\n[DRY RUN] Would save figures to:", output_dir)
        for name in figures:
            print(f"  - {name}.json and {name}.png")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, chart in figures.items():
        json_path = output_dir / f"{name}.json"
        png_path = output_dir / f"{name}.png"

        # ---- Save JSON (always works) ----
        chart.save(str(json_path))

        # ---- Try to save PNG ----
        try:
            chart.save(str(png_path), scale_factor=2)
            if verbose:
                print(f"Saved: {json_path} and {png_path}")
        except Exception as e:
            if verbose:
                print(f"Saved: {json_path} (PNG export failed: {e})")


# ==== Main ====


def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        print("Creating supplementary figures...")
        print(f"Input file: {args.input_file}")
        print(f"Output dir: {args.output_dir}")

    # ---- Load data ----
    df = load_assessment_data(args.input_file, args.verbose)

    # ---- Create figures ----
    figures = create_all_supplementary_figures(df, args.verbose)

    # ---- Save figures ----
    save_figures(figures, args.output_dir, args.dry_run, args.verbose)

    if args.verbose:
        print(f"\nCreated {len(figures)} supplementary figures")

    if not args.dry_run:
        print(f"\nFigures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
