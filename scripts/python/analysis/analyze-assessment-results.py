"""Analyze LLM extraction assessment results.

This script performs comprehensive analysis of assessment data from two
independent reviewers evaluating 8 LLMs on data extraction tasks across
5 extraction groups and 3 assessment dimensions (accuracy, detail,
completeness).
"""

# ==== Imports ====

# ---- Standard library ----
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import sys

# ---- Data and computation ----
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

# ---- Visualization ----
import altair as alt


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
        default=Path("data/intermediate/assessment-analysis"),
        help="Directory for output files",
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


# ==== Data Loading and Preprocessing ====


def load_and_validate_data(input_file: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and validate assessment data.

    Args:
        input_file: Path to CSV file with assessment results
        verbose: Print detailed information if True

    Returns:
        Validated DataFrame with assessment data
    """
    if verbose:
        print(f"Loading data from: {input_file}")

    df = pd.read_csv(input_file)

    if verbose:
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Unique PMIDs: {df['pmid'].nunique()}")

    # ---- Data validation ----
    score_cols = [col for col in df.columns if col.endswith(("a-3", "b-2", "c-2"))]

    for col in score_cols:
        valid_scores = df[col][(df[col] > 0) & (df[col] <= 10)]
        if len(valid_scores) > 0:
            if (valid_scores.min() < 1) or (valid_scores.max() > 10):
                print(f"Warning: {col} has scores outside 1-10 range")

    # ---- Check for missing values ----
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0 and verbose:
        print("\nMissing values detected:")
        print(missing_counts[missing_counts > 0])

    return df


def identify_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """Add reviewer identifier column.

    Args:
        df: Assessment DataFrame

    Returns:
        DataFrame with added 'reviewer' column
    """
    df = df.copy()
    df = df.sort_values(["pmid", "model"]).reset_index(drop=True)

    # ---- Assume first occurrence = reviewer 1, second = reviewer 2 ----
    df["reviewer"] = df.groupby(["pmid", "model"]).cumcount() + 1

    return df


# ==== Summary Statistics Tables ====


def create_table_1a_overall_performance(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 1A: Overall performance metrics by model.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with overall performance metrics
    """
    if verbose:
        print("\nCreating Table 1A: Overall performance metrics")

    models = sorted(df["model"].unique())
    results = []

    for model in models:
        model_data = df[df["model"] == model]

        # ---- Accuracy scores ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy_scores = model_data[acc_cols].values.flatten()
        accuracy_scores = accuracy_scores[accuracy_scores > 0]

        # ---- Detail scores ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail_scores = model_data[detail_cols].values.flatten()
        detail_scores = detail_scores[detail_scores > 0]

        # ---- Completeness scores ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        comp_scores = model_data[comp_cols].values.flatten()
        comp_scores = comp_scores[comp_scores > 0]

        # ---- Overall scores ----
        all_scores = np.concatenate([accuracy_scores, detail_scores, comp_scores])

        results.append(
            {
                "Model": model,
                "Mean_Accuracy": np.mean(accuracy_scores),
                "SD_Accuracy": np.std(accuracy_scores, ddof=1),
                "Mean_Detail": np.mean(detail_scores),
                "SD_Detail": np.std(detail_scores, ddof=1),
                "Mean_Completeness": np.mean(comp_scores),
                "SD_Completeness": np.std(comp_scores, ddof=1),
                "Overall_Mean": np.mean(all_scores),
                "Overall_SD": np.std(all_scores, ddof=1),
            }
        )

    res = pd.DataFrame(results)
    res = res.sort_values("Overall_Mean", ascending=False).reset_index(drop=True)

    return res


def create_table_1b_performance_by_group(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 1B: Performance metrics by extraction group.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with performance by extraction group
    """
    if verbose:
        print("\nCreating Table 1B: Performance by extraction group")

    models = sorted(df["model"].unique())
    results = []

    for model in models:
        model_data = df[df["model"] == model]
        row = {"Model": model}

        for q in range(1, 6):
            acc_col = f"Q-{q}-a-3"
            detail_col = f"Q-{q}-b-2"
            comp_col = f"Q-{q}-c-2"

            acc_vals = model_data[acc_col]
            acc_vals = acc_vals[acc_vals > 0]
            row[f"Q-{q}-Accuracy"] = np.mean(acc_vals)

            detail_vals = model_data[detail_col]
            detail_vals = detail_vals[detail_vals > 0]
            row[f"Q-{q}-Detail"] = np.mean(detail_vals)

            comp_vals = model_data[comp_col]
            comp_vals = comp_vals[comp_vals > 0]
            row[f"Q-{q}-Completeness"] = np.mean(comp_vals)

        results.append(row)

    res = pd.DataFrame(results)
    return res


def create_table_2_error_rates(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Create Table 2: Error rates and data quality metrics.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with error rates and quality metrics
    """
    if verbose:
        print("\nCreating Table 2: Error rates and data quality")

    models = sorted(df["model"].unique())
    results = []

    for model in models:
        model_data = df[df["model"] == model]

        # ---- Total results ----
        total_cols = ["Q-1-a-1", "Q-2-a-1", "Q-3-a-1", "Q-4-a-1", "Q-5-a-1"]
        total_vals = model_data[total_cols].values.flatten()
        mean_total = np.mean(total_vals)

        # ---- Incorrect results ----
        incorrect_cols = ["Q-1-a-2", "Q-2-a-2", "Q-3-a-2", "Q-4-a-2", "Q-5-a-2"]
        incorrect_vals = model_data[incorrect_cols].values.flatten()
        mean_incorrect = np.mean(incorrect_vals)

        # ---- Missing results ----
        missing_cols = ["Q-1-c-1", "Q-2-c-1", "Q-3-c-1", "Q-4-c-1", "Q-5-c-1"]
        missing_vals = model_data[missing_cols].values.flatten()
        mean_missing = np.mean(missing_vals)

        # ---- Sufficient detail ----
        detail_cols = ["Q-1-b-1", "Q-2-b-1", "Q-3-b-1", "Q-4-b-1", "Q-5-b-1"]
        detail_vals = model_data[detail_cols].values.flatten()
        mean_sufficient_detail = np.mean(detail_vals)

        # ---- Calculated rates ----
        error_rate = mean_incorrect / mean_total if mean_total > 0 else 0
        completeness_rate = (
            (mean_total - mean_missing) / mean_total if mean_total > 0 else 0
        )

        results.append(
            {
                "Model": model,
                "Mean_Total": mean_total,
                "Mean_Incorrect": mean_incorrect,
                "Mean_Missing": mean_missing,
                "Mean_Sufficient_Detail": mean_sufficient_detail,
                "Error_Rate": error_rate,
                "Completeness_Rate": completeness_rate,
            }
        )

    res = pd.DataFrame(results)
    return res


def create_table_3_pairwise_comparisons(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 3: Pairwise model comparisons.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with pairwise statistical comparisons
    """
    if verbose:
        print("\nCreating Table 3: Pairwise model comparisons")

    # ---- Calculate overall scores per study ----
    df_scores = identify_reviewers(df)

    score_cols = [
        "Q-1-a-3",
        "Q-1-b-2",
        "Q-1-c-2",
        "Q-2-a-3",
        "Q-2-b-2",
        "Q-2-c-2",
        "Q-3-a-3",
        "Q-3-b-2",
        "Q-3-c-2",
        "Q-4-a-3",
        "Q-4-b-2",
        "Q-4-c-2",
        "Q-5-a-3",
        "Q-5-b-2",
        "Q-5-c-2",
    ]

    df_scores["overall_score"] = df_scores[score_cols].mean(axis=1)

    # ---- Define key comparisons ----
    comparisons = [
        ("gpt-5", "gpt-4o"),
        ("gpt-5", "deepseek-r1"),
        ("gpt-5", "o4-mini"),
        ("gpt-4o", "deepseek-r1"),
        ("gpt-4o", "gpt-4-1"),
        ("deepseek-r1", "llama3-2"),
        ("o4-mini", "gpt-5-mini"),
        ("llama3-2", "llama3"),
    ]

    results = []

    for model1, model2 in comparisons:
        scores1 = df_scores[df_scores["model"] == model1]["overall_score"]
        scores2 = df_scores[df_scores["model"] == model2]["overall_score"]

        if len(scores1) == 0 or len(scores2) == 0:
            continue

        # ---- Align paired data ----
        df1 = df_scores[df_scores["model"] == model1][
            ["pmid", "reviewer", "overall_score"]
        ].copy()
        df2 = df_scores[df_scores["model"] == model2][
            ["pmid", "reviewer", "overall_score"]
        ].copy()

        merged = pd.merge(df1, df2, on=["pmid", "reviewer"], suffixes=("_1", "_2"))

        if len(merged) == 0:
            continue

        # ---- Statistical test ----
        stat, pval = stats.wilcoxon(
            merged["overall_score_1"],
            merged["overall_score_2"],
            alternative="two-sided",
        )

        # ---- Effect size (Cohen's d) ----
        mean_diff = merged["overall_score_1"].mean() - merged["overall_score_2"].mean()
        pooled_std = np.sqrt(
            (merged["overall_score_1"].var() + merged["overall_score_2"].var()) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # ---- Significance level ----
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = ""

        results.append(
            {
                "Model_1": model1,
                "Model_2": model2,
                "Model_1_Mean": merged["overall_score_1"].mean(),
                "Model_2_Mean": merged["overall_score_2"].mean(),
                "Mean_Difference": mean_diff,
                "Test_Statistic": stat,
                "P_Value": pval,
                "Significance": sig,
                "Cohens_D": cohens_d,
                "N_Pairs": len(merged),
            }
        )

    res = pd.DataFrame(results)
    return res


def create_table_4_inter_rater_reliability(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 4: Inter-rater reliability.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with ICC statistics
    """
    if verbose:
        print("\nCreating Table 4: Inter-rater reliability")

    df_with_reviewer = identify_reviewers(df)

    results = []

    # ---- Overall ICC ----
    score_cols = [
        "Q-1-a-3",
        "Q-1-b-2",
        "Q-1-c-2",
        "Q-2-a-3",
        "Q-2-b-2",
        "Q-2-c-2",
        "Q-3-a-3",
        "Q-3-b-2",
        "Q-3-c-2",
        "Q-4-a-3",
        "Q-4-b-2",
        "Q-4-c-2",
        "Q-5-a-3",
        "Q-5-b-2",
        "Q-5-c-2",
    ]

    # ---- Reshape for ICC calculation ----
    icc_data = []
    for _, row in df_with_reviewer.iterrows():
        for col in score_cols:
            if row[col] > 0:
                icc_data.append(
                    {
                        "pmid": row["pmid"],
                        "model": row["model"],
                        "reviewer": row["reviewer"],
                        "score": row[col],
                        "metric": col,
                    }
                )

    icc_df = pd.DataFrame(icc_data)

    # ---- Calculate overall ICC ----
    try:
        icc_df["target"] = (
            icc_df["pmid"].astype(str)
            + "_"
            + icc_df["model"].astype(str)
            + "_"
            + icc_df["metric"].astype(str)
        )
        icc_result = pg.intraclass_corr(
            data=icc_df, targets="target", raters="reviewer", ratings="score"
        )
        icc_row = icc_result[icc_result["Type"] == "ICC2"].iloc[0]

        results.append(
            {
                "Category": "Overall",
                "ICC": icc_row["ICC"],
                "CI_Lower": icc_row["CI95%"][0],
                "CI_Upper": icc_row["CI95%"][1],
                "Interpretation": interpret_icc(icc_row["ICC"]),
            }
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Could not calculate overall ICC: {e}")

    # ---- ICC by metric type ----
    for metric_type in ["Accuracy", "Detail", "Completeness"]:
        if metric_type == "Accuracy":
            metric_cols = [f"Q-{i}-a-3" for i in range(1, 6)]
        elif metric_type == "Detail":
            metric_cols = [f"Q-{i}-b-2" for i in range(1, 6)]
        else:
            metric_cols = [f"Q-{i}-c-2" for i in range(1, 6)]

        subset = icc_df[icc_df["metric"].isin(metric_cols)]

        try:
            icc_result = pg.intraclass_corr(
                data=subset, targets="target", raters="reviewer", ratings="score"
            )
            icc_row = icc_result[icc_result["Type"] == "ICC2"].iloc[0]

            results.append(
                {
                    "Category": metric_type,
                    "ICC": icc_row["ICC"],
                    "CI_Lower": icc_row["CI95%"][0],
                    "CI_Upper": icc_row["CI95%"][1],
                    "Interpretation": interpret_icc(icc_row["ICC"]),
                }
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not calculate {metric_type} ICC: {e}")

    # ---- ICC by extraction group ----
    for q in range(1, 6):
        group_cols = [f"Q-{q}-a-3", f"Q-{q}-b-2", f"Q-{q}-c-2"]
        subset = icc_df[icc_df["metric"].isin(group_cols)]

        try:
            icc_result = pg.intraclass_corr(
                data=subset, targets="target", raters="reviewer", ratings="score"
            )
            icc_row = icc_result[icc_result["Type"] == "ICC2"].iloc[0]

            results.append(
                {
                    "Category": f"Q-{q}",
                    "ICC": icc_row["ICC"],
                    "CI_Lower": icc_row["CI95%"][0],
                    "CI_Upper": icc_row["CI95%"][1],
                    "Interpretation": interpret_icc(icc_row["ICC"]),
                }
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not calculate Q-{q} ICC: {e}")

    res = pd.DataFrame(results)
    return res


def interpret_icc(icc: float) -> str:
    """Interpret ICC value.

    Args:
        icc: ICC coefficient

    Returns:
        Interpretation string
    """
    if icc < 0.5:
        return "Poor"
    elif icc < 0.75:
        return "Moderate"
    elif icc < 0.9:
        return "Good"
    else:
        return "Excellent"


def create_table_5_model_insights(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 5: Model-specific insights.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with model-specific metrics
    """
    if verbose:
        print("\nCreating Table 5: Model-specific insights")

    models = sorted(df["model"].unique())
    results = []

    for model in models:
        model_data = df[df["model"] == model]

        # ---- Median and IQR for each dimension ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy_scores = model_data[acc_cols].values.flatten()
        accuracy_scores = accuracy_scores[accuracy_scores > 0]

        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail_scores = model_data[detail_cols].values.flatten()
        detail_scores = detail_scores[detail_scores > 0]

        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        comp_scores = model_data[comp_cols].values.flatten()
        comp_scores = comp_scores[comp_scores > 0]

        # ---- Performance by group ----
        group_scores = {}
        for q in range(1, 6):
            q_scores = model_data[
                [f"Q-{q}-a-3", f"Q-{q}-b-2", f"Q-{q}-c-2"]
            ].values.flatten()
            q_scores = q_scores[q_scores > 0]
            group_scores[f"Q-{q}"] = np.mean(q_scores)

        best_group = max(group_scores, key=group_scores.get)
        worst_group = min(group_scores, key=group_scores.get)
        performance_range = group_scores[best_group] - group_scores[worst_group]

        results.append(
            {
                "Model": model,
                "Median_Accuracy": np.median(accuracy_scores),
                "IQR_Accuracy": np.percentile(accuracy_scores, 75)
                - np.percentile(accuracy_scores, 25),
                "Median_Detail": np.median(detail_scores),
                "IQR_Detail": np.percentile(detail_scores, 75)
                - np.percentile(detail_scores, 25),
                "Median_Completeness": np.median(comp_scores),
                "IQR_Completeness": np.percentile(comp_scores, 75)
                - np.percentile(comp_scores, 25),
                "Best_Group": best_group,
                "Worst_Group": worst_group,
                "Performance_Range": performance_range,
            }
        )

    res = pd.DataFrame(results)
    return res


def create_table_6_model_family_comparison(
    df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create Table 6: Model family comparison.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        DataFrame with family-level comparisons
    """
    if verbose:
        print("\nCreating Table 6: Model family comparison")

    # ---- Define families ----
    openai_models = ["o4-mini", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini"]
    local_models = ["llama3", "llama3-2", "deepseek-r1"]

    results = []

    for family_name, family_models in [
        ("OpenAI", openai_models),
        ("Local", local_models),
    ]:
        family_data = df[df["model"].isin(family_models)]

        # ---- Accuracy ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy_scores = family_data[acc_cols].values.flatten()
        accuracy_scores = accuracy_scores[accuracy_scores > 0]

        # ---- Detail ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail_scores = family_data[detail_cols].values.flatten()
        detail_scores = detail_scores[detail_scores > 0]

        # ---- Completeness ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        comp_scores = family_data[comp_cols].values.flatten()
        comp_scores = comp_scores[comp_scores > 0]

        # ---- Overall ----
        all_scores = np.concatenate([accuracy_scores, detail_scores, comp_scores])

        results.append(
            {
                "Family": family_name,
                "Mean_Accuracy": np.mean(accuracy_scores),
                "SD_Accuracy": np.std(accuracy_scores, ddof=1),
                "Mean_Detail": np.mean(detail_scores),
                "SD_Detail": np.std(detail_scores, ddof=1),
                "Mean_Completeness": np.mean(comp_scores),
                "SD_Completeness": np.std(comp_scores, ddof=1),
                "Mean_Overall": np.mean(all_scores),
                "SD_Overall": np.std(all_scores, ddof=1),
                "N_Models": len(family_models),
            }
        )

    # ---- Statistical comparison ----
    openai_data = df[df["model"].isin(openai_models)]
    local_data = df[df["model"].isin(local_models)]

    score_cols = [
        "Q-1-a-3",
        "Q-1-b-2",
        "Q-1-c-2",
        "Q-2-a-3",
        "Q-2-b-2",
        "Q-2-c-2",
        "Q-3-a-3",
        "Q-3-b-2",
        "Q-3-c-2",
        "Q-4-a-3",
        "Q-4-b-2",
        "Q-4-c-2",
        "Q-5-a-3",
        "Q-5-b-2",
        "Q-5-c-2",
    ]

    openai_scores = openai_data[score_cols].values.flatten()
    openai_scores = openai_scores[openai_scores > 0]

    local_scores = local_data[score_cols].values.flatten()
    local_scores = local_scores[local_scores > 0]

    stat, pval = stats.mannwhitneyu(
        openai_scores, local_scores, alternative="two-sided"
    )

    # ---- Effect size ----
    cohens_d = (np.mean(openai_scores) - np.mean(local_scores)) / np.sqrt(
        (np.var(openai_scores) + np.var(local_scores)) / 2
    )

    results.append(
        {
            "Family": "Statistical_Comparison",
            "Mean_Accuracy": np.nan,
            "SD_Accuracy": np.nan,
            "Mean_Detail": np.nan,
            "SD_Detail": np.nan,
            "Mean_Completeness": np.nan,
            "SD_Completeness": np.nan,
            "Mean_Overall": pval,
            "SD_Overall": cohens_d,
            "N_Models": stat,
        }
    )

    res = pd.DataFrame(results)
    return res


# ==== Visualizations ====


def create_figure_1_overall_heatmap(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 1: Overall model performance heatmap.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 1: Overall performance heatmap")

    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]

        # ---- Accuracy ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy = model_data[acc_cols].values.flatten()
        accuracy = accuracy[accuracy > 0]

        # ---- Detail ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail = model_data[detail_cols].values.flatten()
        detail = detail[detail > 0]

        # ---- Completeness ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        completeness = model_data[comp_cols].values.flatten()
        completeness = completeness[completeness > 0]

        data.extend(
            [
                {"Model": model, "Metric": "Accuracy", "Score": np.mean(accuracy)},
                {"Model": model, "Metric": "Detail", "Score": np.mean(detail)},
                {
                    "Model": model,
                    "Metric": "Completeness",
                    "Score": np.mean(completeness),
                },
            ]
        )

    plot_df = pd.DataFrame(data)

    chart = (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("Model:N", sort=models, title="Model"),
            y=alt.Y("Metric:N", title="Assessment Dimension"),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 10]),
                title="Mean Score",
            ),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2f")],
        )
        .properties(width=600, height=200, title="Overall Model Performance Heatmap")
    )

    text = (
        alt.Chart(plot_df)
        .mark_text(baseline="middle", fontSize=11)
        .encode(
            x=alt.X("Model:N", sort=models),
            y=alt.Y("Metric:N"),
            text=alt.Text("Score:Q", format=".1f"),
            color=alt.condition(
                alt.datum.Score > 5, alt.value("white"), alt.value("black")
            ),
        )
    )

    res = (
        (chart + text)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_2_performance_by_group(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 2: Performance by extraction group.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 2: Performance by extraction group")

    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]

        for q in range(1, 6):
            # ---- Accuracy ----
            acc_vals = model_data[f"Q-{q}-a-3"]
            acc_vals = acc_vals[acc_vals > 0]

            # ---- Detail ----
            detail_vals = model_data[f"Q-{q}-b-2"]
            detail_vals = detail_vals[detail_vals > 0]

            # ---- Completeness ----
            comp_vals = model_data[f"Q-{q}-c-2"]
            comp_vals = comp_vals[comp_vals > 0]

            data.extend(
                [
                    {
                        "Model": model,
                        "Group": f"Q-{q}",
                        "Metric": "Accuracy",
                        "Mean": np.mean(acc_vals),
                        "SE": stats.sem(acc_vals),
                    },
                    {
                        "Model": model,
                        "Group": f"Q-{q}",
                        "Metric": "Detail",
                        "Mean": np.mean(detail_vals),
                        "SE": stats.sem(detail_vals),
                    },
                    {
                        "Model": model,
                        "Group": f"Q-{q}",
                        "Metric": "Completeness",
                        "Mean": np.mean(comp_vals),
                        "SE": stats.sem(comp_vals),
                    },
                ]
            )

    plot_df = pd.DataFrame(data)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Group:N", title="Extraction Group"),
            y=alt.Y("Mean:Q", title="Mean Score", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("Model:N", title="Model"),
            xOffset="Model:N",
            tooltip=["Model", "Group", "Metric", alt.Tooltip("Mean:Q", format=".2f")],
        )
        .properties(width=150, height=200)
    )

    error_bars = (
        alt.Chart(plot_df)
        .mark_errorbar()
        .encode(x=alt.X("Group:N"), y=alt.Y("Mean:Q"), yError="SE:Q", xOffset="Model:N")
    )

    res = (
        alt.layer(chart, error_bars)
        .facet(row=alt.Row("Metric:N", title="Assessment Dimension"))
        .properties(title="Performance by Extraction Group")
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_3_model_ranking(df: pd.DataFrame, verbose: bool = False) -> alt.Chart:
    """Create Figure 3: Model ranking across metrics.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 3: Model ranking")

    openai_models = ["o4-mini", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini"]
    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]
        family = "OpenAI" if model in openai_models else "Local"

        # ---- Accuracy ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy = model_data[acc_cols].values.flatten()
        accuracy = accuracy[accuracy > 0]

        # ---- Detail ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail = model_data[detail_cols].values.flatten()
        detail = detail[detail > 0]

        # ---- Completeness ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        completeness = model_data[comp_cols].values.flatten()
        completeness = completeness[completeness > 0]

        data.extend(
            [
                {
                    "Model": model,
                    "Metric": "Accuracy",
                    "Mean": np.mean(accuracy),
                    "SD": np.std(accuracy, ddof=1),
                    "Family": family,
                },
                {
                    "Model": model,
                    "Metric": "Detail",
                    "Mean": np.mean(detail),
                    "SD": np.std(detail, ddof=1),
                    "Family": family,
                },
                {
                    "Model": model,
                    "Metric": "Completeness",
                    "Mean": np.mean(completeness),
                    "SD": np.std(completeness, ddof=1),
                    "Family": family,
                },
            ]
        )

    plot_df = pd.DataFrame(data)

    chart = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X("Mean:Q", title="Mean Score", scale=alt.Scale(domain=[0, 10])),
            y=alt.Y(
                "Model:N",
                sort=alt.EncodingSortField(field="Mean", op="mean", order="descending"),
                title="Model",
            ),
            color=alt.Color("Family:N", title="Model Family"),
            tooltip=[
                "Model",
                "Metric",
                alt.Tooltip("Mean:Q", format=".2f"),
                alt.Tooltip("SD:Q", format=".2f"),
            ],
        )
        .properties(width=250, height=300)
    )

    error_bars = (
        alt.Chart(plot_df)
        .mark_errorbar()
        .encode(
            x=alt.X("Mean:Q"),
            y=alt.Y(
                "Model:N",
                sort=alt.EncodingSortField(field="Mean", op="mean", order="descending"),
            ),
            xError="SD:Q",
        )
    )

    res = (
        alt.layer(chart, error_bars)
        .facet(column=alt.Column("Metric:N", title="Assessment Dimension"))
        .properties(title="Model Ranking Across Metrics")
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_4_error_completeness(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 4: Error and missing data patterns.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 4: Error and completeness patterns")

    openai_models = ["o4-mini", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini"]
    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]
        family = "OpenAI" if model in openai_models else "Local"

        # ---- Total results ----
        total_cols = ["Q-1-a-1", "Q-2-a-1", "Q-3-a-1", "Q-4-a-1", "Q-5-a-1"]
        total_vals = model_data[total_cols].values.flatten()
        mean_total = np.mean(total_vals)

        # ---- Incorrect results ----
        incorrect_cols = ["Q-1-a-2", "Q-2-a-2", "Q-3-a-2", "Q-4-a-2", "Q-5-a-2"]
        incorrect_vals = model_data[incorrect_cols].values.flatten()
        mean_incorrect = np.mean(incorrect_vals)

        # ---- Missing results ----
        missing_cols = ["Q-1-c-1", "Q-2-c-1", "Q-3-c-1", "Q-4-c-1", "Q-5-c-1"]
        missing_vals = model_data[missing_cols].values.flatten()
        mean_missing = np.mean(missing_vals)

        error_rate = mean_incorrect / mean_total if mean_total > 0 else 0
        completeness_rate = (
            (mean_total - mean_missing) / mean_total if mean_total > 0 else 0
        )

        data.append(
            {
                "Model": model,
                "Error_Rate": error_rate,
                "Completeness_Rate": completeness_rate,
                "Mean_Total": mean_total,
                "Family": family,
            }
        )

    plot_df = pd.DataFrame(data)

    chart = (
        alt.Chart(plot_df)
        .mark_circle()
        .encode(
            x=alt.X(
                "Error_Rate:Q",
                title="Error Rate (Incorrect / Total)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Completeness_Rate:Q",
                title="Completeness Rate ((Total - Missing) / Total)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            size=alt.Size("Mean_Total:Q", title="Mean Total Results"),
            color=alt.Color("Family:N", title="Model Family"),
            tooltip=[
                "Model",
                alt.Tooltip("Error_Rate:Q", format=".3f"),
                alt.Tooltip("Completeness_Rate:Q", format=".3f"),
                alt.Tooltip("Mean_Total:Q", format=".1f"),
            ],
        )
        .properties(width=500, height=400, title="Error Rate vs Completeness Rate")
    )

    text = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=7, fontSize=10)
        .encode(x="Error_Rate:Q", y="Completeness_Rate:Q", text="Model:N")
    )

    res = (
        (chart + text)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_5_score_distributions(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 5: Distribution of scores.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 5: Score distributions")

    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]

        # ---- Accuracy scores ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        for col in acc_cols:
            vals = model_data[col]
            vals = vals[vals > 0]
            for val in vals:
                data.append({"Model": model, "Metric": "Accuracy", "Score": val})

        # ---- Detail scores ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        for col in detail_cols:
            vals = model_data[col]
            vals = vals[vals > 0]
            for val in vals:
                data.append({"Model": model, "Metric": "Detail", "Score": val})

        # ---- Completeness scores ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        for col in comp_cols:
            vals = model_data[col]
            vals = vals[vals > 0]
            for val in vals:
                data.append({"Model": model, "Metric": "Completeness", "Score": val})

    plot_df = pd.DataFrame(data)

    chart = (
        alt.Chart(plot_df)
        .mark_boxplot(size=30)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(width=500, height=200)
    )

    res = (
        chart.facet(row=alt.Row("Metric:N", title="Assessment Dimension"))
        .properties(title="Distribution of Scores by Model and Metric")
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_6_extraction_volume(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 6: Results extraction volume.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 6: Extraction volume")

    openai_models = ["o4-mini", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini"]
    models = sorted(df["model"].unique())
    data = []

    for model in models:
        model_data = df[df["model"] == model]
        family = "OpenAI" if model in openai_models else "Local"

        total_cols = ["Q-1-a-1", "Q-2-a-1", "Q-3-a-1", "Q-4-a-1", "Q-5-a-1"]
        total_vals = model_data[total_cols].values.flatten()

        data.append(
            {
                "Model": model,
                "Mean_Total": np.mean(total_vals),
                "SE": stats.sem(total_vals),
                "Family": family,
            }
        )

    plot_df = pd.DataFrame(data)
    plot_df = plot_df.sort_values("Mean_Total")

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort="-x", title="Model"),
            x=alt.X("Mean_Total:Q", title="Mean Total Results Extracted"),
            color=alt.Color("Family:N", title="Model Family"),
            tooltip=[
                "Model",
                alt.Tooltip("Mean_Total:Q", format=".2f"),
                alt.Tooltip("SE:Q", format=".2f"),
            ],
        )
        .properties(
            width=500, height=300, title="Mean Total Results Extracted by Model"
        )
    )

    error_bars = (
        alt.Chart(plot_df)
        .mark_errorbar()
        .encode(y=alt.Y("Model:N", sort="-x"), x=alt.X("Mean_Total:Q"), xError="SE:Q")
    )

    res = (
        (chart + error_bars)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=14)
    )

    return res


def create_figure_7_openai_vs_local(
    df: pd.DataFrame, verbose: bool = False
) -> alt.Chart:
    """Create Figure 7: OpenAI vs Local models comparison.

    Args:
        df: Assessment DataFrame
        verbose: Print progress if True

    Returns:
        Altair chart object
    """
    if verbose:
        print("\nCreating Figure 7: OpenAI vs Local comparison")

    openai_models = ["o4-mini", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini"]
    local_models = ["llama3", "llama3-2", "deepseek-r1"]

    data = []

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        family = "OpenAI" if model in openai_models else "Local"

        # ---- Accuracy ----
        acc_cols = ["Q-1-a-3", "Q-2-a-3", "Q-3-a-3", "Q-4-a-3", "Q-5-a-3"]
        accuracy = model_data[acc_cols].values.flatten()
        accuracy = accuracy[accuracy > 0]

        # ---- Detail ----
        detail_cols = ["Q-1-b-2", "Q-2-b-2", "Q-3-b-2", "Q-4-b-2", "Q-5-b-2"]
        detail = model_data[detail_cols].values.flatten()
        detail = detail[detail > 0]

        # ---- Completeness ----
        comp_cols = ["Q-1-c-2", "Q-2-c-2", "Q-3-c-2", "Q-4-c-2", "Q-5-c-2"]
        completeness = model_data[comp_cols].values.flatten()
        completeness = completeness[completeness > 0]

        # ---- Overall ----
        overall = np.concatenate([accuracy, detail, completeness])

        for metric_name, scores in [
            ("Accuracy", accuracy),
            ("Detail", detail),
            ("Completeness", completeness),
            ("Overall", overall),
        ]:
            for score in scores:
                data.append(
                    {
                        "Model": model,
                        "Family": family,
                        "Metric": metric_name,
                        "Score": score,
                    }
                )

    plot_df = pd.DataFrame(data)

    # ---- Create faceted chart without layering ----
    chart = (
        alt.Chart(plot_df)
        .mark_boxplot()
        .encode(
            x=alt.X("Family:N", title="Model Family"),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("Family:N", title="Model Family"),
            tooltip=["Family", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(width=150, height=250)
        .facet(column=alt.Column("Metric:N", title="Assessment Dimension"))
        .properties(title="OpenAI vs Local Models Comparison")
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=14)
    )

    return chart


# ==== Summary Report ====


def create_summary_report(
    df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    verbose: bool = False,
) -> str:
    """Create summary report markdown file.

    Args:
        df: Assessment DataFrame
        tables: Dictionary of generated tables
        output_dir: Output directory path
        verbose: Print progress if True

    Returns:
        Markdown string content
    """
    if verbose:
        print("\nCreating summary report")

    # ---- Extract key findings ----
    table_1a = tables["table_1a"]
    top_3_models = table_1a.head(3)

    report = []
    report.append("# LLM Extraction Assessment Results Analysis Report")
    report.append("")
    report.append("This report summarizes the comprehensive analysis of LLM extraction")
    report.append(
        "assessment results from two independent reviewers evaluating 8 LLMs."
    )
    report.append("")

    # ---- Executive Summary ----
    report.append("## Executive Summary")
    report.append("")
    report.append("### Key Findings")
    report.append("")

    top_model = top_3_models.iloc[0]
    report.append(
        f"1. **Best Overall Performance**: {top_model['Model']} "
        f"achieved the highest overall score "
        f"({top_model['Overall_Mean']:.2f} +/- "
        f"{top_model['Overall_SD']:.2f})"
    )

    openai_mean = tables["table_6"][tables["table_6"]["Family"] == "OpenAI"][
        "Mean_Overall"
    ].values[0]
    local_mean = tables["table_6"][tables["table_6"]["Family"] == "Local"][
        "Mean_Overall"
    ].values[0]

    report.append(
        f"2. **Model Family Comparison**: OpenAI models (mean: {openai_mean:.2f}) "
    )
    report.append(
        f"   {'outperformed' if openai_mean > local_mean else 'underperformed'} "
        f"Local models (mean: {local_mean:.2f})"
    )

    table_2 = tables["table_2"]
    lowest_error = table_2.nsmallest(1, "Error_Rate").iloc[0]
    report.append(
        f"3. **Data Quality**: {lowest_error['Model']} "
        f"had the lowest error rate "
        f"({lowest_error['Error_Rate']:.3f})"
    )

    highest_complete = table_2.nlargest(1, "Completeness_Rate").iloc[0]
    report.append(
        f"4. **Completeness**: {highest_complete['Model']} "
        f"achieved the highest completeness rate "
        f"({highest_complete['Completeness_Rate']:.3f})"
    )

    if len(tables["table_4"]) > 0:
        icc_overall = tables["table_4"][tables["table_4"]["Category"] == "Overall"]
        if len(icc_overall) > 0:
            icc_val = icc_overall.iloc[0]["ICC"]
            icc_interp = icc_overall.iloc[0]["Interpretation"]
            report.append(
                f"5. **Inter-Rater Reliability**: {icc_interp} "
                f"agreement (ICC = {icc_val:.3f})"
            )

    report.append("")
    report.append("### Top 3 Performing Models")
    report.append("")

    for idx, row in top_3_models.iterrows():
        report.append(
            f"{idx + 1}. **{row['Model']}**: "
            f"Overall = {row['Overall_Mean']:.2f}, "
            f"Accuracy = {row['Mean_Accuracy']:.2f}, "
            f"Detail = {row['Mean_Detail']:.2f}, "
            f"Completeness = {row['Mean_Completeness']:.2f}"
        )

    report.append("")

    # ---- Overall Model Performance ----
    report.append("## Overall Model Performance")
    report.append("")
    report.append("### Model Rankings")
    report.append("")

    for idx, row in table_1a.iterrows():
        report.append(
            f"{idx + 1}. {row['Model']}: {row['Overall_Mean']:.2f} "
            f"+/- {row['Overall_SD']:.2f}"
        )

    report.append("")

    # ---- Performance by Assessment Dimension ----
    report.append("## Performance by Assessment Dimension")
    report.append("")

    best_acc = table_1a.nlargest(1, "Mean_Accuracy").iloc[0]
    best_detail = table_1a.nlargest(1, "Mean_Detail").iloc[0]
    best_comp = table_1a.nlargest(1, "Mean_Completeness").iloc[0]

    report.append("### Accuracy")
    report.append(f"- Best: {best_acc['Model']} ({best_acc['Mean_Accuracy']:.2f})")
    report.append("")

    report.append("### Detail")
    report.append(f"- Best: {best_detail['Model']} ({best_detail['Mean_Detail']:.2f})")
    report.append("")

    report.append("### Completeness")
    report.append(
        f"- Best: {best_comp['Model']} ({best_comp['Mean_Completeness']:.2f})"
    )
    report.append("")

    # ---- Performance by Extraction Group ----
    report.append("## Performance by Extraction Group")
    report.append("")

    table_1b = tables["table_1b"]
    for q in range(1, 6):
        group_name = {
            1: "Q-1 (Exposure traits)",
            2: "Q-2 (Outcome traits)",
            3: "Q-3 (Analytical methods)",
            4: "Q-4 (Populations)",
            5: "Q-5 (Reported results)",
        }[q]

        acc_col = f"Q-{q}-Accuracy"
        best_model = table_1b.nlargest(1, acc_col).iloc[0]

        report.append(f"### {group_name}")
        report.append(
            f"- Best performer: {best_model['Model']} "
            f"(Accuracy: {best_model[acc_col]:.2f})"
        )
        report.append("")

    # ---- OpenAI vs Local Models ----
    report.append("## OpenAI vs Local Models")
    report.append("")

    table_6 = tables["table_6"]
    openai_row = table_6[table_6["Family"] == "OpenAI"].iloc[0]
    local_row = table_6[table_6["Family"] == "Local"].iloc[0]

    report.append("### OpenAI Models")
    report.append(
        f"- Mean Overall: {openai_row['Mean_Overall']:.2f} "
        f"+/- {openai_row['SD_Overall']:.2f}"
    )
    report.append(
        f"- Mean Accuracy: {openai_row['Mean_Accuracy']:.2f} "
        f"+/- {openai_row['SD_Accuracy']:.2f}"
    )
    report.append(
        f"- Mean Detail: {openai_row['Mean_Detail']:.2f} "
        f"+/- {openai_row['SD_Detail']:.2f}"
    )
    report.append(
        f"- Mean Completeness: {openai_row['Mean_Completeness']:.2f} "
        f"+/- {openai_row['SD_Completeness']:.2f}"
    )
    report.append("")

    report.append("### Local Models")
    report.append(
        f"- Mean Overall: {local_row['Mean_Overall']:.2f} "
        f"+/- {local_row['SD_Overall']:.2f}"
    )
    report.append(
        f"- Mean Accuracy: {local_row['Mean_Accuracy']:.2f} "
        f"+/- {local_row['SD_Accuracy']:.2f}"
    )
    report.append(
        f"- Mean Detail: {local_row['Mean_Detail']:.2f} "
        f"+/- {local_row['SD_Detail']:.2f}"
    )
    report.append(
        f"- Mean Completeness: {local_row['Mean_Completeness']:.2f} "
        f"+/- {local_row['SD_Completeness']:.2f}"
    )
    report.append("")

    stat_row = table_6[table_6["Family"] == "Statistical_Comparison"].iloc[0]
    report.append("### Statistical Comparison")
    report.append(f"- P-value: {stat_row['Mean_Overall']:.4f}")
    report.append(f"- Effect size (Cohen's d): {stat_row['SD_Overall']:.3f}")
    report.append("")

    # ---- Error and Data Quality Analysis ----
    report.append("## Error and Data Quality Analysis")
    report.append("")

    table_2_sorted = table_2.sort_values("Error_Rate")
    report.append("### Error Rates (Low to High)")
    report.append("")

    for idx, row in table_2_sorted.iterrows():
        report.append(
            f"- {row['Model']}: {row['Error_Rate']:.3f} "
            f"({row['Mean_Incorrect']:.1f} / {row['Mean_Total']:.1f})"
        )

    report.append("")

    # ---- Statistical Findings ----
    report.append("## Statistical Findings")
    report.append("")

    table_3 = tables["table_3"]
    sig_comparisons = table_3[table_3["Significance"] != ""]

    if len(sig_comparisons) > 0:
        report.append("### Significant Pairwise Differences")
        report.append("")

        for _, row in sig_comparisons.iterrows():
            report.append(
                f"- {row['Model_1']} vs {row['Model_2']}: "
                f"p = {row['P_Value']:.4f} {row['Significance']}, "
                f"d = {row['Cohens_D']:.3f}"
            )

        report.append("")
    else:
        report.append("No statistically significant pairwise differences.")
        report.append("")

    # ---- Inter-Rater Reliability ----
    report.append("## Inter-Rater Reliability")
    report.append("")

    table_4 = tables["table_4"]
    if len(table_4) > 0:
        for _, row in table_4.iterrows():
            report.append(
                f"- {row['Category']}: ICC = {row['ICC']:.3f} "
                f"(95% CI: [{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]), "
                f"{row['Interpretation']}"
            )
    else:
        report.append(
            "ICC calculations could not be completed due to "
            "missing data or unbalanced design."
        )

    report.append("")

    # ---- Recommendations ----
    report.append("## Recommendations")
    report.append("")

    top_model_name = top_3_models.iloc[0]["Model"]
    report.append(
        f"1. **Best Overall Choice**: {top_model_name} "
        "for comprehensive extraction tasks"
    )

    lowest_error_model = table_2.nsmallest(1, "Error_Rate").iloc[0]["Model"]
    report.append(
        f"2. **Highest Accuracy**: {lowest_error_model} "
        "when error minimization is critical"
    )

    highest_volume = table_2.nlargest(1, "Mean_Total").iloc[0]["Model"]
    report.append(
        f"3. **Maximum Extraction**: {highest_volume} for comprehensive data capture"
    )

    report.append(
        "4. **Cost-Performance Trade-off**: Consider local models "
        "for budget-constrained scenarios"
    )

    report.append("")

    res = "\n".join(report)
    return res


# ==== Output Management ====


def save_outputs(
    tables: Dict[str, pd.DataFrame],
    figures: Dict[str, alt.Chart],
    summary_report: str,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Save all outputs to files.

    Args:
        tables: Dictionary of table DataFrames
        figures: Dictionary of figure charts
        summary_report: Summary report markdown string
        output_dir: Output directory path
        dry_run: If True, only preview without saving
        verbose: Print progress if True
    """
    if dry_run:
        if verbose:
            print("\n==== DRY RUN MODE ====")
            print("Would create the following outputs:")
            print(f"\nOutput directory: {output_dir}")
            print(f"\nTables ({len(tables)}):")
            for name in tables.keys():
                print(f"  - {name}.csv")
            print(f"\nFigures ({len(figures)}):")
            for name in figures.keys():
                print(f"  - {name}.json")
                print(f"  - {name}.png")
            print("\nSummary report: summary-report.md")
        return

    # ---- Create directories ----
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving outputs to: {output_dir}")

    # ---- Save tables ----
    if verbose:
        print("\nSaving tables...")

    for name, table in tables.items():
        output_path = tables_dir / f"{name}.csv"
        table.to_csv(output_path, index=False)
        if verbose:
            print(f"  - {name}.csv ({len(table)} rows, {len(table.columns)} cols)")

    # ---- Save figures ----
    if verbose:
        print("\nSaving figures...")

    for name, chart in figures.items():
        json_path = figures_dir / f"{name}.json"
        png_path = figures_dir / f"{name}.png"

        chart.save(str(json_path))

        try:
            chart.save(str(png_path), scale_factor=2.0)
            if verbose:
                print(f"  - {name}.json and {name}.png")
        except Exception as e:
            if verbose:
                print(f"  - {name}.json (PNG export failed: {e})")

    # ---- Save summary report ----
    if verbose:
        print("\nSaving summary report...")

    summary_path = output_dir / "summary-report.md"
    summary_path.write_text(summary_report)

    if verbose:
        print(f"  - summary-report.md")


# ==== Main Workflow ====


def main():
    """Main analysis workflow."""
    args = parse_args()

    if args.verbose:
        print("==== LLM Assessment Results Analysis ====\n")
        print(f"Input file: {args.input_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Dry run: {args.dry_run}")
        print("")

    # ---- Load data ----
    try:
        df = load_and_validate_data(args.input_file, args.verbose)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # ---- Create tables ----
    if args.verbose:
        print("\n==== Creating Tables ====")

    tables = {}

    try:
        tables["table-1a-overall-performance"] = create_table_1a_overall_performance(
            df, args.verbose
        )
        tables["table-1b-performance-by-group"] = create_table_1b_performance_by_group(
            df, args.verbose
        )
        tables["table-2-error-rates"] = create_table_2_error_rates(df, args.verbose)
        tables["table-3-pairwise-comparisons"] = create_table_3_pairwise_comparisons(
            df, args.verbose
        )
        tables["table-4-inter-rater-reliability"] = (
            create_table_4_inter_rater_reliability(df, args.verbose)
        )
        tables["table-5-model-insights"] = create_table_5_model_insights(
            df, args.verbose
        )
        tables["table-6-model-family-comparison"] = (
            create_table_6_model_family_comparison(df, args.verbose)
        )
    except Exception as e:
        print(f"Error creating tables: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Create figures ----
    if args.verbose:
        print("\n==== Creating Figures ====")

    figures = {}

    try:
        figures["fig-1-overall-heatmap"] = create_figure_1_overall_heatmap(
            df, args.verbose
        )
        figures["fig-2-performance-by-group"] = create_figure_2_performance_by_group(
            df, args.verbose
        )
        figures["fig-3-model-ranking"] = create_figure_3_model_ranking(df, args.verbose)
        figures["fig-4-error-completeness"] = create_figure_4_error_completeness(
            df, args.verbose
        )
        figures["fig-5-score-distributions"] = create_figure_5_score_distributions(
            df, args.verbose
        )
        figures["fig-6-extraction-volume"] = create_figure_6_extraction_volume(
            df, args.verbose
        )
        figures["fig-7-openai-vs-local"] = create_figure_7_openai_vs_local(
            df, args.verbose
        )
    except Exception as e:
        print(f"Error creating figures: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Create summary report ----
    try:
        summary_report = create_summary_report(
            df,
            {
                "table_1a": tables["table-1a-overall-performance"],
                "table_1b": tables["table-1b-performance-by-group"],
                "table_2": tables["table-2-error-rates"],
                "table_3": tables["table-3-pairwise-comparisons"],
                "table_4": tables["table-4-inter-rater-reliability"],
                "table_5": tables["table-5-model-insights"],
                "table_6": tables["table-6-model-family-comparison"],
            },
            args.output_dir,
            args.verbose,
        )
    except Exception as e:
        print(f"Error creating summary report: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Save all outputs ----
    try:
        save_outputs(
            tables, figures, summary_report, args.output_dir, args.dry_run, args.verbose
        )
    except Exception as e:
        print(f"Error saving outputs: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        print("\n==== Analysis Complete ====")

    if not args.dry_run:
        print(f"\nOutputs saved to: {args.output_dir}")
        print(f"  - {len(tables)} tables in tables/ subdirectory")
        print(f"  - {len(figures)} figures in figures/ subdirectory")
        print("  - summary-report.md in output directory")


if __name__ == "__main__":
    main()
