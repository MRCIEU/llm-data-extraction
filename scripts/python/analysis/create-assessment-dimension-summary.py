"""Create a standalone study-cluster uncertainty prototype from fixed scores."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ==== Fixed study design ====

MODELS = (
    "deepseek-r1", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini",
    "llama3", "llama3-2", "o4-mini",
)
# Plot order is independent of the statistical target-array order above.
DISPLAY_MODELS = (
    "gpt-5-mini", "gpt-5", "gpt-4-1", "o4-mini", "llama3-2",
    "gpt-4o", "deepseek-r1", "llama3",
)
DIMENSIONS = {"Accuracy": "a-3", "Detail": "b-2", "Completeness": "c-2"}
REVIEWERS = ("Reviewer 1", "Reviewer 2")
SCORE_COLUMNS = [
    f"Q-{q}-{suffix}" for suffix in DIMENSIONS.values() for q in range(1, 6)
]
INPUT = Path("data/artifacts/assessment-results/assessment-results-numeric.csv")
AUDIT = INPUT.with_name("assessment-score-exclusions.csv")
OUTPUT = Path("data/artifacts/assessment-dimension-summary")
BASENAME = "assessment-dimension-summary"
COLORS = ("#689d6a", "#b16286", "#d79921")
HEATMAP = Path("data/artifacts/manuscript-figures/llm-performance.png")
COMBINED = "assessment-combined"
WIDTH = 6.175
COMPACT_HEIGHT = 2.2
COMBINED_HEIGHT = 5.77
HEATMAP_ROWS = 1005
HEATMAP_HEIGHT = 3.35


# ==== Input validation ====


def validate_tagged_data(df: pd.DataFrame) -> None:
    """Require the complete study/model/reviewer grid after reviewer tagging."""
    required = {"pmid", "model", "reviewer", *SCORE_COLUMNS}
    if not required.issubset(df.columns):
        raise ValueError("Missing required assessment columns")
    keys = ["pmid", "model", "reviewer"]
    if len(df) != 1600 or df[keys].isna().any().any():
        raise ValueError("Expected 1600 rows with nonmissing study keys")
    if df.pmid.astype(str).str.strip().eq("").any():
        raise ValueError("Study identifiers must not be blank or whitespace")
    if set(df.model) != set(MODELS) or set(df.reviewer) != set(REVIEWERS):
        raise ValueError("Unexpected model or reviewer identifiers")
    if df.pmid.nunique() != 100 or df.duplicated(keys).any():
        raise ValueError("Expected 100 studies and unique study/model/reviewer keys")
    expected = pd.MultiIndex.from_product(
        [df.pmid.unique(), MODELS, REVIEWERS], names=keys
    )
    actual = pd.MultiIndex.from_frame(df[keys])
    if len(expected.difference(actual)):
        raise ValueError("Incomplete study/model/reviewer pairing")
    values = df[SCORE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 10)).any():
        raise ValueError("Processed scores must be missing or within 0-10")


def load_assessments(path: Path) -> pd.DataFrame:
    """Tag only the original fixed 800/800 partition, then validate keys."""
    df = pd.read_csv(path, dtype={"pmid": str})
    if len(df) != 1600 or "reviewer" in df.columns:
        raise ValueError("Expected untagged original 1600-row input")
    df["reviewer"] = np.repeat(REVIEWERS, 800)
    validate_tagged_data(df)
    return df


def validate_exclusions(df: pd.DataFrame, audit_path: Path) -> None:
    """Verify the settled 16 excluded Q5 cells remain missing by exact key."""
    audit = pd.read_csv(audit_path, dtype={"pmid": str})
    columns = {"reviewer", "pmid", "model", "question", "reason", "raw_value"}
    if not columns.issubset(audit.columns):
        raise ValueError("Missing exclusion audit columns")
    expected = {
        ("Reviewer 2", pmid, model, "Q-5-b-2")
        for pmid in ("38293563", "34025845") for model in MODELS
    }
    keys = ["reviewer", "pmid", "model", "question"]
    if len(audit) != 16 or set(audit[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Unexpected exclusion audit keys")
    if not audit.reason.eq("raw_score_outside_1_10").all():
        raise ValueError("Unexpected exclusion audit reason")
    indexed = df.set_index(["reviewer", "pmid", "model"])
    for reviewer, pmid, model, question in expected:
        if not pd.isna(indexed.loc[(reviewer, pmid, model), question]):
            raise ValueError("Previously excluded score is no longer missing")


# ==== Estimation and cluster resampling ====


def cluster_totals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Return study by target sums/counts, retaining zero-contribution studies."""
    validate_tagged_data(df)
    studies = sorted(df.pmid.unique())
    sums = np.zeros((len(studies), len(MODELS) * len(DIMENSIONS)))
    counts = np.zeros_like(sums, dtype=np.int64)
    for m, model in enumerate(MODELS):
        subset = df.loc[df.model.eq(model)].set_index("pmid")
        for d, suffix in enumerate(DIMENSIONS.values()):
            values = subset[[f"Q-{q}-{suffix}" for q in range(1, 6)]]
            eligible = values.gt(0) & values.notna()
            target = m * len(DIMENSIONS) + d
            sums[:, target] = (
                values.where(eligible, 0).sum(axis=1)
                .groupby(level=0).sum().reindex(studies).to_numpy()
            )
            counts[:, target] = (
                eligible.sum(axis=1).groupby(level=0).sum()
                .reindex(studies).to_numpy()
            )
    return sums, counts, studies


def resample_ratios(
    sums: np.ndarray, counts: np.ndarray, draws: np.ndarray,
) -> np.ndarray:
    """Apply shared study draws with multiplicity to every sum/count ratio."""
    numerator = sums[draws].sum(axis=1)
    denominator = counts[draws].sum(axis=1)
    return np.divide(
        numerator, denominator,
        out=np.full(numerator.shape, np.nan), where=denominator > 0,
    )


def summarise(
    sums: np.ndarray, counts: np.ndarray, replicates: int = 10000,
    seed: int = 20261010, confidence: float = 0.95,
) -> pd.DataFrame:
    """Keep observed ratios; withhold CIs if any replicate is undefined."""
    if replicates < 1 or not 0 < confidence < 1 or seed < 0:
        raise ValueError("Require positive replicates, nonnegative seed and 0<CI<1")
    rng = np.random.Generator(np.random.PCG64(seed))
    # One common draw matrix; batching bounds temporary score-array memory.
    draws = rng.integers(0, len(sums), size=(replicates, len(sums)))
    boot = np.concatenate([
        resample_ratios(sums, counts, draws[start:start + 250])
        for start in range(0, replicates, 250)
    ])
    alpha = (1 - confidence) / 2
    records = []
    for target, (model, dimension) in enumerate(
        (m, d) for m in MODELS for d in DIMENSIONS
    ):
        n = int(counts[:, target].sum())
        point = float(sums[:, target].sum() / n) if n else np.nan
        undefined = int(np.isnan(boot[:, target]).sum())
        lower = upper = np.nan
        status = "no_eligible_scores" if not n else "undefined_replicates"
        if not undefined:
            lower, upper = np.quantile(
                boot[:, target], [alpha, 1 - alpha], method="linear"
            )
            status = "degenerate" if lower == upper else "ok"
        records.append({
            "model": model, "dimension": dimension, "mean": point,
            "lower": lower, "upper": upper, "eligible_cells": n,
            "contributing_studies": int((counts[:, target] > 0).sum()),
            "undefined_replicates": undefined, "status": status,
        })
    return pd.DataFrame(records)


# ==== Figure and reproducible output ====


def create_figure(
    summary: pd.DataFrame, confidence: float, heatmap: Path | None = None,
) -> plt.Figure:
    """Draw compact intervals, optionally below a full-width heatmap raster."""
    if not summary.status.isin(["ok", "degenerate"]).all():
        raise ValueError("Cannot plot undefined intervals; inspect summary")
    height = COMBINED_HEIGHT if heatmap is not None else COMPACT_HEIGHT
    raster = plt.imread(heatmap) if heatmap is not None else None
    if raster is not None:
        if raster.shape[:2] != (1095, 1852):
            raise ValueError("Expected native 300 DPI heatmap")
        if not np.all(raster[HEATMAP_ROWS:, :, :3] == 1):
            raise ValueError("Heatmap crop would remove nonwhite content")
        raster = raster[:HEATMAP_ROWS]
    fig = plt.figure(figsize=(WIDTH, height))
    if raster is not None:
        ax = fig.add_axes([
            0, COMPACT_HEIGHT / height, 1, HEATMAP_HEIGHT / height,
        ])
        ax.imshow(raster, aspect="auto", interpolation="none")
        ax.set_axis_off()
        fig.text(
            0.015, (height - 0.05) / height,
            "A  Question groups",
            fontsize=8.5, va="top",
        )
    axes = []
    panel_stride = (0.98 - 0.19 + 0.045) / 3
    panel_width = 0.18
    for index in range(3):
        axes.append(fig.add_axes([
            0.19 + index * panel_stride, 0.43 / height,
            panel_width, 1.17 / height,
        ]))
    low = float(summary[["mean", "lower"]].min().min())
    high = float(summary[["mean", "upper"]].max().max())
    pad = max(0.1, (high - low) * 0.08)
    for index, (ax, dimension, color) in enumerate(
        zip(axes, DIMENSIONS, COLORS)
    ):
        rows = summary.loc[summary.dimension.eq(dimension)].set_index("model")
        rows = rows.loc[list(DISPLAY_MODELS)]
        y = np.arange(len(DISPLAY_MODELS))
        ax.hlines(y, rows.lower, rows.upper, color=color, linewidth=1.25)
        ax.scatter(rows["mean"], y, color=color, s=15, zorder=3)
        ax.set_title(dimension, fontsize=8.5, pad=4)
        ax.annotate(
            "Mean", (1.29, 1), xycoords="axes fraction",
            xytext=(0, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, annotation_clip=False,
        )
        for row, mean in enumerate(rows["mean"]):
            ax.text(
                1.29, row, f"{mean:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8,
                fontfamily="DejaVu Sans Mono", clip_on=False,
                gid="mean-label",
            )
        ax.set_xlim(low - pad, high + pad)
        ticks = np.unique(np.round(np.linspace(low, high, 3), 1))
        ax.set_xticks(ticks, [f"{tick:.1f}" for tick in ticks])
        ax.set_yticks(y, DISPLAY_MODELS, fontsize=8)
        ax.tick_params(axis="y", labelleft=index == 0, length=2)
        ax.tick_params(axis="x", labelsize=8, pad=2)
        ax.set_xlabel("Mean score", fontsize=8, labelpad=2)
        ax.set_ylim(7.5, -0.5)
        ax.grid(axis="x", color="#ebdbb2", alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    prefix = "B  " if heatmap is not None else ""
    fig.text(
        0.015, 2.12 / height, prefix + "Overall assessment dimensions",
        fontsize=9, va="top",
    )
    fig.text(
        0.015, 1.94 / height,
        f"{confidence:.0%} percentile study-cluster intervals; models ordered by pooled overall score",
        fontsize=8, va="top",
    )
    return fig


def file_identity(path: Path) -> dict:
    """Record exact bytes and source path without embedding input records."""
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def method_note(args: argparse.Namespace) -> str:
    """Describe the approved estimand, reproducibility and interpretation."""
    return (
        "# Assessment dimension summary prototype\n\n"
        "For each model and dimension, the point estimate is the sum of positive, nonmissing processed scores across five question groups and both observed reviewers divided by the eligible cell count.\n"
        "This available-cell weighting preserves the existing score eligibility rule and gives greater weight to studies, reviewers and groups with more eligible cells.\n"
        "The original reviewer halves are tagged before matching the complete 100-study by eight-model by two-reviewer grid.\n"
        "The 16 audited Q5 Detail exclusions remain missing, and processed completeness scores are not transformed again.\n\n"
        f"Intervals are {args.confidence:.0%} percentile study-cluster bootstrap intervals from {args.replicates:,} replicates, using NumPy Generator(PCG64), seed {args.seed}, and numpy.quantile(method='linear').\n"
        "Each replicate draws 100 study identifiers with replacement from the sorted study list, with the same draws used for all 24 targets.\n"
        "Each selected study retains both reviewers, all question cells and their missingness, including studies contributing no eligible cells to a target.\n"
        "Repeated draws retain their multiplicities and each replicate recomputes the sum/count ratio.\n"
        "Displayed points are the observed estimates, not bootstrap averages.\n"
        "The CSV reports eligible cells, contributing studies, undefined replicate counts and interval status.\n"
        "If any replicate for a target is undefined, its interval is withheld without dropping replicates and figure export fails explicitly after saving the diagnostic summary.\n"
        "Equal endpoints are reported as degenerate intervals.\n\n"
        "The marginal intervals describe study-sampling variation conditional on the two observed reviewers, fixed extraction outputs and the existing missing-score rule.\n"
        "They do not measure variation across reviewers or repeated model runs, correct assessment bias, or resolve missingness assumptions.\n"
        "Interval overlap is not a formal paired model comparison.\n"
        "The display uses the fixed descending pooled overall-score order from the heatmap; this order is separate from the unchanged alphabetical statistical target-array order and is not a formal model comparison.\n"
        "Manuscript integration and final caption/page checks are handled separately.\n\n"
        "## Compact and combined figures\n\n"
        "The compact uncertainty panel is 6.175 by 2.2 inches with text at least 8 points.\n"
        "The right-hand Mean columns display the 24 observed means to two decimal places, aligned with their model rows; interval endpoints are shown graphically without numeric labels.\n"
        "The combined figure is 6.175 by 5.77 inches: panel A embeds the existing heatmap at its full width and original content scale, while panel B displays the same 24 estimates and intervals in the same descending pooled overall-score order as panel A.\n"
        "Only 90 verified all-white bottom rows are cropped from the heatmap display, preserving every original nonwhite pixel and 34 blank rows below its content.\n"
        "Panel A retains descending pooled overall-score order, all cell annotations and the three original dimension palettes.\n"
        "The combined PDF contains the 300 DPI heatmap raster with only its blank bottom margin cropped, and vector uncertainty marks/text, so it is not fully vector.\n"
        "The heatmap input hash, physical geometry and final output hashes are recorded in metadata.json.\n"
        "Use the assessment-combined recipe to render the approved CSV without rerunning bootstrap or rewriting the CSV.\n\n"
        "## Reproduction\n\n```sh\n"
        "just --no-dotenv -f justfile-processing assessment-dimension-summary "
        f"--input-file {args.input_file} --audit-file {args.audit_file} "
        f"--output-dir {args.output_dir} --replicates {args.replicates} "
        f"--seed {args.seed} --confidence {args.confidence}\n```\n"
        "\nInput and script hashes, software versions, study order and output filenames are recorded in metadata.json.\n"
    )


def export_outputs(
    summary: pd.DataFrame, metadata: dict, args: argparse.Namespace,
) -> None:
    """Export compact/composite previews, removing all pairs on failure."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        args.output_dir / f"{name}.{ext}"
        for name in (BASENAME, COMBINED) for ext in ("png", "pdf")
    ]
    fig = None
    try:
        csv_path = args.output_dir / f"{BASENAME}.csv"
        if not getattr(args, "render_only", False):
            summary.to_csv(csv_path, index=False)
        (args.output_dir / "method.md").write_text(method_note(args))
        heatmap = getattr(args, "heatmap_file", HEATMAP)
        for index, source in enumerate((None, heatmap)):
            fig = create_figure(summary, args.confidence, source)
            fig.savefig(images[index * 2], dpi=300)
            if source is not None:
                # Preserve native raster pixels despite half-pixel canvas rounding.
                with Image.open(images[index * 2]) as canvas:
                    with Image.open(source) as original:
                        if original.size != (1852, 1095):
                            raise ValueError("Expected native 300 DPI heatmap")
                        canvas.paste(original.crop((0, 0, 1852, HEATMAP_ROWS)), (0, 66))
                    canvas.save(images[index * 2], dpi=(300, 300))
            fig.savefig(images[index * 2 + 1])
            plt.close(fig)
            fig = None
        metadata.update({
            "source": file_identity(Path(__file__)),
            "embedded_heatmap": file_identity(heatmap),
            "summary": file_identity(csv_path),
            "plot_model_order": DISPLAY_MODELS,
            "layout": {
                "width_inches": WIDTH,
                "compact_height_inches": COMPACT_HEIGHT,
                "combined_height_inches": COMBINED_HEIGHT,
                "heatmap_inches": [WIDTH, HEATMAP_HEIGHT],
                "heatmap_source_pixels": [1852, 1095],
                "heatmap_display_crop": [0, 0, 1852, HEATMAP_ROWS],
                "removed_margin": "90 bottom rows, verified all-white RGB",
                "png_dpi": 300, "minimum_lower_text_points": 8,
                "mean_labels": "right-aligned RHS columns; two decimals",
                "mean_label_font": "DejaVu Sans Mono, 8 points",
                "pdf_components": "heatmap raster with blank-margin crop; vector intervals",
                "panel_A_order": "descending pooled overall score",
                "panel_B_order": "descending pooled overall score",
            },
            "outputs": [p.name for p in images]
                       + [csv_path.name, "method.md", "metadata.json"],
            "output_identities": [
                file_identity(p) for p in images
            ] + [file_identity(args.output_dir / "method.md")],
        })
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
    except Exception:
        for path in images:
            path.unlink(missing_ok=True)
        raise
    finally:
        if fig is not None:
            plt.close(fig)


# ==== CLI ====


def parse_args() -> argparse.Namespace:
    """Parse explicit input, resampling and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- --input-file ----
    parser.add_argument("--input-file", type=Path, default=INPUT)
    # ---- --audit-file ----
    parser.add_argument("--audit-file", type=Path, default=AUDIT)
    # ---- --output-dir ----
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    # ---- --replicates ----
    parser.add_argument("--replicates", type=int, default=10000)
    # ---- --seed ----
    parser.add_argument("--seed", type=int, default=20261010)
    # ---- --confidence ----
    parser.add_argument("--confidence", type=float, default=0.95)
    # ---- --render-only ----
    parser.add_argument("--render-only", action="store_true")
    # ---- --heatmap-file ----
    parser.add_argument("--heatmap-file", type=Path, default=HEATMAP)
    # ---- --dry-run ----
    parser.add_argument("-n", "--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate fixed input and produce isolated scientific artifacts."""
    args = parse_args()
    if args.output_dir.resolve() != OUTPUT.resolve():
        # Alternative dedicated directories are allowed; canonical owners are not.
        protected = [INPUT.parent, Path("data/artifacts/manuscript-figures"),
                     Path("data/artifacts/manuscript-tables")]
        if any(args.output_dir.resolve().is_relative_to(p.resolve()) for p in protected):
            raise ValueError("Output must not target canonical input/artifact directories")
    df = load_assessments(args.input_file)
    validate_exclusions(df, args.audit_file)
    if args.replicates < 1 or args.seed < 0 or not 0 < args.confidence < 1:
        raise ValueError("Invalid resampling options")
    if args.dry_run:
        action = (
            "retain approved CSV without bootstrap" if getattr(
                args, "render_only", False
            ) else "compute bootstrap and write CSV"
        )
        print(
            f"Validated input; would {action}, render compact/combined "
            f"PNG/PDF pairs and method/metadata in {args.output_dir}"
        )
        return
    if getattr(args, "render_only", False):
        metadata = json.loads((args.output_dir / "metadata.json").read_text())
        for key, path in (("input", args.input_file), ("exclusions", args.audit_file)):
            if metadata[key]["sha256"] != file_identity(path)["sha256"]:
                raise ValueError("Stored input identity does not match")
        for key in ("replicates", "seed", "confidence"):
            if metadata[key] != getattr(args, key):
                raise ValueError("Render options must match approved statistics")
        csv_path = args.output_dir / f"{BASENAME}.csv"
        if "summary" in metadata:
            if metadata["summary"]["sha256"] != file_identity(csv_path)["sha256"]:
                raise ValueError("Approved summary identity does not match")
        summary = pd.read_csv(csv_path, float_precision="round_trip")
        export_outputs(summary, metadata, args)
        return
    sums, counts, studies = cluster_totals(df)
    summary = summarise(sums, counts, args.replicates, args.seed, args.confidence)
    metadata = {
        "input": file_identity(args.input_file),
        "exclusions": file_identity(args.audit_file),
        "source": file_identity(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "estimand": "pooled positive nonmissing cell sum / eligible cell count",
        "cluster": "study", "study_order": studies, "model_order": MODELS,
        "dimension_order": list(DIMENSIONS), "reviewer_partition": [800, 800],
        "replicates": args.replicates, "seed": args.seed,
        "rng": "numpy.random.Generator(PCG64)", "confidence": args.confidence,
        "interval": "percentile", "quantile_method": "linear",
        "shared_draws": True, "undefined_policy": "withhold CI if any undefined",
        "status_counts": summary.status.value_counts().to_dict(),
        "undefined_replicates_total": int(summary.undefined_replicates.sum()),
        "outputs": [f"{BASENAME}.{ext}" for ext in ("png", "pdf", "csv")]
                   + ["method.md", "metadata.json"],
    }
    export_outputs(summary, metadata, args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
