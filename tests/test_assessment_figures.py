"""Presentation-only heatmap checks with no validation/table side effects."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heatmap_assets",
    ROOT / "scripts/python/analysis/create-manuscript-assets.py",
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


@pytest.mark.parametrize("dry_run", [False, True])
def test_heatmap_only_never_accesses_validation_or_tables(
    tmp_path, monkeypatch, dry_run
):
    args = SimpleNamespace(
        assessment_file=tmp_path / "input.csv",
        heatmap_only=True,
        assessment_only=False,
        aggregated_dir=tmp_path / "must-not-read",
        output_figures=tmp_path / "figures",
        output_tables=tmp_path / "must-not-create",
        dry_run=dry_run,
        verbose=False,
    )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Heatmap-only path accessed validation or tables")

    monkeypatch.setattr(assets, "parse_args", lambda: args)
    monkeypatch.setattr(assets, "load_assessment_data", lambda *a: None)
    monkeypatch.setattr(assets, "compute_group_metrics", lambda *a: None)
    monkeypatch.setattr(assets, "compute_overall_metrics", lambda *a: None)
    monkeypatch.setattr(assets, "load_validation_metrics", forbidden)
    monkeypatch.setattr(assets, "create_table_performance", forbidden)
    monkeypatch.setattr(assets, "create_table_validation", forbidden)
    monkeypatch.setattr(
        assets, "create_combined_figure", lambda *a: calls.append(a[3])
    )
    assets.main()
    assert calls == (
        [] if dry_run else [args.output_figures / "llm-performance.png"]
    )
    assert args.output_figures.exists() is not dry_run
    assert not args.output_tables.exists()


def test_heatmap_preserves_exact_data_order_encoding_and_physical_size(
    tmp_path, monkeypatch
):
    df = assets.load_assessment_data(
        ROOT / "data/artifacts/assessment-results/assessment-results-numeric.csv"
    )
    groups = assets.compute_group_metrics(df)
    overall = assets.compute_overall_metrics(df)
    original_groups = groups.copy(deep=True)
    original_overall = overall.copy(deep=True)
    saved = []
    original_save = Figure.savefig

    def capture(fig, *args, **kwargs):
        saved.append(fig)
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "llm-performance.png"
    assets.create_combined_figure(groups, overall, pd.DataFrame(), output)
    assert output.is_file() and output.stat().st_size > 0
    pd.testing.assert_frame_equal(groups, original_groups)
    pd.testing.assert_frame_equal(overall, original_overall)
    fig = saved[0]
    np.testing.assert_array_equal(fig.get_size_inches(), [6.175, 3.65])
    assert len(fig.axes) == 4
    ax, *bar_axes = fig.axes
    colorbars = [bar_ax._colorbar for bar_ax in bar_axes]
    assert all(cb.norm is colorbars[0].norm for cb in colorbars)
    assert (colorbars[0].norm.vmin, colorbars[0].norm.vmax) == (6, 10)
    original_stops = [
        [
            (0.0, "#282828"), (0.7, "#1d4540"), (0.8, "#2a5a4a"),
            (0.85, "#3a7058"), (0.9, "#4a8568"), (0.93, "#5a9a70"),
            (0.96, "#6aaf78"), (0.98, "#7ac480"), (1.0, "#8ec07c"),
        ],
        [
            (0.0, "#282828"), (0.5, "#4a2845"), (0.7, "#8f3f71"),
            (0.8, "#9a4a7c"), (0.85, "#a55687"), (0.9, "#b16286"),
            (0.95, "#c275a0"), (1.0, "#d3869b"),
        ],
        [
            (0.0, "#282828"), (0.5, "#5a3c0a"), (0.7, "#b57614"),
            (0.8, "#c4851a"), (0.85, "#cc8f1d"), (0.9, "#d79921"),
            (0.95, "#e8b528"), (1.0, "#fabd2f"),
        ],
    ]
    for cb, stops, dimension in zip(
        colorbars, original_stops, ["accuracy", "detail", "completeness"]
    ):
        assert cb.cmap.name == f"gruvbox_{dimension}"
        positions = np.array([position for position, _ in stops])
        anchors = np.array([to_rgb(color) for _, color in stops])
        # Independently interpolate all 256 original-ramp positions 0.6-1.
        expected = np.column_stack([
            np.interp(np.linspace(0.6, 1, 256), positions, anchors[:, c])
            for c in range(3)
        ])
        colors = cb.cmap(np.linspace(0, 1, 256))[:, :3]
        np.testing.assert_allclose(colors, expected, rtol=0, atol=1e-14)
        # Preserve original anchors, including any luminance nonmonotonicity.
        # Annotation color is checked independently below.
    assert [bar_ax.get_xlabel() for bar_ax in bar_axes] == ["A", "D", "C"]
    assert len(bar_axes[0].get_yticks()) == 0
    assert len(bar_axes[1].get_yticks()) == 0
    np.testing.assert_array_equal(bar_axes[2].get_yticks(), [6, 7, 8, 9, 10])
    assert fig.texts[1].get_text() == "Mean score"
    order = overall.sort_values("overall_mean", ascending=False)[
        "model"
    ].tolist()
    metrics = [
        f"Q-{q}-{dimension}"
        for q in range(1, 6)
        for dimension in ["Accuracy", "Detail", "Completeness"]
    ]
    means = groups.pivot_table(
        index="model", columns="metric", values="mean_score"
    ).reindex(index=order, columns=metrics).to_numpy().ravel()
    assert len(ax.patches) == 120
    assert len(ax.texts) == 125
    for index, (patch, annotation, mean) in enumerate(
        zip(ax.patches, ax.texts[:120], means)
    ):
        colorbar = colorbars[index % 3]
        np.testing.assert_array_equal(
            patch.get_facecolor(), colorbar.cmap(colorbar.norm(mean))
        )
        assert annotation.get_text() == f"{mean:.1f}"
        assert annotation.get_fontsize() == 8.5
        assert annotation.get_color() == "black"
    assert [t.get_text() for t in ax.get_yticklabels()] == order[::-1]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "D", "C"] * 5
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())
    assert ax.texts[-1].get_text() == "Q5\nMR findings"
    assert fig.texts[0].get_text() == (
        "A: Accuracy    D: Detail    C: Completeness"
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headings = [t.get_window_extent(renderer) for t in ax.texts[-5:]]
    for left, right in zip(headings, headings[1:]):
        assert left.x1 < right.x0
    texts = (
        ax.texts
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + [t for bar_ax in bar_axes for t in bar_ax.get_yticklabels()]
        + fig.texts
        + [ax.title]
        + [bar_ax.xaxis.label for bar_ax in bar_axes]
    )
    for text in texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width
        assert bounds.y1 <= fig.bbox.height
    assert not plt.fignum_exists(fig.number)


def test_export_failure_propagates_and_closes_figure(tmp_path, monkeypatch):
    df = assets.load_assessment_data(
        ROOT / "data/artifacts/assessment-results/assessment-results-numeric.csv"
    )
    before = plt.get_fignums()

    def fail_save(*args, **kwargs):
        raise OSError("export unavailable")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="export unavailable"):
        assets.create_combined_figure(
            assets.compute_group_metrics(df),
            assets.compute_overall_metrics(df),
            pd.DataFrame(),
            tmp_path / "llm-performance.png",
        )
    assert plt.get_fignums() == before
