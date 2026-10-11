"""Standalone cluster estimation, input protection and real export checks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dimension_summary",
    ROOT / "scripts/python/analysis/create-assessment-dimension-summary.py",
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


@pytest.fixture
def data():
    """Load corrected data without mutating input or invoking processing."""
    return assets.load_assessments(ROOT / assets.INPUT)


def test_original_partition_and_shuffled_tagged_pairing(data):
    assert data.reviewer.iloc[:800].eq("Reviewer 1").all()
    assert data.reviewer.iloc[800:].eq("Reviewer 2").all()
    original = data.copy(deep=True)
    a, b, studies = assets.cluster_totals(data)
    shuffled = data.sample(frac=1, random_state=42)
    c, d, reordered = assets.cluster_totals(shuffled)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert studies == reordered
    assert a.shape == (100, 24)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("kind", ["short", "duplicate", "model", "column", "range"])
def test_malformed_input_fails(data, kind):
    bad = data.copy(deep=True)
    if kind == "short":
        bad = bad.iloc[:-1]
    elif kind == "duplicate":
        bad.iloc[0] = bad.iloc[1]
    elif kind == "model":
        bad.loc[0, "model"] = "unexpected"
    elif kind == "column":
        bad = bad.drop(columns="Q-1-a-3")
    else:
        bad.loc[0, "Q-1-a-3"] = 11
    with pytest.raises(ValueError):
        assets.validate_tagged_data(bad)


def test_loader_rejects_wrong_original_partition(data, tmp_path):
    path = tmp_path / "bad.csv"
    data.iloc[:-1].drop(columns="reviewer").to_csv(path, index=False)
    with pytest.raises(ValueError, match="1600"):
        assets.load_assessments(path)
    # Make a valid global grid but put duplicate keys in each original half.
    bad = data.drop(columns="reviewer").copy()
    first_key = tuple(bad.loc[0, ["pmid", "model"]])
    matches = bad.index[
        bad.pmid.eq(first_key[0]) & bad.model.eq(first_key[1])
    ].tolist()
    other = next(i for i in range(800, 1600) if i not in matches)
    bad.iloc[[0, other]] = bad.iloc[[other, 0]].to_numpy()
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        assets.load_assessments(path)


def test_exclusions_checked_by_key(data):
    assets.validate_exclusions(data, ROOT / assets.AUDIT)
    bad = data.copy()
    mask = (bad.pmid.eq("38293563") & bad.model.eq("gpt-5")
            & bad.reviewer.eq("Reviewer 2"))
    bad.loc[mask, "Q-5-b-2"] = 7
    with pytest.raises(ValueError, match="no longer missing"):
        assets.validate_exclusions(bad, ROOT / assets.AUDIT)


def test_cell_weighting_eligibility_and_observed_points(data):
    sums, counts, _ = assets.cluster_totals(data)
    result = assets.summarise(sums, counts, replicates=40)
    for row in result.itertuples():
        cols = [f"Q-{q}-{assets.DIMENSIONS[row.dimension]}" for q in range(1, 6)]
        values = data.loc[data.model.eq(row.model), cols].to_numpy().ravel()
        eligible = values[values > 0]
        assert row.eligible_cells == len(eligible)
        assert row.mean == eligible.sum() / len(eligible)
    synthetic = data.copy()
    synthetic[assets.SCORE_COLUMNS] = np.nan
    synthetic.loc[0, "Q-1-a-3"] = 2
    synthetic.loc[0, "Q-2-a-3"] = 0
    synthetic.loc[1, "Q-1-a-3"] = 10
    a, b, _ = assets.cluster_totals(synthetic)
    assert a.sum() == 12 and b.sum() == 2


def test_repeated_draws_zero_clusters_and_shared_targets():
    sums = np.array([[2., 4.], [18., 36.], [0., 0.]])
    counts = np.array([[1, 1], [2, 2], [0, 0]])
    draws = np.array([[0, 0, 1], [1, 2, 2], [2, 2, 2]])
    actual = assets.resample_ratios(sums, counts, draws)
    np.testing.assert_array_equal(actual[:2], [[5.5, 11], [9, 18]])
    assert np.isnan(actual[2]).all()


def test_deterministic_seed_and_preserved_point_estimate(data):
    sums, counts, _ = assets.cluster_totals(data)
    a = assets.summarise(sums, counts, replicates=50, seed=123)
    b = assets.summarise(sums, counts, replicates=50, seed=123)
    c = assets.summarise(sums, counts, replicates=50, seed=456)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(a["mean"], c["mean"])
    assert not a.lower.equals(c.lower)


def test_undefined_degenerate_and_empty_targets():
    counts = np.ones((2, 24), dtype=int)
    sums = np.full((2, 24), 7.)
    counts[:, 0] = 0
    sums[:, 0] = 0
    counts[1, 1] = 0
    sums[1, 1] = 0
    result = assets.summarise(sums, counts, replicates=100, seed=1)
    assert result.iloc[0].status == "no_eligible_scores"
    assert result.iloc[0].undefined_replicates == 100
    assert result.iloc[1].status == "undefined_replicates"
    assert result.iloc[1].undefined_replicates > 0
    assert np.isnan(result.iloc[1].lower)
    assert result.iloc[1]["mean"] == 7
    assert result.iloc[2:].status.eq("degenerate").all()
    assert result.iloc[2:].lower.eq(7).all()
    with pytest.raises(ValueError, match="undefined"):
        assets.create_figure(result, 0.95)


def args_for(tmp_path, **kwargs):
    """Build CLI arguments with an isolated output destination."""
    return SimpleNamespace(
        input_file=ROOT / assets.INPUT, audit_file=ROOT / assets.AUDIT,
        output_dir=tmp_path / "output", replicates=40, seed=20261010,
        confidence=0.95, dry_run=kwargs.get("dry_run", False),
    )


def test_dry_run_has_no_output_or_bootstrap(tmp_path, monkeypatch):
    args = args_for(tmp_path, dry_run=True)
    monkeypatch.setattr(assets, "parse_args", lambda: args)
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted computation or writes")
    monkeypatch.setattr(assets, "summarise", forbidden)
    monkeypatch.setattr(assets, "export_outputs", forbidden)
    assets.main()
    assert not args.output_dir.exists()


def test_canonical_output_rejected(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.output_dir = ROOT / "data/artifacts/manuscript-figures"
    monkeypatch.setattr(assets, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="canonical"):
        assets.main()


def test_real_export_layout_bounds_and_no_input_mutation(data, tmp_path):
    paths = [ROOT / assets.INPUT, ROOT / assets.AUDIT]
    before = [p.read_bytes() for p in paths]
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=40)
    fig = assets.create_figure(summary, 0.95)
    np.testing.assert_array_equal(fig.get_size_inches(), [6.175, 2.2])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        lo, hi = ax.get_xlim()
        assert lo < summary.lower.min() <= summary.upper.max() < hi
        texts = [ax.title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        for text in texts:
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0
            assert bounds.x1 <= fig.bbox.width
            assert bounds.y1 <= fig.bbox.height
    plt.close(fig)
    args = args_for(tmp_path)
    assets.export_outputs(summary, {"test": True}, args)
    png = args.output_dir / f"{assets.BASENAME}.png"
    pdf = args.output_dir / f"{assets.BASENAME}.pdf"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert len(list(args.output_dir.iterdir())) == 7
    assert [p.read_bytes() for p in paths] == before
    assert not plt.get_fignums()


def test_export_failure_is_explicit_and_closes(data, tmp_path, monkeypatch):
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=20)
    def fail(*args, **kwargs):
        raise OSError("export failed")
    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="export failed"):
        assets.export_outputs(summary, {}, args_for(tmp_path))
    assert not plt.get_fignums()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_study_identifiers_rejected(data, blank):
    bad = data.copy()
    bad.loc[bad.pmid.eq(bad.pmid.iloc[0]), "pmid"] = blank
    with pytest.raises(ValueError, match="blank or whitespace"):
        assets.validate_tagged_data(bad)


@pytest.mark.parametrize("failure", ["undefined", "png", "pdf"])
def test_failed_rerun_removes_stale_and_partial_images(
    data, tmp_path, monkeypatch, failure,
):
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=20)
    args = args_for(tmp_path)
    assets.export_outputs(summary, {"run": "successful"}, args)
    images = [
        args.output_dir / f"{assets.BASENAME}.{ext}"
        for ext in ("png", "pdf")
    ]
    assert all(p.exists() for p in images)
    original_save = Figure.savefig
    saves = []

    def fail_at_format(fig, path, **kwargs):
        saves.append(Path(path).suffix)
        if Path(path).suffix == f".{failure}":
            Path(path).write_bytes(b"partial image")
            raise OSError("export failed mid-write")
        return original_save(fig, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_at_format)
    if failure == "undefined":
        summary.loc[0, "status"] = "undefined_replicates"
        summary.loc[0, ["lower", "upper"]] = np.nan
    error = ValueError if failure == "undefined" else OSError
    with pytest.raises(error):
        assets.export_outputs(summary, {"run": "failed"}, args)
    assert all(not p.exists() for p in images)
    if failure == "pdf":
        assert saves == [".png", ".pdf"]
    assert (args.output_dir / f"{assets.BASENAME}.csv").exists()
    assert not plt.get_fignums()


def test_combined_preserves_native_heatmap_and_lower_geometry(data, tmp_path):
    from PIL import Image
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=20)
    fig = assets.create_figure(summary, 0.95, ROOT / assets.HEATMAP)
    np.testing.assert_array_equal(fig.get_size_inches(), [6.175, 5.77])
    raster = plt.imread(ROOT / assets.HEATMAP)
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), raster[:1005])
    assert np.all(raster[1005:, :, :3] == 1)
    assert len(fig.axes) == 4
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes[1:]:
        lo, hi = ax.get_xlim()
        assert lo < summary.lower.min() <= summary.upper.max() < hi
        assert len(ax.collections[0].get_segments()) == 8
        texts = [ax.title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        for text in texts:
            assert text.get_fontsize() >= 8
            bounds = text.get_window_extent(renderer)
            assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.width
            assert 0 <= bounds.y0 < bounds.y1 <= fig.bbox.height
    for text in fig.texts:
        bounds = text.get_window_extent(renderer)
        assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.width
        assert 0 <= bounds.y0 < bounds.y1 <= fig.bbox.height
    plt.close(fig)
    args = args_for(tmp_path)
    assets.export_outputs(summary, {}, args)
    image = Image.open(args.output_dir / f"{assets.COMBINED}.png")
    original = Image.open(ROOT / assets.HEATMAP)
    np.testing.assert_array_equal(
        np.asarray(image.crop((0, 66, 1852, 1071))), np.asarray(original)[:1005]
    )


@pytest.mark.parametrize("failure_index", [2, 3])
def test_combined_failure_cleans_all_preview_pairs(
    data, tmp_path, monkeypatch, failure_index,
):
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=20)
    args = args_for(tmp_path)
    assets.export_outputs(summary, {}, args)
    save = Figure.savefig
    calls = []
    def fail(fig, path, **kwargs):
        calls.append(path)
        if len(calls) - 1 == failure_index:
            Path(path).write_bytes(b"partial")
            raise OSError("composite export failed")
        return save(fig, path, **kwargs)
    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="composite export"):
        assets.export_outputs(summary, {}, args)
    assert not list(args.output_dir.glob("*.png"))
    assert not list(args.output_dir.glob("*.pdf"))
    assert not plt.get_fignums()


def test_render_only_preserves_csv_and_skips_bootstrap(tmp_path, monkeypatch):
    import json
    import shutil
    args = args_for(tmp_path)
    args.render_only = True
    args.replicates = 10000
    args.output_dir.mkdir()
    for name in [f"{assets.BASENAME}.csv", "metadata.json"]:
        shutil.copyfile(ROOT / assets.OUTPUT / name, args.output_dir / name)
    before = (args.output_dir / f"{assets.BASENAME}.csv").read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Render-only attempted bootstrap or CSV write")
    monkeypatch.setattr(assets, "parse_args", lambda: args)
    monkeypatch.setattr(assets, "summarise", forbidden)
    monkeypatch.setattr(assets, "cluster_totals", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    assets.main()
    assert (args.output_dir / f"{assets.BASENAME}.csv").read_bytes() == before
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    assert metadata["layout"]["minimum_lower_text_points"] == 8
    assert metadata["embedded_heatmap"] == assets.file_identity(assets.HEATMAP)


def test_heatmap_margin_crop_rejects_nonwhite_pixels(data, monkeypatch):
    sums, counts, _ = assets.cluster_totals(data)
    summary = assets.summarise(sums, counts, replicates=20)
    raster = plt.imread(ROOT / assets.HEATMAP).copy()
    raster[1050, 20, 0] = 0
    monkeypatch.setattr(plt, "imread", lambda path: raster)
    with pytest.raises(ValueError, match="nonwhite content"):
        assets.create_figure(summary, 0.95, ROOT / assets.HEATMAP)
    assert not plt.get_fignums()


def test_render_only_dry_run_no_writes_or_bootstrap(tmp_path, monkeypatch):
    args = args_for(tmp_path, dry_run=True)
    args.render_only = True
    monkeypatch.setattr(assets, "parse_args", lambda: args)
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run attempted generation or bootstrap")
    monkeypatch.setattr(assets, "summarise", forbidden)
    monkeypatch.setattr(assets, "export_outputs", forbidden)
    monkeypatch.setattr(assets, "create_figure", forbidden)
    assets.main()
    assert not args.output_dir.exists()


@pytest.mark.parametrize("combined", [False, True])
def test_requested_display_order_maps_every_point_and_endpoint(combined):
    summary = pd.read_csv(
        ROOT / assets.OUTPUT / f"{assets.BASENAME}.csv",
        float_precision="round_trip",
    )
    expected_order = [
        "gpt-5-mini", "gpt-5", "gpt-4-1", "o4-mini", "llama3-2",
        "gpt-4o", "deepseek-r1", "llama3",
    ]
    scientific_order = [
        "deepseek-r1", "gpt-4-1", "gpt-4o", "gpt-5", "gpt-5-mini",
        "llama3", "llama3-2", "o4-mini",
    ]
    assert list(assets.MODELS) == scientific_order
    assert list(assets.DISPLAY_MODELS) == expected_order
    assert list(summary.model.drop_duplicates()) == scientific_order
    fig = assets.create_figure(
        summary, 0.95, ROOT / assets.HEATMAP if combined else None,
    )
    axes = fig.axes[-3:]
    assert [t.get_text() for t in axes[0].get_yticklabels()] == expected_order
    positions = axes[0].transData.transform(
        np.column_stack([np.zeros(8), np.arange(8)])
    )[:, 1]
    assert np.all(np.diff(positions) < 0)  # Requested top-to-bottom order.
    for ax, dimension in zip(axes, assets.DIMENSIONS):
        rows = summary.loc[summary.dimension.eq(dimension)].set_index("model")
        rows = rows.loc[expected_order]
        points = ax.collections[1].get_offsets()
        segments = np.asarray(ax.collections[0].get_segments())
        np.testing.assert_array_equal(points[:, 0], rows["mean"])
        np.testing.assert_array_equal(points[:, 1], np.arange(8))
        np.testing.assert_array_equal(segments[:, 0, 0], rows.lower)
        np.testing.assert_array_equal(segments[:, 1, 0], rows.upper)
        np.testing.assert_array_equal(segments[:, :, 1], np.repeat(
            np.arange(8)[:, None], 2, axis=1,
        ))
    assert not any("alphabetical" in t.get_text() for t in fig.texts)
    plt.close(fig)


@pytest.mark.parametrize("combined", [False, True])
def test_mean_labels_values_alignment_and_nonoverlap(combined):
    summary = pd.read_csv(
        ROOT / assets.OUTPUT / f"{assets.BASENAME}.csv",
        float_precision="round_trip",
    )
    fig = assets.create_figure(
        summary, 0.95, ROOT / assets.HEATMAP if combined else None,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = fig.axes[-3:]
    total = 0
    for index, (ax, dimension) in enumerate(zip(axes, assets.DIMENSIONS)):
        rows = summary.loc[summary.dimension.eq(dimension)].set_index("model")
        rows = rows.loc[list(assets.DISPLAY_MODELS)]
        labels = [t for t in ax.texts if t.get_gid() == "mean-label"]
        assert [t.get_text() for t in labels] == [
            f"{mean:.2f}" for mean in rows["mean"]
        ]
        assert len(labels) == 8
        total += len(labels)
        bounds = [t.get_window_extent(renderer) for t in labels]
        for row, (label, box) in enumerate(zip(labels, bounds)):
            assert label.get_fontsize() >= 8
            assert label.get_ha() == "right"
            assert box.x0 > ax.bbox.x1  # Outside every CI and point.
            center_y = ax.transData.transform((0, row))[1]
            assert abs((box.y0 + box.y1) / 2 - center_y) < 1e-10
            if index < 2:
                assert box.x1 < axes[index + 1].bbox.x0
        np.testing.assert_allclose([b.x1 for b in bounds], bounds[0].x1)
        for upper, lower in zip(bounds, bounds[1:]):
            assert upper.y0 > lower.y1
        other_text = (
            [ax.title, ax.xaxis.label] + ax.get_xticklabels()
            + ax.get_yticklabels()
            + [t for t in ax.texts if t.get_gid() != "mean-label"]
        )
        for box in bounds:
            for text in other_text:
                if text.get_visible():
                    assert not box.overlaps(text.get_window_extent(renderer))
        for text in ax.texts + other_text:
            if not text.get_visible():
                continue
            box = text.get_window_extent(renderer)
            assert 0 <= box.x0 < box.x1 <= fig.bbox.width
            assert 0 <= box.y0 < box.y1 <= fig.bbox.height
        header = next(t for t in ax.texts if t.get_text() == "Mean")
        assert not header.get_window_extent(renderer).overlaps(
            ax.title.get_window_extent(renderer)
        )
    assert total == 24
    plt.close(fig)
