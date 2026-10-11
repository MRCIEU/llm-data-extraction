"""Exact supplementary records, shared ordering and offline export checks."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import vl_convert as vlc


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "supplementary_assets",
    ROOT / "scripts/python/analysis/create-supplementary-figures.py",
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def plotted_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the unmodified rows serialized into the chart specification."""
    return [row for rows in spec["datasets"].values() for row in rows]


def rendered_model_order(spec: dict[str, Any]) -> list[str]:
    """Read untruncated model labels in rendered shared-axis order."""
    models = {row["Model"] for row in plotted_records(spec)}
    svg = ET.fromstring(vlc.vegalite_to_svg(spec))
    return [node.text for node in svg.iter() if node.text in models]


@pytest.mark.parametrize("code", list(assets.QUESTION_GROUPS))
def test_exact_canonical_records_domain_and_rendered_order(code: str) -> None:
    """Preserve canonical data, domain visibility and shared rendered order."""
    df = assets.load_assessment_data(
        ROOT / "data/artifacts/assessment-results/assessment-results-numeric.csv"
    )
    original = df.copy(deep=True)
    chart = assets.create_question_group_figure(
        df, code, assets.QUESTION_GROUPS[code]
    )
    spec = chart.to_dict()
    name = assets.FIGURE_NAMES[code]
    canonical = json.loads(
        (ROOT / "data/artifacts/manuscript-figures" / f"{name}.json").read_text()
    )
    rows = plotted_records(spec)
    assert len(rows) == 48
    assert rows == plotted_records(canonical)
    pd.testing.assert_frame_equal(df, original)
    assert df["reviewer"].iloc[:800].eq("Reviewer 1").all()
    assert df["reviewer"].iloc[800:].eq("Reviewer 2").all()
    for row in rows:
        for field in ["Mean", "Lower", "Upper"]:
            assert 4.5 <= row[field] <= 10.2
    old_vega = vlc.vegalite_to_vega(canonical)
    vega = vlc.vegalite_to_vega(spec)
    scales = {scale["name"]: scale for scale in vega["scales"]}
    old_scales = {scale["name"]: scale for scale in old_vega["scales"]}
    assert scales["x"]["domain"] == [4.5, 10.2]
    assert scales["y"] == old_scales["y"]
    assert scales["y"]["domain"]["sort"] == {
        "op": "mean", "field": "Mean", "order": "descending"
    }
    assert rendered_model_order(spec) == rendered_model_order(canonical)
    assert len(rendered_model_order(spec)) == 8
    assert spec["spec"]["layer"][0]["encoding"]["color"]["scale"] == {
        "domain": ["Reviewer 1", "Reviewer 2"],
        "range": ["#689d6a", "#d79921"],
    }
    assert spec["config"]["axis"]["labelFontSize"] == 12
    assert spec["config"]["axis"]["labelLimit"] == 0
    assert spec["config"]["header"]["labelFontSize"] == 13
    assert spec["spec"]["width"] == 150
    assert spec["spec"]["height"] == 220
    assert vlc.vegalite_to_png(spec, scale=1).startswith(b"\x89PNG\r\n\x1a\n")


def test_stable_basenames_and_explicit_q5_title() -> None:
    """Keep output identities fixed while clarifying display titles."""
    df = assets.load_assessment_data(
        ROOT / "data/artifacts/assessment-results/assessment-results-numeric.csv"
    )
    charts = assets.create_all_supplementary_figures(df)
    assert list(charts) == list(assets.FIGURE_NAMES.values())
    assert charts["supplementary-q-5-results"].to_dict()["title"] == (
        "Q-5: MR findings extracted from the abstract"
    )
    assert charts["supplementary-q-3-methods"].to_dict()["title"] == (
        "Q-3: Analytical methods"
    )


def test_failed_png_export_propagates(tmp_path: Path) -> None:
    """Require PNG export failure to propagate instead of succeeding silently."""
    class BrokenChart:
        """Simulate an exporter that can save JSON but not PNG."""

        def save(self, filename: str, **kwargs: Any) -> None:
            """Write JSON or raise the simulated offline PNG export failure."""
            path = Path(filename)
            if path.suffix == ".png":
                raise RuntimeError("offline renderer failed")
            path.write_text("{}")

    with pytest.raises(RuntimeError, match="offline renderer failed"):
        assets.save_figures({"test": BrokenChart()}, tmp_path, False)
    assert not (tmp_path / "test.png").exists()


def test_dry_run_does_not_create_output(tmp_path: Path) -> None:
    """Ensure a presentation dry run creates no output directory."""
    output = tmp_path / "must-not-create"
    assets.save_figures({"test": None}, output, True)
    assert not output.exists()
