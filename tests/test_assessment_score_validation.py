"""Regression checks for raw scored-cell exclusion and assessment-only runs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import an analysis script without invoking its CLI."""
    path = ROOT / "scripts/python/analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processor = load_script("process-assessment-data")
main_assets = load_script("create-manuscript-assets")
supp_assets = load_script("create-supplementary-figures")


@pytest.mark.parametrize("question", processor.SCORE_COLUMNS)
def test_raw_range_keeps_boundaries_sentinels_and_counts(question):
    values = [1, 10, 0, None, -1, 11, 0.5, np.inf, -np.inf, "NA"]
    df = pd.DataFrame(
        {
            "pmid": [str(i) for i in range(len(values))],
            "model": "gpt-5",
            question: values,
            "Q-5-b-1": 26,
            "Q-5-b-3": "private assessment text",
        },
        index=range(10, 20),
    )
    audit = []
    actual = processor.validate_raw_scores(df, "Reviewer 1", audit)
    assert actual.index.equals(df.index)
    pd.testing.assert_frame_equal(
        actual.drop(columns=question), df.drop(columns=question)
    )
    assert actual[question].iloc[:3].tolist() == [1, 10, 0]
    assert actual[question].iloc[3:].isna().all()
    assert len(audit) == 5
    assert {row["question"] for row in audit} == {question}
    assert all(set(row) == set(processor.AUDIT_COLUMNS) for row in audit)
    assert "private assessment text" not in str(audit)
    assert df[question].iloc[5] == 11  # Input is not mutated.


@pytest.mark.parametrize("question", processor.SCORE_COLUMNS)
def test_reviewer_2_validates_before_completeness_recode(tmp_path, question):
    path = tmp_path / "reviewer.csv"
    df = pd.DataFrame(
        {
            "Column1": [1, 2, 3, 4, 5, 6, 7],
            "Column2": "gpt_5",
            question: [1, 10, -1, 11, 0, None, "None"],
            "Q-5-a-1": 42,
        }
    )
    df.to_csv(path, index=False)
    audit = []
    actual = processor.load_reviewer_2_data(path, audit)
    scores = actual[question]
    expected = [9, 0] if question.endswith("c-2") else [1, 10]
    assert scores.iloc[:2].tolist() == expected
    assert scores.iloc[2:4].isna().all()
    assert scores.iloc[4] == (10 if question.endswith("c-2") else 0)
    assert scores.iloc[5:].isna().all()
    assert actual["Q-5-a-1"].eq(42).all()
    assert actual["model"].eq("gpt-5").all()
    assert [row["raw_value"] for row in audit] == [-1, 11]
    assert all(row["reviewer"] == "Reviewer 2" for row in audit)
    assert all(row["model"] == "gpt-5" for row in audit)


def test_reviewer_1_loader_excludes_cells_without_reverse_scoring(tmp_path):
    xlsx = tmp_path / "reviewer.xlsx"
    sheets = tmp_path / "sheets.txt"
    columns = tmp_path / "columns.txt"
    sheets.write_text("gpt4-1\n")
    columns.write_text("pmid\nQ-5-b-2\nQ-5-c-2\nQ-5-b-1\n")
    pd.DataFrame(
        {"pmid": [1, 2], "b": [11, 1], "c": [10, 1], "count": [26, 30]}
    ).to_excel(xlsx, sheet_name="gpt4-1", index=False)
    audit = []
    actual = processor.load_reviewer_1_data(xlsx, sheets, columns, audit)
    assert len(actual) == 2
    assert pd.isna(actual.loc[0, "Q-5-b-2"])
    assert actual.loc[1, "Q-5-b-2"] == 1
    assert actual["Q-5-c-2"].tolist() == [10, 1]
    assert actual["Q-5-b-1"].tolist() == [26, 30]
    assert audit[0]["reviewer"] == "Reviewer 1"
    assert audit[0]["model"] == "gpt-4-1"


def test_downstream_positive_filter_sd_and_bounds_are_unchanged(tmp_path):
    df = pd.DataFrame(
        {
            "pmid": [1, 2, 3, 4],
            "model": "gpt-5",
            "Q-5-b-2": [1.0, 10.0, 0.0, np.nan],
        }
    )
    group = main_assets.compute_group_metrics(df)
    overall = main_assets.compute_overall_metrics(df)
    assert group.loc[0, "mean_score"] == 5.5
    assert overall.loc[0, "overall_mean"] == 5.5
    assert overall.loc[0, "overall_sd"] == 4.5  # Population SD.
    path = tmp_path / "numeric.csv"
    df.to_csv(path, index=False)
    reviewed = supp_assets.load_assessment_data(path)
    assert reviewed["reviewer"].tolist() == [
        "Reviewer 1", "Reviewer 1", "Reviewer 2", "Reviewer 2"
    ]
    chart = supp_assets.create_question_group_figure(reviewed, "Q-5", "Results")
    records = next(iter(chart.to_dict()["datasets"].values()))
    assert len(records) == 1  # Reviewer 2 has no positive scores.
    assert records[0]["SD"] == pytest.approx(np.std([1, 10], ddof=1))
    assert records[0]["Lower"] == 0
    assert records[0]["Upper"] == 10


def test_overall_statistics_pool_cells_not_dimension_means():
    df = pd.DataFrame(
        {
            "model": ["gpt-5"] * 3,
            "Q-1-a-3": [1.0, 3.0, 5.0],
            "Q-5-b-2": [10.0, 0.0, np.nan],
        }
    )
    actual = main_assets.compute_overall_metrics(df).iloc[0]
    assert actual["overall_mean"] == np.mean([1, 3, 5, 10])
    assert actual["overall_mean"] != np.mean([3, 10])
    assert actual["overall_sd"] == np.std([1, 3, 5, 10], ddof=0)


@pytest.mark.parametrize("dry_run", [False, True])
def test_assessment_only_never_reads_or_writes_validation(
    tmp_path, monkeypatch, dry_run
):
    args = SimpleNamespace(
        assessment_file=tmp_path / "input.csv",
        assessment_only=True,
        aggregated_dir=tmp_path / "must-not-read",
        output_figures=tmp_path / "figures",
        output_tables=tmp_path / "tables",
        verbose=False,
        dry_run=dry_run,
    )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Assessment-only path accessed validation data/output")

    monkeypatch.setattr(main_assets, "parse_args", lambda: args)
    monkeypatch.setattr(main_assets, "load_assessment_data", lambda *a: None)
    monkeypatch.setattr(main_assets, "load_validation_metrics", forbidden)
    monkeypatch.setattr(main_assets, "create_table_validation", forbidden)
    monkeypatch.setattr(main_assets, "compute_group_metrics", lambda *a: None)
    monkeypatch.setattr(main_assets, "compute_overall_metrics", lambda *a: None)
    monkeypatch.setattr(
        main_assets, "create_combined_figure", lambda *a: calls.append("figure")
    )
    monkeypatch.setattr(
        main_assets, "create_table_performance", lambda *a: calls.append("table")
    )
    main_assets.main()
    assert calls == ([] if dry_run else ["figure", "table"])
    assert args.output_figures.exists() is not dry_run


def test_save_outputs_emits_keyed_audit_and_honors_dry_run(tmp_path):
    df = pd.DataFrame({"pmid": ["1"], "model": ["gpt-5"], "comment": [""]})
    audit = []
    output = tmp_path / "processed"
    processor.save_outputs(df, output, True, audit)
    assert not output.exists()
    processor.save_outputs(df, output, False, audit)
    saved = pd.read_csv(output / "assessment-score-exclusions.csv")
    assert tuple(saved.columns) == processor.AUDIT_COLUMNS
    assert saved.empty
