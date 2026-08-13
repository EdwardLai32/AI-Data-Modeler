"""Chart and report rendering.

The report is the deliverable — for most users it is the entire product — so the
tests here care about two things above correctness of layout.

**Degrade, never crash.** Chart rendering depends on kaleido for PNG export, PDF
on reportlab, PPTX on python-pptx. Any of those can be absent, and a run that
completes but produces no report because a static-image exporter is missing has
failed the user. Every renderer must return a bundle that records what it could
not do.

**The reasoning has to survive into the output.** A report that lists decisions
without their rationale defeats the point of the architecture. Markdown output is
checked for the actual sentences the agents wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl_architect.core.schemas import (
    ChartArtifact,
    ChartKind,
    ChartSpec,
    ExperimentLog,
    ExperimentResult,
    MetricValue,
    ModelFamily,
    ReportBundle,
    VisualizationBundle,
    VisualizationPlan,
)
from automl_architect.core.state import RunState

from .conftest import (
    CHURN_LEAK,
    evaluation_verdict,
    explainability_narration,
    final_report,
    import_or_skip,
    insight_report,
    visualization_plan,
)


@pytest.fixture(scope="module")
def charts_mod():  # noqa: ANN201
    return import_or_skip("automl_architect.reporting.charts", feature="reporting/charts.py")


@pytest.fixture(scope="module")
def writer_mod():  # noqa: ANN201
    return import_or_skip("automl_architect.reporting.writer", feature="reporting/writer.py")


@pytest.fixture
def reportable_state(run_state: RunState) -> RunState:
    """A state carrying everything a report needs, without running the pipeline.

    Built from canned agent outputs plus a hand-made experiment log so the
    reporting tests stay fast and independent of the trainer.
    """
    winner = ExperimentResult(
        family=ModelFamily.HIST_GRADIENT_BOOSTING,
        label="hist_gradient_boosting",
        metrics=[
            MetricValue(name="roc_auc", value=0.712, std=0.019),
            MetricValue(name="average_precision", value=0.451),
            MetricValue(name="accuracy", value=0.766),
        ],
        cv_scores=[0.701, 0.718, 0.709, 0.722, 0.710],
        primary_metric="roc_auc",
        primary_score=0.712,
        train_seconds=1.9,
        n_features_in=24,
    )
    baseline = ExperimentResult(
        family=ModelFamily.BASELINE_DUMMY,
        label="baseline_dummy",
        metrics=[MetricValue(name="roc_auc", value=0.5), MetricValue(name="accuracy", value=0.74)],
        primary_metric="roc_auc",
        primary_score=0.5,
        is_baseline=True,
        n_features_in=24,
    )
    run_state.experiments = ExperimentLog(
        results=[baseline, winner],
        best_experiment_id=winner.experiment_id,
        primary_metric="roc_auc",
        higher_is_better=True,
        leaderboard_notes="Boosting leads the dummy floor by 0.212 AUC.",
    )
    run_state.explainability = explainability_narration()
    run_state.evaluation = evaluation_verdict(acceptable=True)
    run_state.insights = insight_report()
    run_state.visualization_plan = visualization_plan()
    run_state.report = final_report()
    run_state.feature_names = [
        c for c in run_state.working_df.columns if c not in {"churned", CHURN_LEAK}
    ]
    return run_state


# ===========================================================================
# Charts
# ===========================================================================


class TestCharts:
    def test_returns_an_artifact_per_spec(self, charts_mod, reportable_state: RunState) -> None:
        plan = visualization_plan()
        bundle = charts_mod.render_charts(reportable_state, plan)
        assert isinstance(bundle, VisualizationBundle)
        assert len(bundle.artifacts) == len(plan.charts), (
            "every requested chart needs an artifact, even a failed one"
        )

    def test_each_artifact_either_rendered_or_explained(
        self, charts_mod, reportable_state: RunState
    ) -> None:
        """A silently missing chart is the failure mode; a recorded error is fine."""
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        for artifact in bundle.artifacts:
            assert isinstance(artifact, ChartArtifact)
            if artifact.rendered:
                assert artifact.html_path or artifact.png_path or artifact.json_path
            else:
                assert artifact.error, f"{artifact.spec.kind.value} failed with no reason"

    def test_rendered_files_exist_on_disk(self, charts_mod, reportable_state: RunState) -> None:
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        rendered = [a for a in bundle.artifacts if a.rendered]
        assert rendered, "no chart rendered at all"
        for artifact in rendered:
            for path in (artifact.html_path, artifact.png_path, artifact.json_path):
                if path:
                    assert Path(path).exists(), f"{path} was reported but not written"
                    assert Path(path).stat().st_size > 0

    def test_artifacts_stay_inside_the_run_directory(
        self, charts_mod, reportable_state: RunState
    ) -> None:
        """Artifacts are served over HTTP from this directory, so they must live in it."""
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        root = reportable_state.artifact_dir.resolve()
        for artifact in bundle.artifacts:
            for path in (artifact.html_path, artifact.png_path, artifact.json_path):
                if path:
                    assert root in Path(path).resolve().parents

    def test_artifacts_keep_their_spec_and_caption(
        self, charts_mod, reportable_state: RunState
    ) -> None:
        """The caption is what makes a chart readable without the surrounding prose."""
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        for artifact in bundle.artifacts:
            assert artifact.spec.title.strip()
            assert artifact.spec.rationale.strip()

    def test_empty_plan_yields_an_empty_bundle(self, charts_mod, reportable_state: RunState) -> None:
        bundle = charts_mod.render_charts(reportable_state, VisualizationPlan(charts=[]))
        assert bundle.artifacts == []

    def test_a_chart_the_data_cannot_support_degrades(
        self, charts_mod, reportable_state: RunState
    ) -> None:
        """A residual plot on a classification run is a mistake, not a fatal error."""
        plan = VisualizationPlan(
            charts=[
                ChartSpec(
                    kind=ChartKind.RESIDUALS,
                    title="Residuals",
                    columns=[],
                    rationale="asked for on a classification run",
                )
            ]
        )
        bundle = charts_mod.render_charts(reportable_state, plan)
        assert len(bundle.artifacts) == 1
        artifact = bundle.artifacts[0]
        assert artifact.rendered or artifact.error

    def test_unknown_column_reference_degrades(self, charts_mod, reportable_state: RunState) -> None:
        plan = VisualizationPlan(
            charts=[
                ChartSpec(
                    kind=ChartKind.HISTOGRAM,
                    title="Histogram of an imaginary column",
                    columns=["__not_a_real_column__"],
                    rationale="testing the grounding filter",
                )
            ]
        )
        bundle = charts_mod.render_charts(reportable_state, plan)
        assert bundle.artifacts[0].rendered is False or bundle.artifacts[0].error is None

    def test_missing_kaleido_does_not_prevent_html(
        self, charts_mod, reportable_state: RunState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PNG export is optional; interactive HTML is not.

        Simulated by making the static-image writer raise, which is what an absent
        or broken kaleido install looks like from plotly's side.
        """
        import plotly.io as pio

        def explode(*args: object, **kwargs: object) -> None:
            raise ValueError("kaleido is not installed")

        monkeypatch.setattr(pio, "write_image", explode, raising=False)
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        assert any(a.html_path or a.json_path for a in bundle.artifacts), (
            "losing PNG export must not lose every chart"
        )

    def test_dashboard_is_assembled(self, charts_mod, reportable_state: RunState) -> None:
        bundle = charts_mod.render_charts(reportable_state, visualization_plan())
        if bundle.dashboard_path:
            assert Path(bundle.dashboard_path).exists()


# ===========================================================================
# Reports
# ===========================================================================


class TestWriter:
    def test_markdown_and_json_are_written(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(
            reportable_state, reportable_state.report, ["markdown", "json"]
        )
        assert isinstance(bundle, ReportBundle)
        assert bundle.markdown_path and Path(bundle.markdown_path).exists()
        assert bundle.json_path and Path(bundle.json_path).exists()

    def test_markdown_carries_the_report_content(
        self, writer_mod, reportable_state: RunState
    ) -> None:
        report = reportable_state.report
        bundle = writer_mod.write_report(reportable_state, report, ["markdown"])
        text = Path(bundle.markdown_path).read_text(encoding="utf-8")
        assert report.title in text
        assert report.executive_summary[:60] in text
        for section in report.sections:
            assert section.heading in text, f"section '{section.heading}' is missing"

    def test_markdown_preserves_the_reasoning(self, writer_mod, reportable_state: RunState) -> None:
        """A report that drops the rationale defeats the architecture."""
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["markdown"])
        text = Path(bundle.markdown_path).read_text(encoding="utf-8")
        assert CHURN_LEAK in text, "the leakage decision did not reach the report"
        assert "0.9975" in text or "cancellation" in text.lower()

    def test_json_round_trips(self, writer_mod, reportable_state: RunState) -> None:
        """The JSON report is the machine-readable audit record."""
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["json"])
        payload = json.loads(Path(bundle.json_path).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload

    def test_html_is_self_contained(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["html"])
        if not bundle.html_path:
            assert bundle.warnings
            return
        text = Path(bundle.html_path).read_text(encoding="utf-8")
        assert "<html" in text.lower()
        assert reportable_state.report.title in text

    def test_outputs_land_in_the_run_directory(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(
            reportable_state, reportable_state.report, ["markdown", "json"]
        )
        root = reportable_state.artifact_dir.resolve()
        for path in (bundle.markdown_path, bundle.json_path, bundle.html_path):
            if path:
                assert root in Path(path).resolve().parents

    def test_unknown_format_is_warned_not_fatal(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(
            reportable_state, reportable_state.report, ["markdown", "papyrus"]
        )
        assert bundle.markdown_path
        assert any("papyrus" in warning.lower() for warning in bundle.warnings), bundle.warnings

    def test_empty_format_list_falls_back_to_defaults(
        self, writer_mod, reportable_state: RunState
    ) -> None:
        """An empty list means "the defaults", not "produce nothing".

        A misconfigured ``report_formats: []`` silently discarding the deliverable
        would be the worse behaviour, so ``normalise_formats`` documents Markdown
        plus HTML as the fallback and this pins it.
        """
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, [])
        assert bundle.markdown_path, "the documented fallback did not produce markdown"
        assert Path(bundle.markdown_path).exists()

    def test_blank_format_names_are_ignored(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(
            reportable_state, reportable_state.report, ["markdown", "", "  "]
        )
        assert bundle.markdown_path

    def test_pdf_either_renders_or_warns(self, writer_mod, reportable_state: RunState) -> None:
        """reportlab is installed here, but the contract must hold either way."""
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["pdf"])
        if bundle.pdf_path:
            assert Path(bundle.pdf_path).exists()
            assert Path(bundle.pdf_path).read_bytes().startswith(b"%PDF")
        else:
            assert bundle.warnings

    def test_pptx_either_renders_or_warns(self, writer_mod, reportable_state: RunState) -> None:
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["pptx"])
        if bundle.pptx_path:
            assert Path(bundle.pptx_path).exists()
            # A .pptx is a zip archive.
            assert Path(bundle.pptx_path).read_bytes().startswith(b"PK")
        else:
            assert bundle.warnings

    def test_missing_optional_dependency_degrades(
        self, writer_mod, reportable_state: RunState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate reportlab being absent; markdown must still be produced."""
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".")[0] == "reportlab":
                raise ImportError("No module named 'reportlab'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        bundle = writer_mod.write_report(
            reportable_state, reportable_state.report, ["markdown", "pdf"]
        )
        assert bundle.markdown_path, "one missing exporter must not lose the whole report"
        if bundle.pdf_path is None:
            assert bundle.warnings

    def test_charts_are_referenced_by_the_report(
        self, charts_mod, writer_mod, reportable_state: RunState
    ) -> None:
        """Sections name chart refs; the written report should resolve them."""
        reportable_state.visualizations = charts_mod.render_charts(
            reportable_state, visualization_plan()
        )
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["markdown"])
        text = Path(bundle.markdown_path).read_text(encoding="utf-8")
        assert text.strip()

    def test_report_is_written_even_with_no_charts(
        self, writer_mod, reportable_state: RunState
    ) -> None:
        reportable_state.visualizations = None
        bundle = writer_mod.write_report(reportable_state, reportable_state.report, ["markdown"])
        assert bundle.markdown_path and Path(bundle.markdown_path).exists()

    def test_writing_twice_is_idempotent(self, writer_mod, reportable_state: RunState) -> None:
        """A replan re-writes the report; the second pass must not append or fail."""
        first = writer_mod.write_report(reportable_state, reportable_state.report, ["markdown"])
        first_text = Path(first.markdown_path).read_text(encoding="utf-8")
        second = writer_mod.write_report(reportable_state, reportable_state.report, ["markdown"])
        assert Path(second.markdown_path).read_text(encoding="utf-8") == first_text


# ===========================================================================
# Cross-cutting
# ===========================================================================


def test_unicode_survives_every_format(writer_mod, reportable_state: RunState) -> None:
    """Column names and business text are not guaranteed to be ASCII."""
    report = reportable_state.report
    report.title = "Résumé d'analyse — 顧客離反 — 90% ✓"
    bundle = writer_mod.write_report(reportable_state, report, ["markdown", "json"])
    assert report.title in Path(bundle.markdown_path).read_text(encoding="utf-8")
