"""Unit tests for ingestion, EDA profiling, statistical engine, ML lab, and VRAM budgets."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spectrumbloom.domain.exceptions import DatasetValidationError, MLExperimentError
from spectrumbloom.domain.schemas import MLExperimentRequest
from spectrumbloom.eda.charts import build_eda_figures
from spectrumbloom.eda.loader import parse_tabular_bytes
from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
from spectrumbloom.eda.statistics import compute_statistics
from spectrumbloom.ml.experiments import run_ml_experiment
from spectrumbloom.optimization.vram_budget import build_feasibility_matrix

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "examples" / "sample_ecommerce.csv"


def test_csv_and_xlsx_ingestion_and_guards() -> None:
    df = parse_tabular_bytes(SAMPLE_CSV.read_bytes(), "sample_ecommerce.csv")
    assert df.shape == (180, 12)

    # Excel round-trip
    buf = io.BytesIO()
    df.head(30).to_excel(buf, index=False)
    df_xl = parse_tabular_bytes(buf.getvalue(), "sample.xlsx")
    assert df_xl.shape == (30, 12)

    # Reject disallowed extension
    with pytest.raises(DatasetValidationError):
        parse_tabular_bytes(b"hello", "payload.exe")

    # Reject oversized file
    with pytest.raises(DatasetValidationError):
        parse_tabular_bytes(b"a,b\n1,2\n" * 100, "big.csv", max_upload_bytes=50)

    # Reject zip bomb simulation (> MAX_ZIP_ENTRIES or bad archive)
    bad_zip = io.BytesIO()
    with zipfile.ZipFile(bad_zip, "w") as zf:
        for i in range(105):
            zf.writestr(f"sheet_{i}.xml", "<xml/>")
    with pytest.raises(DatasetValidationError):
        parse_tabular_bytes(bad_zip.getvalue(), "bomb.xlsx")


def test_eda_profiling_statistics_and_charts() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    profile = profile_dataframe(df, filename="sample_ecommerce.csv")

    assert profile.rows == 180
    assert profile.columns == 12
    roles = {cp.name: cp.semantic_type for cp in profile.column_profiles}
    assert roles["order_id"] == "identifier"
    assert roles["order_date"] == "datetime"
    assert roles["revenue"] == "numeric_continuous"
    assert roles["category"] == "categorical"

    stats_rep = compute_statistics(df, profile)
    assert stats_rep.missingness.total_missing_cells == 23
    assert len(stats_rep.outliers) >= 1
    assert len(stats_rep.top_correlations) >= 1
    assert len(stats_rep.group_tests) >= 1

    figures = build_eda_figures(df, profile, stats_rep)
    assert len(figures) >= 5
    prompt_ctx = format_profile_for_prompt(profile, df)
    assert "sample_ecommerce.csv" in prompt_ctx
    assert "revenue" in prompt_ctx


def test_ml_classification_and_regression_pipelines() -> None:
    df = pd.read_csv(SAMPLE_CSV)

    clf_res = run_ml_experiment(
        df,
        MLExperimentRequest(target_column="churned", task_type="auto", cv_folds=3),
    )
    assert clf_res.task_type == "classification"
    assert len(clf_res.models) == 4
    assert len(clf_res.figures) == 3
    assert any("order_id" in note for note in clf_res.notes)

    reg_res = run_ml_experiment(
        df,
        MLExperimentRequest(target_column="revenue", task_type="regression", cv_folds=3),
    )
    assert reg_res.task_type == "regression"
    assert len(reg_res.models) == 4
    assert len(reg_res.figures) == 3

    with pytest.raises(MLExperimentError):
        run_ml_experiment(df, MLExperimentRequest(target_column="nonexistent"))


def test_vram_feasibility_matrix_for_rtx3050_4gb() -> None:
    matrix = build_feasibility_matrix()
    qwen4b_q4 = next(
        row
        for row in matrix["inference_matrix"]
        if row["model_name"] == "Qwen3-4B" and row["quantization"] == "Q4_K_M"
    )
    assert qwen4b_q4["fits_strict_vram"] is True

    qwen4b_qlora = next(row for row in matrix["qlora_matrix"] if row["model"] == "Qwen3-4B")
    assert qwen4b_qlora["feasible_on_rtx3050_4gb"] is False

    qwen05b_qlora = next(
        row for row in matrix["qlora_matrix"] if row["model"] == "Qwen2.5-Coder-0.5B"
    )
    assert qwen05b_qlora["feasible_on_rtx3050_4gb"] is True
