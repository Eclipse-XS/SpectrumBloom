"""Materialize the six diverse evaluation datasets with documented provenance and SHA-256 hashes."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from statsmodels.datasets import co2, macrodata

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "benchmarks" / "datasets"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_datasets() -> list[dict[str, str]]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    # 1. Sample E-Commerce (synthetic repository dataset)
    ecom_dst = DATASET_DIR / "sample_ecommerce.csv"
    shutil.copyfile(ROOT / "examples" / "sample_ecommerce.csv", ecom_dst)
    ecom_df = pd.read_csv(ecom_dst)
    records.append(
        {
            "file": ecom_dst.name,
            "rows": str(len(ecom_df)),
            "cols": str(ecom_df.shape[1]),
            "source": "SpectrumBloom repository sample (`examples/sample_ecommerce.csv`)",
            "license": "MIT",
            "notes": "E-commerce orders with missingness, date column, outliers, binary churned",
            "sha256": _sha256(ecom_dst),
        }
    )

    # 2. Palmer Penguins (Allison Horst / Gorman et al. 2014)
    peng_dst = DATASET_DIR / "palmer_penguins.csv"
    peng_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    req = urllib.request.Request(peng_url, headers={"User-Agent": "SpectrumBloom/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        peng_dst.write_bytes(resp.read())
    peng_df = pd.read_csv(peng_dst)
    records.append(
        {
            "file": peng_dst.name,
            "rows": str(len(peng_df)),
            "cols": str(peng_df.shape[1]),
            "source": "Palmer Station Antarctica LTER / seaborn-data (`penguins.csv`)",
            "license": "CC0 1.0 Universal",
            "notes": "3 penguin species across 3 islands with natural missing numeric & sex values",
            "sha256": _sha256(peng_dst),
        }
    )

    # 3. UCI Breast Cancer Wisconsin (Diagnostic)
    bc = load_breast_cancer(as_frame=True)
    bc_df = bc.frame.copy()
    bc_df["diagnosis"] = bc_df["target"].map({0: "malignant", 1: "benign"})
    bc_df = bc_df.drop(columns=["target"])
    bc_dst = DATASET_DIR / "breast_cancer_uci.csv"
    bc_df.to_csv(bc_dst, index=False)
    records.append(
        {
            "file": bc_dst.name,
            "rows": str(len(bc_df)),
            "cols": str(bc_df.shape[1]),
            "source": "UCI ML Repository via scikit-learn `load_breast_cancer`",
            "license": "CC BY 4.0 (UCI) / BSD-3-Clause (scikit-learn)",
            "notes": "31 columns with spaced names testing wide schema handling and binary target",
            "sha256": _sha256(bc_dst),
        }
    )

    # 4. UCI Wine Cultivars
    wine = load_wine(as_frame=True)
    wine_df = wine.frame.copy()
    wine_df["cultivar"] = wine_df["target"].map({0: "class_0", 1: "class_1", 2: "class_2"})
    wine_df = wine_df.drop(columns=["target"])
    wine_dst = DATASET_DIR / "wine_cultivars_uci.csv"
    wine_df.to_csv(wine_dst, index=False)
    records.append(
        {
            "file": wine_dst.name,
            "rows": str(len(wine_df)),
            "cols": str(wine_df.shape[1]),
            "source": "UCI ML Repository via scikit-learn `load_wine`",
            "license": "CC BY 4.0 (UCI) / BSD-3-Clause (scikit-learn)",
            "notes": "13 chemical measurements across 3 grape cultivars",
            "sha256": _sha256(wine_dst),
        }
    )

    # 5. US Macroeconomic Quarterly Time Series (FRED / Statsmodels)
    macro_df = macrodata.load_pandas().data.copy()
    macro_df["year"] = macro_df["year"].astype(int)
    macro_df["quarter"] = macro_df["quarter"].astype(int)
    macro_dst = DATASET_DIR / "us_macro_fred.csv"
    macro_df.to_csv(macro_dst, index=False)
    records.append(
        {
            "file": macro_dst.name,
            "rows": str(len(macro_df)),
            "cols": str(macro_df.shape[1]),
            "source": "Federal Reserve / Statsmodels `datasets.macrodata` (1959Q1-2009Q3)",
            "license": "Public Domain (US Gov) / BSD-3-Clause (statsmodels)",
            "notes": "Quarterly macroeconomic time series (GDP, CPI, unemployment, inflation)",
            "sha256": _sha256(macro_dst),
        }
    )

    # 6. Mauna Loa Atmospheric CO2 (NOAA / Scripps via Statsmodels)
    co2_df = co2.load_pandas().data.copy().reset_index()
    co2_df = co2_df.rename(columns={"index": "date"})
    co2_df["date"] = pd.to_datetime(co2_df["date"]).dt.strftime("%Y-%m-%d")
    co2_df["year"] = pd.to_datetime(co2_df["date"]).dt.year
    co2_dst = DATASET_DIR / "mauna_loa_co2.csv"
    co2_df.to_csv(co2_dst, index=False)
    records.append(
        {
            "file": co2_dst.name,
            "rows": str(len(co2_df)),
            "cols": str(co2_df.shape[1]),
            "source": "NOAA ESRL Mauna Loa Observatory via statsmodels `datasets.co2`",
            "license": "Public Domain (NOAA) / BSD-3-Clause (statsmodels)",
            "notes": (
                "Weekly atmospheric CO2 concentration (1958-2001) with real missing observations"
            ),
            "sha256": _sha256(co2_dst),
        }
    )

    # 7. Independent Dataset 1: Clinical Diabetes Progression (Efron et al. 2004 via scikit-learn)
    diab = load_diabetes(as_frame=True, scaled=False)
    diab_df = diab.frame.copy()
    diab_df = diab_df.rename(
        columns={
            "s1": "total_cholesterol",
            "s2": "ldl_cholesterol",
            "s3": "hdl_cholesterol",
            "s4": "tch_ratio",
            "s5": "log_triglycerides",
            "s6": "blood_glucose",
            "target": "disease_progression",
        }
    )
    diab_df["sex_label"] = diab_df["sex"].map({1.0: "female", 2.0: "male"})
    diab_df = diab_df.drop(columns=["sex"])
    diab_df["bmi_tier"] = np.where(
        diab_df["bmi"] >= 30.0,
        "obese",
        np.where(diab_df["bmi"] >= 25.0, "overweight", "normal"),
    )
    diab_df["high_bp"] = diab_df["bp"] >= 100.0
    # Deterministic clinical lab missingness (missing lab panels on fixed row indices)
    diab_df.loc[diab_df.index % 37 == 5, "ldl_cholesterol"] = np.nan
    diab_df.loc[diab_df.index % 53 == 11, "blood_glucose"] = np.nan
    diab_dst = DATASET_DIR / "diabetes_clinical_progression.csv"
    diab_df.to_csv(diab_dst, index=False)
    records.append(
        {
            "file": diab_dst.name,
            "rows": str(len(diab_df)),
            "cols": str(diab_df.shape[1]),
            "source": "Efron et al. (2004) LARS / scikit-learn `load_diabetes(scaled=False)`",
            "license": "BSD-3-Clause (scikit-learn)",
            "notes": (
                "Out-of-dataset clinical biomarker cohort (442 patients, 13 columns) with "
                "unscaled serum measurements, BMI tiers, boolean hypertension flag, and "
                "structured lab missingness"
            ),
            "sha256": _sha256(diab_dst),
        }
    )

    # 8. Independent Dataset 2: Grid Substation Telemetry & Stack-Loss Process Cohort
    rng = np.random.default_rng(20260415)
    n_grid = 240
    zones = np.array(["North_Grid", "South_Grid", "Coastal_Grid", "Inland_Grid"])
    cooling_types = np.array(["forced_air", "oil_natural", "liquid_immersion"])
    grid_zone = zones[np.arange(n_grid) % len(zones)]
    cooling_mode = cooling_types[(np.arange(n_grid) // 3) % len(cooling_types)]
    hour_of_day = (np.arange(n_grid) * 3) % 24
    ambient_c = np.round(
        14.0 + 9.5 * np.sin(np.arange(n_grid) / 18.0) + rng.normal(0, 2.2, n_grid), 2
    )
    load_mva = np.round(
        np.where(grid_zone == "North_Grid", 68.0, np.where(grid_zone == "Coastal_Grid", 54.0, 42.0))
        + 0.45 * ambient_c
        + rng.gamma(shape=3.0, scale=3.5, size=n_grid),
        2,
    )
    winding_temp_c = np.round(
        38.0 + 0.72 * load_mva + 0.55 * ambient_c + rng.normal(0, 1.8, n_grid), 2
    )
    harmonic_thd_pct = np.round(np.exp(rng.normal(0.85, 0.42, n_grid)) + 0.03 * load_mva, 3)
    # Inject extreme transformer fault spikes
    harmonic_thd_pct[[17, 89, 143, 211]] = [19.4, 22.8, 25.1, 21.6]
    alarm_tripped = (winding_temp_c > 98.0) | (harmonic_thd_pct > 12.0)

    grid_df = pd.DataFrame(
        {
            "substation_id": [f"SUB_{i:03d}" for i in range(1, n_grid + 1)],
            "grid_zone": grid_zone,
            "cooling_mode": cooling_mode,
            "hour_of_day": hour_of_day,
            "ambient_temp_c": ambient_c,
            "load_mva": load_mva,
            "winding_temp_c": winding_temp_c,
            "harmonic_thd_pct": harmonic_thd_pct,
            "alarm_tripped": alarm_tripped,
        }
    )
    # Deterministic telemetry sensor dropouts
    grid_df.loc[grid_df.index % 29 == 3, "harmonic_thd_pct"] = np.nan
    grid_df.loc[grid_df.index % 41 == 7, "ambient_temp_c"] = np.nan
    grid_dst = DATASET_DIR / "grid_substation_telemetry.csv"
    grid_df.to_csv(grid_dst, index=False)
    records.append(
        {
            "file": grid_dst.name,
            "rows": str(len(grid_df)),
            "cols": str(grid_df.shape[1]),
            "source": "Deterministic Industrial Substation Telemetry Benchmark (`seed=20260415`)",
            "license": "MIT",
            "notes": (
                "Out-of-dataset industrial power-grid telemetry (240 readings, 9 columns) with "
                "log-normal harmonic distortion, fault spikes, boolean alarms, and sensor dropouts"
            ),
            "sha256": _sha256(grid_dst),
        }
    )

    lines = [
        "# Evaluation Dataset Provenance",
        "",
        "All six datasets used in SpectrumBloom's 60+ task evaluation suite are materialized",
        "reproducibly via `python benchmarks/prepare_datasets.py`.",
        "",
        "| Dataset | Rows | Cols | Source | License | SHA-256 | Notes |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for rec in records:
        lines.append(
            f"| `{rec['file']}` | {rec['rows']} | {rec['cols']} | {rec['source']} | "
            f"{rec['license']} | `{rec['sha256'][:16]}...` | {rec['notes']} |"
        )
    lines.extend(["", "## Full SHA-256 Checksums", ""])
    for rec in records:
        lines.append(f"- `{rec['file']}`: `{rec['sha256']}`")

    (DATASET_DIR / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    rows = build_datasets()
    for row in rows:
        print(f"{row['file']}: {row['rows']}x{row['cols']} sha256={row['sha256'][:12]}...")
