"""Materialize the six diverse evaluation datasets with documented provenance and SHA-256 hashes."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
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
