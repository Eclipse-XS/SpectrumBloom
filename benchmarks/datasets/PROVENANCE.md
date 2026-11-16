# Evaluation Dataset Provenance

All six datasets used in SpectrumBloom's 60+ task evaluation suite are materialized
reproducibly via `python benchmarks/prepare_datasets.py`.

| Dataset | Rows | Cols | Source | License | SHA-256 | Notes |
|---|---:|---:|---|---|---|---|
| `sample_ecommerce.csv` | 180 | 12 | SpectrumBloom repository sample (`examples/sample_ecommerce.csv`) | MIT | `05a6dab3946e2d32...` | E-commerce orders with missingness, date column, outliers, binary churned |
| `palmer_penguins.csv` | 344 | 7 | Palmer Station Antarctica LTER / seaborn-data (`penguins.csv`) | CC0 1.0 Universal | `e07636bd8af74260...` | 3 penguin species across 3 islands with natural missing numeric & sex values |
| `breast_cancer_uci.csv` | 569 | 31 | UCI ML Repository via scikit-learn `load_breast_cancer` | CC BY 4.0 (UCI) / BSD-3-Clause (scikit-learn) | `43c8b3fc35af48eb...` | 31 columns with spaced names testing wide schema handling and binary target |
| `wine_cultivars_uci.csv` | 178 | 14 | UCI ML Repository via scikit-learn `load_wine` | CC BY 4.0 (UCI) / BSD-3-Clause (scikit-learn) | `41b2f77f59e21cd6...` | 13 chemical measurements across 3 grape cultivars |
| `us_macro_fred.csv` | 203 | 14 | Federal Reserve / Statsmodels `datasets.macrodata` (1959Q1-2009Q3) | Public Domain (US Gov) / BSD-3-Clause (statsmodels) | `eb31aa95d84d493d...` | Quarterly macroeconomic time series (GDP, CPI, unemployment, inflation) |
| `mauna_loa_co2.csv` | 2284 | 3 | NOAA ESRL Mauna Loa Observatory via statsmodels `datasets.co2` | Public Domain (NOAA) / BSD-3-Clause (statsmodels) | `082eba31fbb6962b...` | Weekly atmospheric CO2 concentration (1958-2001) with real missing observations |
| `diabetes_clinical_progression.csv` | 442 | 13 | Efron et al. (2004) LARS / scikit-learn `load_diabetes(scaled=False)` | BSD-3-Clause (scikit-learn) | `3674ffd3057c53a0...` | Out-of-dataset clinical biomarker cohort (442 patients, 13 columns) with unscaled serum measurements, BMI tiers, boolean hypertension flag, and structured lab missingness |
| `grid_substation_telemetry.csv` | 240 | 9 | Deterministic Industrial Substation Telemetry Benchmark (`seed=20260415`) | MIT | `362d08c77b2ea1d4...` | Out-of-dataset industrial power-grid telemetry (240 readings, 9 columns) with log-normal harmonic distortion, fault spikes, boolean alarms, and sensor dropouts |

## Full SHA-256 Checksums

- `sample_ecommerce.csv`: `05a6dab3946e2d32010758ced5c15ae6a11b509cd12992a6662baf50fb0aebb2`
- `palmer_penguins.csv`: `e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1`
- `breast_cancer_uci.csv`: `43c8b3fc35af48eb32103fe187eb2f6e1541b5f3cfbdc2376c171a59f990aeb2`
- `wine_cultivars_uci.csv`: `41b2f77f59e21cd616ddc2d955d4082b9043754b26b26f9860f036046b2e2c66`
- `us_macro_fred.csv`: `eb31aa95d84d493ddb87837511eb04bfceb8fb5d634f2ae8e1f24ecd55cfa0da`
- `mauna_loa_co2.csv`: `082eba31fbb6962b6fc491d24e006232a51fec9c880f64be2aca0d9dfb7013a7`
- `diabetes_clinical_progression.csv`: `3674ffd3057c53a043acef82ea463bf752c557d8071c16df5f0d264a53f9a4b9`
- `grid_substation_telemetry.csv`: `362d08c77b2ea1d4cf0eca119d6902bca6f5cf43f668f4e67382babd2e411b32`
