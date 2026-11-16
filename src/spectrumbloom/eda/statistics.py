"""Statistical EDA diagnostics: missingness, normality, correlations, outliers, group tests."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from spectrumbloom.domain.schemas import (
    CorrelationPair,
    DatasetProfile,
    MissingnessReport,
    NormalityCheck,
    OutlierSummary,
    StatisticalReport,
)


def _clean_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 5)


def _compute_missingness(df: pd.DataFrame) -> MissingnessReport:
    rows, cols = df.shape
    total_cells = rows * cols
    null_counts = df.isna().sum()
    total_missing = int(null_counts.sum())
    rows_with_any = int(df.isna().any(axis=1).sum())

    columns_with_missing = [
        {
            "column": str(col),
            "missing_count": int(cnt),
            "missing_pct": round((int(cnt) / rows) * 100.0, 2) if rows else 0.0,
        }
        for col, cnt in null_counts.items()
        if cnt > 0
    ]
    columns_with_missing.sort(key=lambda item: item["missing_count"], reverse=True)

    cooccurrence_pairs: list[dict[str, Any]] = []
    missing_cols = [item["column"] for item in columns_with_missing[:12]]
    if len(missing_cols) >= 2:
        null_indicator = df[missing_cols].isna().astype(float)
        corr = null_indicator.corr()
        for i, col_a in enumerate(missing_cols):
            for col_b in missing_cols[i + 1 :]:
                val = _clean_float(corr.loc[col_a, col_b])
                if val is not None and abs(val) >= 0.25:
                    cooccurrence_pairs.append(
                        {"column_a": col_a, "column_b": col_b, "null_correlation": val}
                    )
        cooccurrence_pairs.sort(key=lambda x: abs(x["null_correlation"]), reverse=True)

    return MissingnessReport(
        total_cells=total_cells,
        total_missing_cells=total_missing,
        overall_missing_pct=round((total_missing / total_cells) * 100.0, 2) if total_cells else 0.0,
        rows_with_any_missing=rows_with_any,
        rows_with_any_missing_pct=round((rows_with_any / rows) * 100.0, 2) if rows else 0.0,
        columns_with_missing=columns_with_missing,
        missing_cooccurrence_pairs=cooccurrence_pairs[:10],
    )


def _cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    contingency = pd.crosstab(x, y)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1, n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return _clean_float(math.sqrt(phi2corr / denom))


def compute_statistics(df: pd.DataFrame, profile: DatasetProfile) -> StatisticalReport:
    """Compute missingness, distributions, correlations, outliers, and group hypothesis tests."""
    missingness = _compute_missingness(df)

    numeric_cols = [
        cp.name
        for cp in profile.column_profiles
        if cp.semantic_type in {"numeric_continuous", "numeric_discrete"}
    ]
    cat_cols = [
        cp.name
        for cp in profile.column_profiles
        if cp.semantic_type in {"categorical", "boolean"} and cp.unique_count >= 2
    ]

    # 1. Correlations (Pearson + Spearman)
    pearson_matrix: dict[str, dict[str, float | None]] = {}
    spearman_matrix: dict[str, dict[str, float | None]] = {}
    top_correlations: list[CorrelationPair] = []

    if len(numeric_cols) >= 2:
        num_df = df[numeric_cols].astype(float)
        p_corr = num_df.corr(method="pearson")
        s_corr = num_df.corr(method="spearman")

        for col in numeric_cols:
            pearson_matrix[col] = {
                other: _clean_float(p_corr.loc[col, other]) for other in numeric_cols
            }
            spearman_matrix[col] = {
                other: _clean_float(s_corr.loc[col, other]) for other in numeric_cols
            }

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                pr = _clean_float(p_corr.loc[col_a, col_b])
                sr = _clean_float(s_corr.loc[col_a, col_b])
                if (pr is not None and abs(pr) >= 0.25) or (sr is not None and abs(sr) >= 0.25):
                    top_correlations.append(
                        CorrelationPair(
                            column_a=col_a,
                            column_b=col_b,
                            pearson_r=pr,
                            spearman_rho=sr,
                        )
                    )
        top_correlations.sort(
            key=lambda pair: max(abs(pair.pearson_r or 0.0), abs(pair.spearman_rho or 0.0)),
            reverse=True,
        )

    # 2. Categorical associations (Cramer's V)
    cat_associations: list[dict[str, Any]] = []
    usable_cats = [c for c in cat_cols if df[c].nunique(dropna=True) <= 25][:8]
    for i, col_a in enumerate(usable_cats):
        for col_b in usable_cats[i + 1 :]:
            v = _cramers_v(df[col_a].dropna(), df[col_b].dropna())
            if v is not None and v >= 0.15:
                cat_associations.append({"column_a": col_a, "column_b": col_b, "cramers_v": v})
    cat_associations.sort(key=lambda item: item["cramers_v"], reverse=True)

    # 3. Outliers (IQR, Z-score, robust MAD) + Normality checks
    outliers: list[OutlierSummary] = []
    normality_checks: list[NormalityCheck] = []

    for col in numeric_cols:
        series = df[col].dropna().astype(float)
        n = len(series)
        if n < 4:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_cnt = int(((series < lower) | (series > upper)).sum())

        std = float(series.std(ddof=1))
        if std > 0:
            z_cnt = int((((series - float(series.mean())) / std).abs() > 3.0).sum())
        else:
            z_cnt = 0

        med = float(series.median())
        mad = float((series - med).abs().median())
        if mad > 0:
            mod_z = 0.6745 * (series - med).abs() / mad
            mad_cnt = int((mod_z > 3.5).sum())
        else:
            mad_cnt = 0

        outliers.append(
            OutlierSummary(
                column=col,
                iqr_lower=round(lower, 4),
                iqr_upper=round(upper, 4),
                iqr_outlier_count=iqr_cnt,
                iqr_outlier_pct=round((iqr_cnt / n) * 100.0, 2),
                zscore_outlier_count=z_cnt,
                mad_outlier_count=mad_cnt,
            )
        )

        # Normality test
        skew_val = _clean_float(series.skew())
        kurt_val = _clean_float(series.kurt())
        stat_val = None
        p_val = None
        test_name = "insufficient_variance"
        if series.nunique() > 3 and std > 0:
            if n <= 5000:
                test_name = "shapiro_wilk"
                res = stats.shapiro(series)
                stat_val, p_val = _clean_float(res.statistic), _clean_float(res.pvalue)
            elif n >= 8:
                test_name = "dagostino_pearson"
                res = stats.normaltest(series)
                stat_val, p_val = _clean_float(res.statistic), _clean_float(res.pvalue)

        normality_checks.append(
            NormalityCheck(
                column=col,
                sample_size=n,
                skewness=skew_val,
                kurtosis=kurt_val,
                test_name=test_name,
                statistic=stat_val,
                p_value=p_val,
                is_normal_alpha_05=(p_val >= 0.05) if p_val is not None else None,
            )
        )

    # 4. Multivariate outlier scan via IsolationForest
    multivariate_anomalies = 0
    if len(numeric_cols) >= 2 and len(df) >= 25:
        try:
            X_num = df[numeric_cols].astype(float)
            valid_cols = [c for c in numeric_cols if X_num[c].notna().sum() >= 10]
            if len(valid_cols) >= 2:
                imputed = SimpleImputer(strategy="median").fit_transform(X_num[valid_cols])
                iso = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
                preds = iso.fit_predict(imputed)
                multivariate_anomalies = int((preds == -1).sum())
        except Exception:
            multivariate_anomalies = 0

    # 5. Automated group difference tests (Kruskal-Wallis across low-cardinality categoricals)
    group_tests: list[dict[str, Any]] = []
    low_card_cats = [c for c in cat_cols if 2 <= df[c].nunique(dropna=True) <= 8][:4]
    for cat_col in low_card_cats:
        for num_col in numeric_cols[:5]:
            subset = df[[cat_col, num_col]].dropna()
            groups = [
                grp[num_col].astype(float).to_numpy()
                for _, grp in subset.groupby(cat_col, observed=True)
                if len(grp) >= 3
            ]
            if len(groups) < 2:
                continue
            try:
                stat, p_value = stats.kruskal(*groups)
                p_clean = _clean_float(p_value)
                if p_clean is not None and p_clean < 0.05:
                    group_tests.append(
                        {
                            "categorical_column": cat_col,
                            "numeric_column": num_col,
                            "test": "kruskal_wallis",
                            "group_count": len(groups),
                            "statistic": _clean_float(stat),
                            "p_value": p_clean,
                        }
                    )
            except ValueError:
                continue

    group_tests.sort(key=lambda item: item["p_value"])

    return StatisticalReport(
        missingness=missingness,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        top_correlations=top_correlations[:25],
        categorical_associations=cat_associations[:10],
        outliers=outliers,
        multivariate_anomaly_count=multivariate_anomalies,
        normality_checks=normality_checks,
        group_tests=group_tests[:10],
    )
