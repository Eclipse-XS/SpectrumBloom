"""Leak-safe scikit-learn baseline ML experiments with cross-validation and Plotly diagnostics."""

from __future__ import annotations

import json
import math
import uuid
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from spectrumbloom.core.security import sanitize_json_payload
from spectrumbloom.domain.exceptions import MLExperimentError
from spectrumbloom.domain.schemas import (
    MLExperimentRequest,
    MLExperimentResponse,
    MLModelResult,
)


def _encode_plot(fig: Any, title: str, chart_kind: str) -> dict[str, Any]:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02},
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
    )
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "chart_kind": chart_kind,
        "figure_json": json.dumps(sanitize_json_payload(fig.to_dict())),
    }


def _infer_task_type(target: pd.Series) -> Literal["classification", "regression"]:
    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return "classification"
    unique_count = int(target.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(target) and unique_count <= 15:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [
        col
        for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
    ]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=25,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return (
        ColumnTransformer(transformers=transformers, remainder="drop"),
        numeric_features,
        categorical_features,
    )


def run_ml_experiment(
    df: pd.DataFrame,
    request: MLExperimentRequest,
) -> MLExperimentResponse:
    """Execute a leak-safe baseline ML benchmark on the active dataset."""
    if request.target_column not in df.columns:
        raise MLExperimentError(f"Target column '{request.target_column}' not found in dataset.")

    work_df = df.dropna(subset=[request.target_column]).copy()
    if len(work_df) < 25:
        raise MLExperimentError(
            f"Need >= 25 non-null rows for target '{request.target_column}' (got {len(work_df)})."
        )

    notes: list[str] = []
    candidate_features = (
        [c for c in request.feature_columns if c in work_df.columns and c != request.target_column]
        if request.feature_columns
        else [c for c in work_df.columns if c != request.target_column]
    )

    # Drop identifier-like or all-null or constant columns
    usable_features: list[str] = []
    for col in candidate_features:
        non_null = work_df[col].dropna()
        if non_null.empty or non_null.nunique() <= 1:
            notes.append(f"Dropped constant/empty feature '{col}'.")
            continue
        col_lower = col.lower()
        if non_null.nunique() == len(work_df) and ("id" in col_lower or "uuid" in col_lower):
            notes.append(f"Dropped identifier-like column '{col}' to prevent leakage.")
            continue
        usable_features.append(col)

    if not usable_features:
        raise MLExperimentError("No usable predictor columns remain after filtering constants/IDs.")

    y_raw = work_df[request.target_column]
    task_type: Literal["classification", "regression"] = (
        _infer_task_type(y_raw) if request.task_type == "auto" else request.task_type
    )

    X = work_df[usable_features].copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(str)

    label_encoder: LabelEncoder | None = None
    if task_type == "classification":
        if y_raw.nunique() < 2:
            raise MLExperimentError("Classification target needs at least 2 distinct classes.")
        if y_raw.nunique() > 25:
            raise MLExperimentError(
                f"Classification target has {y_raw.nunique()} unique classes (max 25 supported)."
            )
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        min_class_count = int(pd.Series(y).value_counts().min())
        stratify_y = y if min_class_count >= 2 else None
        if stratify_y is None:
            notes.append("Rare class has <2 samples; disabled stratified split.")
    else:
        try:
            y = y_raw.astype(float).to_numpy()
        except (TypeError, ValueError) as exc:
            raise MLExperimentError(
                f"Regression target '{request.target_column}' must be numeric."
            ) from exc
        stratify_y = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=request.test_size,
        random_state=request.random_state,
        stratify=stratify_y,
    )

    preprocessor, _, _ = _build_preprocessor(X_train)

    if task_type == "classification":
        effective_folds = min(request.cv_folds, int(pd.Series(y_train).value_counts().min()))
        if effective_folds < 2:
            effective_folds = 2
            cv = KFold(n_splits=effective_folds, shuffle=True, random_state=request.random_state)
        else:
            cv = StratifiedKFold(
                n_splits=effective_folds, shuffle=True, random_state=request.random_state
            )
        scoring = "f1_weighted"
        candidates = [
            ("Dummy (Majority)", DummyClassifier(strategy="most_frequent")),
            (
                "Logistic Regression",
                LogisticRegression(max_iter=1000, random_state=request.random_state),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=120, random_state=request.random_state, n_jobs=1
                ),
            ),
            (
                "HistGradientBoosting",
                HistGradientBoostingClassifier(random_state=request.random_state),
            ),
        ]
    else:
        effective_folds = min(request.cv_folds, len(X_train))
        cv = KFold(
            n_splits=max(2, effective_folds), shuffle=True, random_state=request.random_state
        )
        scoring = "r2"
        candidates = [
            ("Dummy (Mean)", DummyRegressor(strategy="mean")),
            ("Ridge Regression", Ridge(random_state=request.random_state)),
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=120, random_state=request.random_state, n_jobs=1
                ),
            ),
            (
                "HistGradientBoosting",
                HistGradientBoostingRegressor(random_state=request.random_state),
            ),
        ]

    model_results: list[MLModelResult] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    best_model_name = ""
    best_score = -float("inf")

    for model_name, estimator in candidates:
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        pipe.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipe
        preds = pipe.predict(X_test)

        if task_type == "classification":
            roc_auc: float | None = None
            if len(np.unique(y_test)) == 2 and hasattr(pipe, "predict_proba"):
                try:
                    probs = pipe.predict_proba(X_test)[:, 1]
                    roc_auc = round(float(roc_auc_score(y_test, probs)), 4)
                except Exception:
                    roc_auc = None
            test_metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
                "roc_auc": roc_auc,
            }
        else:
            test_metrics = {
                "rmse": round(float(root_mean_squared_error(y_test, preds)), 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                "r2": round(float(r2_score(y_test, preds)), 4),
            }

        model_results.append(
            MLModelResult(
                model_name=model_name,
                cv_primary_metric_name=scoring,
                cv_primary_mean=round(cv_mean, 4),
                cv_primary_std=round(cv_std, 4),
                test_metrics=test_metrics,
            )
        )

        if not math.isnan(cv_mean) and cv_mean > best_score:
            best_score = cv_mean
            best_model_name = model_name

    best_pipe = fitted_pipelines[best_model_name]

    # Permutation importance on held-out test set (model-agnostic & leak-free)
    perm = permutation_importance(
        best_pipe,
        X_test,
        y_test,
        n_repeats=5,
        random_state=request.random_state,
        scoring=scoring,
    )
    feature_importances = sorted(
        [
            {
                "feature": feat,
                "importance_mean": round(float(mean_val), 5),
                "importance_std": round(float(std_val), 5),
            }
            for feat, mean_val, std_val in zip(
                usable_features, perm.importances_mean, perm.importances_std, strict=False
            )
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )

    figures: list[dict[str, Any]] = []

    # Chart 1: CV Model comparison
    comp_df = pd.DataFrame(
        [
            {
                "model": m.model_name,
                "cv_score": m.cv_primary_mean,
                "cv_std": m.cv_primary_std,
            }
            for m in model_results
        ]
    )
    fig_comp = px.bar(
        comp_df,
        x="model",
        y="cv_score",
        error_y="cv_std",
        color="cv_score",
        color_continuous_scale="Teal",
        labels={"cv_score": f"CV {scoring}", "model": "Model"},
    )
    figures.append(_encode_plot(fig_comp, f"Model Comparison ({scoring.upper()})", "ml_comparison"))

    # Chart 2: Feature importances
    imp_df = pd.DataFrame(feature_importances[:15])
    if not imp_df.empty:
        fig_imp = px.bar(
            imp_df,
            x="importance_mean",
            y="feature",
            orientation="h",
            color="importance_mean",
            color_continuous_scale="Viridis",
            labels={"importance_mean": f"Permutation Importance ({scoring})", "feature": "Feature"},
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        figures.append(
            _encode_plot(fig_imp, f"Feature Importance — {best_model_name}", "ml_importance")
        )

    # Chart 3: Task diagnostic (confusion matrix or predicted vs actual)
    best_preds = best_pipe.predict(X_test)
    if task_type == "classification" and label_encoder is not None:
        labels_idx = sorted(np.unique(np.concatenate([y_test, best_preds])))
        class_names = [str(label_encoder.inverse_transform([i])[0]) for i in labels_idx]
        cm = confusion_matrix(y_test, best_preds, labels=labels_idx)
        fig_diag = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        )
        figures.append(
            _encode_plot(fig_diag, f"Confusion Matrix — {best_model_name}", "ml_diagnostic")
        )
    else:
        diag_df = pd.DataFrame({"actual": y_test, "predicted": best_preds})
        fig_diag = px.scatter(
            diag_df,
            x="actual",
            y="predicted",
            opacity=0.75,
            color_discrete_sequence=["#0F766E"],
            labels={"actual": f"Actual ({request.target_column})", "predicted": "Predicted"},
        )
        min_val = float(min(diag_df["actual"].min(), diag_df["predicted"].min()))
        max_val = float(max(diag_df["actual"].max(), diag_df["predicted"].max()))
        fig_diag.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line={"color": "#DC2626", "dash": "dash"},
        )
        figures.append(
            _encode_plot(fig_diag, f"Actual vs Predicted — {best_model_name}", "ml_diagnostic")
        )

    return MLExperimentResponse(
        target_column=request.target_column,
        task_type=task_type,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_count=len(usable_features),
        best_model=best_model_name,
        models=model_results,
        feature_importances=feature_importances,
        figures=figures,
        notes=notes,
    )
