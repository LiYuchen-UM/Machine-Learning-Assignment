"""Run an auto-sklearn benchmark for the dating app drop-off risk project.

This script is intended for Linux/WSL, not native Windows Python.
It uses the same target definition as the notebook:
dropoff_risk = 1 for Ghosted or Chat Ignored, otherwise 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import autosklearn.classification
import autosklearn.metrics


RANDOM_STATE = 42
DROPOFF_OUTCOMES = ["Ghosted", "Chat Ignored"]


def split_interest_tags(text):
    if pd.isna(text):
        return []
    return [tag.strip() for tag in str(text).split(",") if tag.strip()]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x_frame.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        col for col in x_frame.select_dtypes(exclude=np.number).columns.tolist()
        if col != "interest_tags"
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
            (
                "interest_tags",
                CountVectorizer(
                    tokenizer=split_interest_tags,
                    token_pattern=None,
                    binary=True,
                    lowercase=False,
                ),
                "interest_tags",
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/dating_app_behavior_dataset.csv",
        help="Path to dating_app_behavior_dataset.csv.",
    )
    parser.add_argument(
        "--time-left",
        type=int,
        default=600,
        help="Total auto-sklearn search time in seconds.",
    )
    parser.add_argument(
        "--per-run-time-limit",
        type=int,
        default=60,
        help="Time limit per model run in seconds.",
    )
    parser.add_argument(
        "--output",
        default="autosklearn_results.json",
        help="Path to write JSON metrics.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["dropoff_risk"] = df["match_outcome"].isin(DROPOFF_OUTCOMES).astype(int)

    x = df.drop(columns=["match_outcome", "dropoff_risk"])
    y = df["dropoff_risk"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_auto = preprocessor.fit_transform(x_train)
    x_test_auto = preprocessor.transform(x_test)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=args.time_left,
        per_run_time_limit=args.per_run_time_limit,
        metric=autosklearn.metrics.f1,
        seed=RANDOM_STATE,
        n_jobs=-1,
    )
    automl.fit(x_train_auto, y_train)

    y_pred = automl.predict(x_test_auto)
    y_scores = automl.predict_proba(x_test_auto)[:, 1]

    results = {
        "status": "completed",
        "time_left_for_this_task": args.time_left,
        "per_run_time_limit": args.per_run_time_limit,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "average_precision": average_precision_score(y_test, y_scores),
        "sprint_statistics": automl.sprint_statistics(),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

