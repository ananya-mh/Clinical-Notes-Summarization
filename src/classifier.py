"""
Section-Type Classifier
========================
Trains Logistic Regression, Random Forest, and XGBoost to predict
section_header from entity-based features. Uses stratified k-fold CV.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "classification")

FEATURE_COLS_BASE = [
    "symptom_count", "medication_count", "diagnosis_count", "procedure_count",
    "other_count", "total_entities", "dialogue_word_count", "summary_word_count",
    "speaker_turn_count", "unique_entity_ratio",
]


def load_features(split_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"features_{split_name}.csv")
    if not os.path.exists(path):
        print(f"Feature file not found: {path}")
        print("Run feature_engineer.py first.")
        return None
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    has_cols = [c for c in df.columns if c.startswith("has_")]
    return FEATURE_COLS_BASE + has_cols


def prepare_data(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df["section_header"])
    return X, y, le, feature_cols


def run_cross_validation(X, y, le):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss",
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro"]

    results = []
    print(f"\n{'='*60}")
    print("STRATIFIED 5-FOLD CROSS-VALIDATION (on train)")
    print(f"{'='*60}")

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        acc = scores["test_accuracy"]
        f1 = scores["test_f1_macro"]

        print(f"\n--- {name} ---")
        print(f"  Accuracy:  {acc.mean():.4f} (+/- {acc.std():.4f})")
        print(f"  Macro F1:  {f1.mean():.4f} (+/- {f1.std():.4f})")

        results.append({
            "model": name,
            "accuracy_mean": round(acc.mean(), 4),
            "accuracy_std": round(acc.std(), 4),
            "f1_macro_mean": round(f1.mean(), 4),
            "f1_macro_std": round(f1.std(), 4),
        })

    return results, models


def train_final_model(model, X_train, y_train, X_test, y_test, le, model_name, feature_cols):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {model_name} (train → test)")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    labels = le.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png"), dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {RESULTS_DIR}/confusion_matrix_{model_name}.png")

    return model, acc, f1


def save_results(cv_results, test_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nCross-validation (train):")
    print(cv_df.to_string(index=False))
    print("\nTest set:")
    print(test_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Train section-type classifiers")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    args = parser.parse_args()

    train_df = load_features(args.train_split)
    test_df = load_features(args.test_split)
    if train_df is None or test_df is None:
        return

    feature_cols = get_feature_columns(train_df)
    test_feature_cols = get_feature_columns(test_df)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    for col in test_feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0
    feature_cols = get_feature_columns(train_df)

    X_train, y_train, le, _ = prepare_data(train_df)
    X_test = test_df[feature_cols].fillna(0).values
    y_test = le.transform(test_df["section_header"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} examples, {X_test.shape[1]} features")
    print(f"Classes: {list(le.classes_)}")

    cv_results, models = run_cross_validation(X_train_scaled, y_train, le)

    best = max(cv_results, key=lambda x: x["f1_macro_mean"])
    print(f"\nBest CV model: {best['model']} (macro F1 = {best['f1_macro_mean']:.4f})")

    test_results = []
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        if name == "LogisticRegression":
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        trained, acc, f1 = train_final_model(
            model, X_tr, y_train, X_te, y_test, le, name, feature_cols,
        )
        test_results.append({"model": name, "accuracy": acc, "f1_macro": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = trained

    save_results(cv_results, test_results)

    import joblib
    best_name = max(test_results, key=lambda x: x["f1_macro"])["model"]
    model_path = os.path.join(RESULTS_DIR, "best_classifier.joblib")
    joblib.dump(best_model, model_path)

    meta = {
        "model_name": best_name,
        "feature_cols": feature_cols,
        "classes": list(le.classes_),
    }
    meta_path = os.path.join(RESULTS_DIR, "best_classifier_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved best model ({best_name}) → {model_path}")


if __name__ == "__main__":
    main()
