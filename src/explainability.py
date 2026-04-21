"""
SHAP Explainability
====================
Generates SHAP summary, dependence, and force plots for the best
section-type classifier.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "classification")


def load_model_and_meta():
    model_path = os.path.join(RESULTS_DIR, "best_classifier.joblib")
    meta_path = os.path.join(RESULTS_DIR, "best_classifier_meta.json")

    if not os.path.exists(model_path):
        print(f"No saved model at {model_path}. Run classifier.py first.")
        return None, None

    model = joblib.load(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


def load_data(split_name: str, feature_cols: list[str]):
    path = os.path.join(PROCESSED_DIR, f"features_{split_name}.csv")
    if not os.path.exists(path):
        print(f"Feature file not found: {path}")
        return None, None

    df = pd.read_csv(path)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)
    y = df["section_header"]
    return X, y


def summary_plot(shap_values, X, classes, output_dir):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=classes, show=False)
    plt.title("SHAP Feature Importance (All Classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → shap_summary_bar.png")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary (Dot Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → shap_summary_dot.png")


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def per_class_plots(shap_values, X, classes, output_dir):
    class_dir = os.path.join(output_dir, "per_class")
    os.makedirs(class_dir, exist_ok=True)

    for i, cls in enumerate(classes):
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values[:, :, i]
        shap.summary_plot(sv, X, show=False, plot_type="dot")
        plt.title(f"SHAP — {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, f"shap_{safe_filename(cls)}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved per-class plots → {class_dir}/")


def dependence_plots(shap_values, X, classes, output_dir, top_n=4):
    dep_dir = os.path.join(output_dir, "dependence")
    os.makedirs(dep_dir, exist_ok=True)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))

    top_features = X.columns[np.argsort(mean_abs)[::-1][:top_n]]

    for feat in top_features:
        plt.figure(figsize=(8, 5))
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values[:, :, 0]
        shap.dependence_plot(feat, sv, X, show=False)
        plt.title(f"SHAP Dependence — {feat}")
        plt.tight_layout()
        plt.savefig(os.path.join(dep_dir, f"dep_{feat}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved dependence plots → {dep_dir}/")


def force_plots(explainer, shap_values, X, y, classes, output_dir, n_examples=3):
    force_dir = os.path.join(output_dir, "force_plots")
    os.makedirs(force_dir, exist_ok=True)

    le = LabelEncoder()
    le.fit(classes)

    for cls in classes:
        idx = y[y == cls].index[:n_examples]
        for j, i in enumerate(idx):
            if isinstance(shap_values, list):
                cls_idx = list(classes).index(cls)
                sv = shap_values[cls_idx][X.index.get_loc(i)]
            else:
                cls_idx = list(classes).index(cls)
                sv = shap_values[X.index.get_loc(i), :, cls_idx]

            force = shap.force_plot(
                explainer.expected_value[list(classes).index(cls)] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                sv,
                X.loc[i],
                matplotlib=True,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(force_dir, f"force_{safe_filename(cls)}_{j}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Saved force plots → {force_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability analysis")
    parser.add_argument("--split", default="test", help="Split to explain")
    args = parser.parse_args()

    model, meta = load_model_and_meta()
    if model is None:
        return

    feature_cols = meta["feature_cols"]
    classes = meta["classes"]
    model_name = meta["model_name"]

    print(f"Model: {model_name}")
    print(f"Classes: {classes}")
    print(f"Features: {len(feature_cols)}")

    X, y = load_data(args.split, feature_cols)
    if X is None:
        return

    output_dir = os.path.join(RESULTS_DIR, "shap")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nComputing SHAP values...")
    if model_name == "LogisticRegression":
        scaler = StandardScaler()
        train_X, _ = load_data("train", feature_cols)
        scaler.fit(train_X)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        X_display = X
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        X_display = X

    print("SHAP values computed.\n")

    print("Generating plots...")
    summary_plot(shap_values, X_display, classes, output_dir)
    per_class_plots(shap_values, X_display, classes, output_dir)
    dependence_plots(shap_values, X_display, classes, output_dir)
    force_plots(explainer, shap_values, X_display, y, classes, output_dir)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    print(f"\n{'='*60}")
    print("TOP FEATURES BY SHAP IMPORTANCE")
    print(f"{'='*60}")
    print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
