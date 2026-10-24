"""
Train and evaluate several ML models for SOC alert triage (binary:
attack / true-positive vs normal / false-positive) on NSL-KDD.

Models compared:
  1. Logistic Regression   - interpretable linear baseline
  2. Random Forest          - strong non-linear baseline, feature importances
  3. XGBoost                - gradient-boosted trees, typically top performer
  4. MLP (small neural net) - non-linear baseline for comparison

Metrics reported (with SOC framing):
  - Precision  : of alerts flagged as attacks, how many really were attacks
                 (low precision -> analysts still drown in false alarms)
  - Recall     : of real attacks, how many the model actually caught
                 (low recall -> missed attacks; the costliest failure mode)
  - F1         : harmonic mean of precision/recall
  - ROC-AUC    : overall separability of the two classes
  - False Positive Rate (FPR) : fraction of normal alerts wrongly flagged
                 -> directly proportional to analyst workload
  - False Negative Rate (FNR) : fraction of real attacks missed
                 -> directly proportional to residual risk
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from xgboost import XGBClassifier

from preprocess import load_raw, encode_and_scale, add_category_column

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(name, model, X_test, y_test, results, roc_data):
    start = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_rate = fp / (fp + tn)
    fnr_rate = fn / (fn + tp)

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "false_positive_rate": round(fpr_rate, 4),
        "false_negative_rate": round(fnr_rate, 4),
        "true_negatives_correctly_dismissed": int(tn),
        "false_positives_wasted_analyst_time": int(fp),
        "false_negatives_missed_attacks": int(fn),
        "true_positives_correctly_flagged": int(tp),
        "inference_seconds_total": round(infer_time, 4),
        "inference_ms_per_alert": round(1000 * infer_time / len(y_test), 5),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[name] = (fpr, tpr, metrics["roc_auc"])

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal (0)", "Attack (1)"],
                yticklabels=["Normal (0)", "Attack (1)"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    return metrics


def main():
    train, test = load_raw("data/NSL_KDD_Train.csv", "data/NSL_KDD_Test.csv")
    train_cat = add_category_column(train)
    test_cat = add_category_column(test)

    print("Train label distribution:\n", train["label"].value_counts().head(10))
    print("\nAttack category distribution (train):\n", train_cat["attack_category"].value_counts())

    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = encode_and_scale(train, test)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class balance -> normal: {(y_train==0).sum()}, attack: {(y_train==1).sum()}")
    print(f"Test class balance  -> normal: {(y_test==0).sum()}, attack: {(y_test==1).sum()}")
    print("\nNote: NSL-KDD's test set intentionally includes attack types unseen in "
          "training, which stress-tests generalisation — analogous to a SOC "
          "facing novel attack variants not covered by existing detection rules.")

    results = []
    roc_data = {}

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    evaluate_model("Logistic Regression", log_reg, X_test, y_test, results, roc_data)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test, results, roc_data)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=42
    )
    xgb.fit(X_train, y_train)
    evaluate_model("XGBoost", xgb, X_test, y_test, results, roc_data)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    evaluate_model("MLP Neural Net", mlp, X_test, y_test, results, roc_data)

    # --- Combined ROC plot ---
    plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_comparison.png", dpi=150)
    plt.close()

    # --- Feature importance (Random Forest) ---
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#4C72B0")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # --- Results table ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    with open(f"{RESULTS_DIR}/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== FINAL MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    # --- Alert volume reduction simulation ---
    # If analysts previously had to review 100% of alerts, estimate how many
    # they'd need to review if only "predicted attack" alerts were escalated.
    best_model_name = results_df.sort_values("f1", ascending=False).iloc[0]["model"]
    print(f"\nBest model by F1: {best_model_name}")

    total_alerts = len(y_test)
    for r in results:
        escalated = r["true_positives_correctly_flagged"] + r["false_positives_wasted_analyst_time"]
        reduction_pct = 100 * (1 - escalated / total_alerts)
        r["alerts_escalated_to_analyst"] = escalated
        r["queue_reduction_pct"] = round(reduction_pct, 2)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    with open(f"{RESULTS_DIR}/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== ALERT QUEUE REDUCTION (vs reviewing all alerts) ===")
    print(results_df[["model", "alerts_escalated_to_analyst", "queue_reduction_pct",
                       "false_negative_rate"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
