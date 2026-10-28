"""
Leave-one-attack-family-out (LOFO) evaluation for SOC alert triage.

Question: how well does each approach handle attack families it has never seen?

For each family F in {dos, probe, r2l, u2r}:
  * F is removed from training entirely and treated as the "novel" case.
  * The remaining records (normal + the other three families) are split 70/30
    (stratified). 70% trains the models, 30% is held out for evaluation.
  * Evaluation set = the 30% hold-out of known traffic + ALL records of F.

Detectors compared:
  1. Classifier  : XGBoost trained on normal + known families (binary).
  2. Anomaly lane: Isolation Forest trained on NORMAL traffic only. Its alert
                   threshold is calibrated on held-out normal training records so
                   that it flags a target share of normal traffic (1%, 5%, 10%).
                   Attacks are never used to set the threshold.
  3. Two-lane    : classifier says attack  -> ESCALATE (high confidence)
                   else anomaly lane flags -> REVIEW   (unfamiliar pattern)
                   else                    -> DISMISS
                   Novel attacks are "caught" if they are escalated or reviewed.

Reported per family: novel-attack recall (with 95% Wilson interval), recall on
known attacks, and the analyst-workload cost (share of normal alerts that still
reach an analyst, and the queue reduction on the evaluation mix).

Data: official NSL-KDD train and test files are pooled so that every family has
enough records. Because this uses a random 70/30 split for known traffic, recall on
KNOWN attacks is optimistic; the novel-family recall is the number of interest.

Run from the project root:  python code/lofo_experiment.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import load_raw, add_category_column, COLUMN_NAMES, CATEGORICAL_COLS

SEED = 42
FAMILIES = ["dos", "probe", "r2l", "u2r"]
FPR_TARGETS = [0.01, 0.05, 0.10]
MAIN_FPR = 0.05          # operating point used in the headline chart
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_pooled():
    train, test = load_raw("data/NSL_KDD_Train.csv", "data/NSL_KDD_Test.csv")
    df = pd.concat([train, test], ignore_index=True)
    df = add_category_column(df)
    assert (df["attack_category"] != "unknown").all(), "unmapped attack label"
    feature_cols = [c for c in COLUMN_NAMES if c != "label"]
    X = pd.get_dummies(df[feature_cols], columns=CATEGORICAL_COLS, dtype=float)
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    num_idx = [X.columns.get_loc(c) for c in numeric_cols]
    return df, X.values.astype(np.float32), num_idx


def run_fold(held, df, X, num_idx):
    cats = df["attack_category"].values
    is_novel = cats == held
    novel_idx = np.where(is_novel)[0]
    known_idx = np.where(~is_novel)[0]

    tr_idx, ev_known_idx = train_test_split(
        known_idx, test_size=0.3, stratify=cats[known_idx], random_state=SEED
    )
    ev_idx = np.concatenate([ev_known_idx, novel_idx])

    # Scaling is fit on training rows only
    scaler = StandardScaler().fit(X[tr_idx][:, num_idx])
    Xs = X.copy()
    Xs[:, num_idx] = scaler.transform(X[:, num_idx])

    y = (cats != "normal").astype(int)

    # ---- Classifier ----
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        eval_metric="logloss", n_jobs=-1, random_state=SEED)
    clf.fit(Xs[tr_idx], y[tr_idx])
    p = clf.predict_proba(Xs[ev_idx])[:, 1]
    clf_flag = p >= 0.5

    # ---- Anomaly lane: Isolation Forest on normal-only training data ----
    tr_normal = tr_idx[cats[tr_idx] == "normal"]
    fit_normal, cal_normal = train_test_split(tr_normal, test_size=0.2, random_state=SEED)
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iso.fit(Xs[fit_normal])
    score = -iso.score_samples(Xs[ev_idx])          # higher = more anomalous
    cal_scores = -iso.score_samples(Xs[cal_normal])

    ev_cats = cats[ev_idx]
    ev_novel = ev_cats == held
    ev_known_attack = (ev_cats != "normal") & ~ev_novel
    ev_normal = ev_cats == "normal"
    n_eval = len(ev_idx)

    def row(detector, escalated, review=None, fpr_target=np.nan):
        caught = escalated if review is None else (escalated | review)
        to_analyst = caught
        k = int(caught[ev_novel].sum())
        n = int(ev_novel.sum())
        lo, hi = wilson(k, n)
        return {
            "held_out_family": held,
            "detector": detector,
            "anomaly_fpr_target": fpr_target,
            "n_novel": n,
            "novel_caught": k,
            "novel_recall": k / n,
            "novel_recall_ci_lo": lo,
            "novel_recall_ci_hi": hi,
            "known_attack_recall": float(caught[ev_known_attack].mean()),
            "normal_to_analyst_pct": 100 * float(caught[ev_normal].mean()),
            "queue_reduction_pct": 100 * (1 - to_analyst.sum() / n_eval),
        }

    rows = [row("Classifier only", clf_flag)]
    for t in FPR_TARGETS:
        thr = np.quantile(cal_scores, 1 - t)
        flag = score > thr
        rows.append(row("Anomaly lane only", flag, fpr_target=t))
        rows.append(row("Two-lane (classifier + anomaly review)",
                        clf_flag, review=flag & ~clf_flag, fpr_target=t))

        # extra detail for the two-lane: how many novel attacks came via the review lane
    return rows


def main():
    df, X, num_idx = load_pooled()
    print("Pooled records:", len(df))
    print(df["attack_category"].value_counts().to_string(), "\n")

    all_rows = []
    for fam in FAMILIES:
        print(f"Holding out family: {fam} ...", flush=True)
        all_rows += run_fold(fam, df, X, num_idx)

    res = pd.DataFrame(all_rows)
    res.to_csv(f"{RESULTS_DIR}/lofo_results.csv", index=False)

    show = res.copy()
    for c in ["novel_recall", "novel_recall_ci_lo", "novel_recall_ci_hi", "known_attack_recall"]:
        show[c] = show[c].round(3)
    for c in ["normal_to_analyst_pct", "queue_reduction_pct"]:
        show[c] = show[c].round(1)
    pd.set_option("display.width", 250)
    print("\n=== LOFO results (all operating points) ===")
    print(show.drop(columns=["novel_caught"]).to_string(index=False))

    # ---- Headline chart at MAIN_FPR ----
    sel = res[(res.detector == "Classifier only") |
              (res.anomaly_fpr_target.round(3) == MAIN_FPR)]
    order = ["Classifier only", "Anomaly lane only", "Two-lane (classifier + anomaly review)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.26
    x = np.arange(len(FAMILIES))
    for i, det in enumerate(order):
        d = sel[sel.detector == det].set_index("held_out_family").loc[FAMILIES]
        err = [d.novel_recall - d.novel_recall_ci_lo, d.novel_recall_ci_hi - d.novel_recall]
        axes[0].bar(x + (i - 1) * w, 100 * d.novel_recall, w, yerr=[100 * e for e in err],
                    capsize=3, label=det, color=colors[i])
        axes[1].bar(x + (i - 1) * w, d.normal_to_analyst_pct, w, label=det, color=colors[i])
    labels = [f"{f.upper()}\n(n={int(sel[sel.held_out_family==f].n_novel.iloc[0])})" for f in FAMILIES]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    axes[0].set_ylabel("Novel attacks caught (%)")
    axes[0].set_title("Recall on the held-out (never-seen) family")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("Normal alerts reaching an analyst (%)")
    axes[1].set_title(f"Analyst workload cost (anomaly lane at {int(MAIN_FPR*100)}% FPR target)")
    fig.suptitle("Leave-one-family-out evaluation (error bars: 95% Wilson interval)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lofo_novel_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {RESULTS_DIR}/lofo_results.csv and {RESULTS_DIR}/lofo_novel_recall.png")


if __name__ == "__main__":
    main()
