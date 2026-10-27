import sys; sys.path.insert(0, "code")
import pandas as pd, numpy as np
from xgboost import XGBClassifier
from preprocess import load_raw, encode_and_scale, add_category_column

train, test = load_raw("data/NSL_KDD_Train.csv", "data/NSL_KDD_Test.csv")
X_tr, X_te, y_tr, y_te, *_ = encode_and_scale(train, test)
m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", n_jobs=-1, random_state=42).fit(X_tr, y_tr)
p = m.predict_proba(X_te)[:, 1]
pred = (p >= 0.5).astype(int)

df = test.copy(); df["y"] = y_te.values; df["pred"] = pred; df["p"] = p
df = add_category_column(df)
train_counts = train["label"].value_counts()
df["train_n"] = df["label"].map(train_counts).fillna(0).astype(int)
att = df[df.y == 1]

print("=== Overall attack recall:", round(att.pred.mean(), 4), "| attacks:", len(att), "| missed:", int((att.pred == 0).sum()))
seen = att[att.train_n > 0]; unseen = att[att.train_n == 0]
print(f"Seen-in-training attack types : n={len(seen)}, recall={seen.pred.mean():.3f}, missed={int((seen.pred==0).sum())}")
print(f"UNSEEN attack types           : n={len(unseen)}, recall={unseen.pred.mean():.3f}, missed={int((unseen.pred==0).sum())}")

g = att.groupby("label").agg(n_test=("y","size"), recall=("pred","mean"), missed=("pred", lambda s:int((s==0).sum())), train_n=("train_n","first")).sort_values("missed", ascending=False)
print("\nTop attack types by number missed:"); print(g.head(12).to_string())

print("\nBy category:")
print(att.groupby("attack_category").agg(n=("y","size"), recall=("pred","mean"), missed=("pred", lambda s:int((s==0).sum()))).round(3).to_string())

print("\nShare of all misses from unseen types:", round((unseen.pred==0).sum()/(att.pred==0).sum(), 3))

# Two-lane triage: auto-dismiss only if p < lo; review lane = lo<=p ; (escalate high-confidence = p>=0.5 anyway)
print("\n=== Lower-confidence second lane: auto-dismiss only if p < lo ===")
N = len(df)
rows = []
for lo in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
    dismissed = df[df.p < lo]
    missed = int(dismissed.y.sum())
    rows.append((lo, N - len(dismissed), round(100*len(dismissed)/N,1), missed, round(100*missed/len(att),1),
                 int(dismissed[dismissed.label.isin(unseen.label.unique())].y.sum())))
print(pd.DataFrame(rows, columns=["dismiss_if_p<","to_analyst","queue_reduction_%","attacks_dismissed","miss_rate_%","unseen_type_misses"]).to_string(index=False))
