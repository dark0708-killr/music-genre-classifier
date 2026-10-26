"""Part 1 - Machine Learning baseline (Random Forest on GTZAN audio features).

Run:  python ml_baseline.py
Needs: data/features_30_sec.csv   (comes with the Kaggle GTZAN download)
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/features_30_sec.csv")
args = parser.parse_args()

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# 1. Load the data
df = pd.read_csv(args.csv)
X = df.drop(columns=["filename", "length", "label"], errors="ignore")
y = df["label"]
print(f"Loaded {len(df)} songs, {X.shape[1]} audio features, {y.nunique()} genres")

# 2. Train / test split (80 / 20, same genre balance in both)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Train two models and compare
tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

print(f"\nDecision Tree accuracy : {accuracy_score(y_te, tree.predict(X_te)):.3f}")
pred = forest.predict(X_te)
print(f"Random Forest accuracy : {accuracy_score(y_te, pred):.3f}\n")
print(classification_report(y_te, pred))

# 4. Save the Random Forest + a couple of charts for the report
joblib.dump({"model": forest, "features": list(X.columns)}, "models/genre_rf.joblib")

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, xticks_rotation=45, cmap="Blues")
ax.set_title("Random Forest - confusion matrix")
fig.tight_layout(); fig.savefig("reports/rf_confusion_matrix.png", dpi=150)

imp = pd.Series(forest.feature_importances_, index=X.columns).nlargest(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot.barh(ax=ax, color="#4c72b0")
ax.set_title("Top 15 most important audio features")
fig.tight_layout(); fig.savefig("reports/rf_feature_importance.png", dpi=150)

print("Saved: models/genre_rf.joblib, reports/rf_confusion_matrix.png, reports/rf_feature_importance.png")
