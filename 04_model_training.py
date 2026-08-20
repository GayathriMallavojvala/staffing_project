"""
Step 4: ML Model Training & Evaluation
------------------------------------------
Goal: Predict placement probability for a candidate-job match.

Key consideration: our data is IMBALANCED (25.7% placed vs 74.3% not placed).
This means plain accuracy is a misleading metric - a model that always
predicts "not placed" would score 74% accuracy while being useless.
We handle this with class_weight balancing and evaluate using precision,
recall, F1, and ROC-AUC instead of accuracy alone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)
from xgboost import XGBClassifier

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("../data/model_features.csv")

X = df.drop(columns=["placed", "days_to_place"])
y = df["placed"]

# ---------------------------------------------------------
# 1. Train/test split - stratified to preserve the 25.7% placement
#    ratio in BOTH train and test sets (important with imbalanced data)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train placement rate: {y_train.mean()*100:.1f}%, Test placement rate: {y_test.mean()*100:.1f}%")

# ---------------------------------------------------------
# 2. Train three models of increasing complexity
#    (baseline -> ensemble -> gradient boosting - shows a proper
#    progression, not just picking one algorithm blindly)
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # manual imbalance handling for XGBoost
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model": model, "f1": f1, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))
    print(f"ROC-AUC: {auc:.3f}")

# ---------------------------------------------------------
# 3. Pick the best model by F1 (better than accuracy for imbalanced data)
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\n{'='*60}\nBEST MODEL: {best_name} (F1={results[best_name]['f1']:.3f}, AUC={results[best_name]['auc']:.3f})\n{'='*60}")

# ---------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------
# ROC curves - all models on one plot for comparison
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("../data/model_roc_curves.png")
plt.close()

# Confusion matrix for best model
fig, ax = plt.subplots(figsize=(5.5, 5))
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("../data/model_confusion_matrix.png")
plt.close()

# Feature importance (for tree-based best model) or coefficients (for logistic)
fig, ax = plt.subplots(figsize=(8, 6))
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
else:
    importance = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=True)
importance.tail(12).plot(kind="barh", ax=ax, color="#2E5090")
ax.set_title(f"Top Features Driving Placement Prediction ({best_name})")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("../data/model_feature_importance.png")
plt.close()

# ---------------------------------------------------------
# 5. Save the best model for use in the dashboard
# ---------------------------------------------------------
joblib.dump(best_model, "../data/best_model.pkl")
joblib.dump(list(X.columns), "../data/model_feature_columns.pkl")
print(f"\nSaved best model ({best_name}) to data/best_model.pkl")
