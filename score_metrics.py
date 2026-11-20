
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. Load all positive and negative CSV files
# ==========================================

pos_files = sorted(glob.glob("./pred_not_1_figures/*_positive_predictions.csv"))
neg_files = sorted(glob.glob("./pred_not_0_figures/*_negative_predictions.csv"))

print("Positive files:", len(pos_files))
print("Negative files:", len(neg_files))

assert len(pos_files) > 0, "No positive CSV files found"
assert len(neg_files) > 0, "No negative CSV files found"

positive_df = pd.concat( [pd.read_csv(f) for f in pos_files], ignore_index=True)
negative_df = pd.concat( [pd.read_csv(f) for f in neg_files],ignore_index=True)

# ==========================================
# 2. Combine all samples
# ==========================================

results = pd.concat([positive_df, negative_df], ignore_index=True)
# Validate labels and probabilities
assert (positive_df["y_true"] == 1).all()
assert (negative_df["y_true"] == 0).all()
assert results["p_ied"].between(0, 1).all()
assert results[["y_true", "p_ied"]].notna().all().all()

# Check duplicate sample identifiers
assert not results.duplicated(subset=["patient", "y_true", "sample_idx"]).any(), "Duplicate samples detected"

y_true = results["y_true"].to_numpy(dtype=int)
y_score = results["p_ied"].to_numpy(dtype=float)

# Original model decision threshold
threshold = 0.5
y_pred = (y_score >= threshold).astype(int)

print("Total samples:", len(results))
print("Positive IED samples:", (y_true == 1).sum())
print("Negative baseline samples:", (y_true == 0).sum())

# ==========================================
# 3. Confusion matrix
# ==========================================

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

accuracy = accuracy_score(y_true, y_pred)

# ==========================================
# 4. ROC-AUC and PR-AUC
# ==========================================

roc_auc = roc_auc_score(y_true, y_score)

average_precision = average_precision_score(y_true, y_score)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

pr_precision, pr_recall, pr_thresholds = (precision_recall_curve(y_true, y_score))
pr_auc = auc(pr_recall, pr_precision)

# ==========================================
# 5. Print results
# ==========================================

print("\n========== External Validation ==========")

print(f"ROC-AUC:          {roc_auc:.4f}")
print(f"PR-AUC:           {pr_auc:.4f}")
print(f"Average Precision:{average_precision:.4f}")
print(f"Sensitivity:      {sensitivity:.4f}")
print(f"Specificity:      {specificity:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1-score:         {f1:.4f}")
print(f"Accuracy:         {accuracy:.4f}")

print("\nConfusion Matrix")
print(f"TP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")

# ==========================================
# 6. Save combined results
# ==========================================

os.makedirs("./external_validation", exist_ok=True)

results.to_csv(
    "./external_validation/all_predictions.csv",
    index=False
)

metrics_df = pd.DataFrame([{
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Average_Precision": average_precision,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "Precision": precision,
    "F1": f1,
    "Accuracy": accuracy,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn
}])

metrics_df.to_csv(
    "./external_validation/overall_metrics.csv",
    index=False
)

# ==========================================
# 7. ROC curve
# ==========================================

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2,
         label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("External Validation - ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "./external_validation/roc_curve.png",
    dpi=300
)
plt.close()

# ==========================================
# 8. Precision-Recall curve
# ==========================================

plt.figure(figsize=(6, 5))
plt.plot(pr_recall, pr_precision, linewidth=2,
         label=f"AP = {average_precision:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("External Validation - PR Curve")
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig(
    "./external_validation/pr_curve.png",
    dpi=300
)
plt.close()
