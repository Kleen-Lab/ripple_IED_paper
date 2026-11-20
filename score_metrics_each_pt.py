
import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ======================================
# 1. Load all patient prediction files
# ======================================

pos_files = sorted(glob.glob(
    "./pred_not_1_figures/*_positive_predictions.csv"
))

neg_files = sorted(glob.glob(
    "./pred_not_0_figures/*_negative_predictions.csv"
))

assert pos_files, "No positive prediction files found"
assert neg_files, "No negative prediction files found"

pos_df = pd.concat(
    [pd.read_csv(f) for f in pos_files],
    ignore_index=True
)

neg_df = pd.concat(
    [pd.read_csv(f) for f in neg_files],
    ignore_index=True
)

assert (pos_df["y_true"] == 1).all()
assert (neg_df["y_true"] == 0).all()

df = pd.concat([pos_df, neg_df], ignore_index=True)

os.makedirs("./external_validation", exist_ok=True)

# ======================================
# 2. Calculate metrics for each patient
# ======================================

patient_results = []

for patient, patient_df in df.groupby("patient"):

    y_true = patient_df["y_true"].to_numpy(dtype=int)
    y_score = patient_df["p_ied"].to_numpy(dtype=float)
    y_pred = patient_df["y_pred"].to_numpy(dtype=int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else np.nan
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    # ROC-AUC requires both classes
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_ap = average_precision_score(y_true, y_score)
    else:
        roc_auc = np.nan
        pr_ap = np.nan

    patient_results.append({
        "patient": patient,
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "ROC_AUC": roc_auc,
        "PR_AP": pr_ap,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    })

# ======================================
# 3. Save per-patient statistics
# ======================================

patient_metrics = pd.DataFrame(patient_results)
patient_metrics = patient_metrics.sort_values("patient")

print("\n========== Per-Patient Metrics ==========")
print(patient_metrics.to_string(index=False, float_format="%.4f"))

patient_metrics.to_csv(
    "./external_validation/per_patient_metrics.csv",
    index=False
)

# ======================================
# 4. Patient-level mean and standard dev
# ======================================

metric_columns = [
    "ROC_AUC",
    "PR_AP",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1",
    "Accuracy"
]

summary = patient_metrics[metric_columns].agg(
    ["mean", "std", "median"]
)

print("\n========== Patient-Level Summary ==========")
print(summary.to_string(float_format="%.4f"))

summary.to_csv(
    "./external_validation/patient_metrics_summary.csv"
)
