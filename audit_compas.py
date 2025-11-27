"""
audit_compas.py
Loads COMPAS from AIF360, trains a logistic regression,
computes group fairness metrics (FPR, TPR, differences),
and plots disparities. Saves plots and a 300-word report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import utils as aif_utils

# local helper functions
from utils import plot_group_bars, save_report_300w

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_prep():
    # Load the COMPAS dataset (AIF360). Label: two_year_recid (0=no, 1=yes)
    dataset = CompasDataset()
    # Inspect
    print("Dataset shape:", dataset.features.shape)
    print("Protected attribute names:", dataset.protected_attribute_names)
    return dataset

def dataset_to_xy(dataset):
    # Convert AIF360 BinaryLabelDataset to X,y (numpy)
    X = dataset.features
    y = dataset.labels.ravel()  # shape (n,)
    feature_names = dataset.feature_names
    return X, y, feature_names

def train_model(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y_train)
    return clf, scaler

def evaluate_by_group(dataset_orig, clf, scaler):
    # Create a copy and get predictions as AIF360 ClassifiedDataset
    X, y, fnames = dataset_to_xy(dataset_orig)
    Xs = scaler.transform(X)
    y_pred = clf.predict_proba(Xs)[:,1] >= 0.5
    # Build classified dataset
    from aif360.datasets import BinaryLabelDataset
    classified = dataset_orig.copy()
    classified.labels = y_pred.reshape(-1,1).astype(int)

    # define privileged/unprivileged groups (by race)
    privileged_groups = [{'race': 1.0}]   # AIF360 maps: 1.0 -> Caucasian (see metadata)
    unprivileged_groups = [{'race': 0.0}] # 0.0 -> Not Caucasian

    metric = ClassificationMetric(dataset_orig, classified,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

    # compute FPR/TPR for both groups
    fpr_priv = metric.false_positive_rate(privileged=True)
    fpr_unpriv = metric.false_positive_rate(privileged=False)
    tpr_priv = metric.true_positive_rate(privileged=True)
    tpr_unpriv = metric.true_positive_rate(privileged=False)

    results = {
        'fpr_priv': fpr_priv, 'fpr_unpriv': fpr_unpriv,
        'tpr_priv': tpr_priv, 'tpr_unpriv': tpr_unpriv,
        'avg_odds_diff': metric.average_odds_difference(),
        'stat_par_diff': metric.mean_difference()
    }
    return results, metric

def main():
    dataset = load_and_prep()

    # Simple train/test split using AIF360 dataset indices
    X, y, feature_names = dataset_to_xy(dataset)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

    # create AIF360 datasets for train/test (so ClassificationMetric can inspect protected attrs)
    from aif360.datasets import BinaryLabelDataset
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    # NOTE: to keep things simple we use dataset.copy() trick for evaluation on full dataset
    clf, scaler = train_model(X_train, y_train)
    results, metric = evaluate_by_group(dataset, clf, scaler)

    print("=== Fairness metrics ===")
    print(f"FPR (privileged - Caucasian)   : {results['fpr_priv']:.4f}")
    print(f"FPR (unprivileged - non-Cauc): {results['fpr_unpriv']:.4f}")
    print(f"TPR (privileged)             : {results['tpr_priv']:.4f}")
    print(f"TPR (unprivileged)           : {results['tpr_unpriv']:.4f}")
    print(f"Average odds difference      : {results['avg_odds_diff']:.4f}")
    print(f"Statistical parity difference: {results['stat_par_diff']:.4f}")

    # Create bar plots
    fpr_vals = {'Caucasian (privileged)': results['fpr_priv'],
                'Non-Caucasian (unprivileged)': results['fpr_unpriv']}
    tpr_vals = {'Caucasian (privileged)': results['tpr_priv'],
                'Non-Caucasian (unprivileged)': results['tpr_unpriv']}

    plot_group_bars(fpr_vals, os.path.join(OUT_DIR, "fpr_by_race.png"),
                    title="False Positive Rate by Race")
    plot_group_bars(tpr_vals, os.path.join(OUT_DIR, "tpr_by_race.png"),
                    title="True Positive Rate by Race")

    # Save a 300-word report
    report_path = os.path.join(OUT_DIR, "report.txt")
    save_report_300w(report_path, results)
    print(f"\nReport saved to: {report_path}")
    print(f"Plots saved to: {OUT_DIR}/fpr_by_race.png and tpr_by_race.png")

if __name__ == "__main__":
    main()
