# utils.py -- helpers for plotting and report
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import datetime

def plot_group_bars(values_dict, out_path, title=""):
    labels = list(values_dict.keys())
    vals = [values_dict[k] for k in labels]
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=vals)
    plt.ylim(0, max(0.5, max(vals)*1.2))
    plt.ylabel("Rate")
    plt.title(title)
    plt.xticks(rotation=10)
    for i,v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_report_300w(path, metrics):
    # 300-word report (approx)
    report = f"""
COMPAS fairness audit — generated {datetime.date.today().isoformat()}

Summary of key findings:
The classifier trained on the COMPAS features exhibits measurable disparities by race.
False Positive Rate (FPR) for the privileged group (Caucasian) = {metrics['fpr_priv']:.4f},
for the unprivileged group (non-Caucasian) = {metrics['fpr_unpriv']:.4f}.
True Positive Rate (TPR) for privileged = {metrics['tpr_priv']:.4f},
for unprivileged = {metrics['tpr_unpriv']:.4f}.
Average odds difference = {metrics['avg_odds_diff']:.4f}.
Statistical parity difference = {metrics['stat_par_diff']:.4f}.

Interpretation & remediation suggestions:
- The difference in FPR indicates that non-Caucasian individuals receive false positives (predicted recidivism when they did not recidivate) at a different rate than Caucasian individuals. Such disparity can lead to unfair higher-risk labeling for the unprivileged group.
- Consider pre-processing methods (reweighing, disparate impact remover) to reduce dataset bias, in-processing fairness-aware training (e.g., adversarial debiasing, prejudice-remover), or post-processing methods (equalized odds post-processing) to adjust decision thresholds by group.
- Evaluate trade-offs between overall predictive accuracy and fairness metrics. Prefer multiple metrics (FPR difference, TPR difference, average odds difference, calibration) rather than a single number.
- Monitor and document deployment decisions; include human oversight, explainability, and regular fairness checks.

Next steps:
(1) Apply a mitigation algorithm from AIF360 (reweighing or equalized odds) and re-evaluate metrics.
(2) Report metrics to stakeholders including confusion-matrix-level rates and use-case–specific tolerances.
(3) Consider collecting more representative data and designing decision processes that avoid over-reliance on risk scores.

(End of 300-word summary)
"""
    # truncate/clean to ~300 words
    wrapper = textwrap.dedent(report).strip()
    with open(path, "w") as f:
        f.write(wrapper)
