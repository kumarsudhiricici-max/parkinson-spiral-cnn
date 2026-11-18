# Research figures

After training, the application generates:

1. **Dataset class distribution** — checks class balance.
2. **Model performance comparison** — balanced accuracy, sensitivity, specificity, F1 and ROC-AUC.
3. **Confusion matrix** — TP/TN/FP/FN for the selected primary model.
4. **ROC curve** — discrimination across thresholds.
5. **Precision–recall curve** — useful when class prevalence is unequal.
6. **Held-out score distribution** — shows overlap between classes and the decision boundary.
7. **Feature importance** — permutation importance measured on the held-out set.
8. **Feature correlation heatmap** — lower-triangle Pearson correlation matrix between engineered features, averaging repeated drawings per participant first. Positive values indicate features that increase together; negative values indicate inverse relationships.

These figures are saved under `models/figures/` at 300 DPI for use in reports/presentations.

Do not present these figures as evidence of clinical validity without an appropriately designed validation study and independent test cohort.
