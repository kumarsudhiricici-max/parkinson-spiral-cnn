# Parkinson Spiral Research Prototype

This project is designed as a stronger research/PhD demonstration than a simple image classifier.

## Research design

The pipeline includes:

- deterministic preprocessing
- handcrafted spiral geometry features
- SVM, Random Forest and XGBoost
- subject-level hold-out evaluation
- sensitivity and specificity
- ROC-AUC and PR-AUC
- Brier score
- bootstrap 95% confidence interval for held-out ROC-AUC
- held-out prediction export
- indeterminate zone around the decision boundary
- Streamlit training and inference UI

## Dataset

Preferred:

```text
dataset.zip
├── control/
│   ├── subject01_spiral.png
│   ├── subject02_spiral.png
│   └── ...
└── parkinson/
    ├── subject101_spiral.png
    ├── subject102_spiral.png
    └── ...
```

If a participant has multiple drawings, use a consistent subject identifier in filenames. The trainer recognizes identifiers such as `P001_trial1` and numeric participant/trial filenames such as `0068-3.jpg`; adapt `infer_subject()` in `src/research_train.py` for other naming conventions. Correct grouping is required to keep one participant's drawings out of both training and test sets.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research warning

A PhD-level presentation should not claim clinical diagnostic performance from a small public or convenience dataset. The project should report subject-level separation, external validation where possible, confidence intervals, class balance, missing/skipped images, and limitations.

The score displayed by the application is an ML model score. It is not a medical probability and is not a diagnosis.


## CNN model addition

This project now includes a 2D Convolutional Neural Network (CNN) that learns
image patterns directly from spiral drawings, in addition to the existing
SVM, Random Forest and XGBoost models. Images are converted to 128 x 128
grayscale and normalized to [0, 1]. The CNN uses convolution, batch
normalization, max pooling, dropout and a sigmoid output for binary
classification. Validation and test partitions are separated by subject.
The saved model is `models/cnn.keras` and can be selected in the Streamlit
Analyze Spiral tab after training.

The CNN's class decision cutoff is selected using a participant-separated
validation set to maximize balanced accuracy; held-out test data is not used
to choose it. The saved cutoff is applied during image analysis and is
reported with the held-out test metrics. This can prevent a model whose scores
all fall below 0.5 from predicting only one class, but it does not guarantee
that the CNN has useful predictive performance.

Install TensorFlow from the updated requirements. If it is unavailable for
your Python version, use a TensorFlow-supported Python release (Python 3.11
is a practical option). This is a research prototype, not a clinical
diagnostic system.
