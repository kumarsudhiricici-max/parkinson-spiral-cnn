# Dataset format

Preferred ZIP:

```text
dataset.zip
├── control/
└── parkinson/
```

Use subject identifiers consistently. The trainer recognizes common names
such as `subject_P001_trial1.png` and numeric participant/trial names such as
`0068-3.jpg` (participant `0068`, trial `3`). Example:

```text
control/P001_spiral.png
control/P002_spiral.png
parkinson/P101_spiral.png
parkinson/P102_spiral.png
```

If you have multiple drawings per participant, ensure they share the same
participant identifier:

```text
control/P001_trial1.png
control/P001_trial2.png
parkinson/P101_trial1.png
```

The trainer groups `P001_trial1` and `P001_trial2` as the same subject. If your
dataset uses another naming convention, adapt `infer_subject()` in
`src/research_train.py`; otherwise, drawings from one person could appear in
both training and test sets and inflate reported performance.
