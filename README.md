# Stellar-Classification-with-Machine-Learning
The project utilises the Random Forest Classification to categorise astronomical objects as stars, galaxies, or quasars. This is a multi-class supervised ML classifier trained on SDSS DR18 photometric and spectroscopic data, hyperparameter tuned with GridSearchCV having an overall accuracy of 95.34% and an f1-score of 0.95.

---

## Table of Contents

- [Dataset](#dataset)
- [Reproducible Setup](#reproducible-setup)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Interpretation](#interpretation)
- [Project Structure](#project-structure)

---

## Dataset

This project uses the **[Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)** from Kaggle, sourced from the Sloan Digital Sky Survey (Data Release 17).

- **Target column:** `class`, with three values: `GALAXY`, `STAR`, `QSO`
- **Features:** photometric magnitudes (u, g, r, i, z), redshift, and other observational metadata

> Download the CSV from Kaggle and place it in `data/` before running the notebook. The dataset is not included in this repo due to size/licensing.

---

## Reproducible Setup

### 1. Clone and create an environment

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
```

### 3. Set random seeds

To ensure reproducibility across runs, a `random_state=[any_integer]` is used for:
- `train_test_split`
- `SMOTE`
- `RandomForestClassifier`

### 4. Run the notebook

```bash
jupyter notebook stellar_classification.ipynb
```

Run all cells top to bottom. The notebook expects `data/star_classification.csv` (or update the path in the loading cell to match your filename).

---

## Pipeline Overview

1. **Loading and Analysing data** — applying read_csv, dropping identifier/non-predictive columns (e.g. object IDs, run/camcol/field metadata if not used as features).
2. **Encoding the target** — `class` (`GALAXY`/`STAR`/`QSO`) is converted to integers via `LabelEncoder`. The mapping is alphabetical by default:
3. **Balancing the classes with SMOTE** — `SMOTE(random_state=42)` is applied to oversample minority classes so all three classes have equal representation.
4. **Train/test split** — features (`x`) and resampled target (`y`) are split via `train_test_split`.
5. **Training the model** — `RandomForestClassifier` is fit on the training set.
6. **Evaluation** — accuracy, classification report (precision/recall/F1 per class), ROC AUC per class, confusion matrix, and a feauture importance bar chart are generated on the test set.

---

## Results

**Overall accuracy:** 95.34%

| Class (encoded) | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 — GALAXY | 0.93 | 0.93 | 0.93 | 16,599 |
| 1 — QSO | 1.00 | 0.99 | 1.00 | 16,838 |
| 2 — STAR | 0.93 | 0.94 | 0.93 | 16,752 |

| Metric | Value |
|---|---|
| Macro avg F1 | 0.95 |
| Weighted avg F1 | 0.95 |
| Per-class F1 (avg) | 0.95 |
| Per-class ROC AUC (avg) | 0.9983 |

---

## Interpretation

- **QSO (quasars) are classified almost perfectly** (precision and recall both ≈0.99–1.00). This makes physical sense: quasars have a very distinctive spectral signature (strong, broad emission lines and characteristically high redshift) that separates them cleanly from stars and galaxies in feature space.
- **GALAXY and STAR are slightly harder to separate from each other** (precision/recall both ~0.94–0.95), which is consistent with the astrophysical reality that some galaxy and stellar photometric profiles can overlap, especially for compact or distant galaxies that appear point-like, similar to stars.
- **High AUC (0.98 average) across all classes** indicates the model's predicted probabilities rank true positives above false positives very reliably — performance is strong and consistent even before applying a specific classification threshold.
- **The macro and weighted averages are nearly identical** (0.96 vs 0.96), which tells us the classes are well-balanced in the evaluation set and that no single class is dominating the aggregate metrics. This is expected here since SMOTE balanced the classes before they were split.

- The reported 95.34% accuracy and 0.95 F1/0.9983 AUC are **optimistic** relative to true generalization performance.

---

## Project Structure

```
.
├── data/
│   └── star_classification.csv     # not included — download from Kaggle
├── stellar_classification.ipynb    # main notebook
├── requirements.txt
└── README.md
```
