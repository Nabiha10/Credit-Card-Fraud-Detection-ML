# Credit Card Fraud Detection using Machine Learning

This project implements a credit card fraud detection system using the Kaggle Credit Card Fraud Detection dataset. It was developed as a reproduction and critical evaluation of the methodology described in the paper "Credit Card Fraud Detection - Machine Learning Methods" (Varmedja, Karanovic, Sladojevic, Arsenovic & Anderla, 2019).

## Table of Contents

- [Project Overview](#project-overview)
- [Reference Paper](#reference-paper)
- [Dataset](#dataset)
- [Reference GitHub Repository](#reference-github-repository)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Required Libraries](#required-libraries)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Key Findings](#key-findings)
- [References](#references)

## Project Overview

Credit card fraud detection involves classifying transactions as either legitimate or fraudulent, made difficult by the fact that fraud accounts for a tiny fraction of all transactions (~0.17% in this dataset). This project trains and evaluates two machine learning models — Random Forest and XGBoost — on the Credit Card Fraud Detection dataset, and critically evaluates how well the reference paper's reported results could be reproduced, including whether the author's claims are supported by the evidence presented.

The project goes beyond reproduction: it examines the appropriateness of the original evaluation methodology, investigates feature engineering choices, and performs a detailed error analysis of false positives and false negatives with their real-world financial and cybersecurity implications.

## Reference Paper

Varmedja, D., Karanovic, M., Sladojevic, S., Arsenovic, M., & Anderla, A. (2019). *Credit Card Fraud Detection - Machine Learning Methods*. 2019 18th International Symposium INFOTEH-JAHORINA (INFOTEH), 1-5.

- Official DOI (IEEE Xplore): https://doi.org/10.1109/INFOTEH.2019.8717766
- Free full text (author-uploaded, ResearchGate): https://www.researchgate.net/publication/333229231_Credit_Card_Fraud_Detection_-_Machine_Learning_methods

## Dataset

**Credit Card Fraud Detection Dataset** — anonymized European cardholder transactions from September 2013, containing 284,807 transactions with 492 confirmed frauds (~0.172% of all transactions). Features V1-V28 are PCA-transformed for confidentiality; only `Time` and `Amount` remain in their original form.

Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> **Note:** The dataset (~150 MB) is not included in this repository due to size and licensing. Download it from Kaggle and place `creditcard.csv` inside the `data/` folder before running the notebook.

## Reference GitHub Repository

This project also references the implementation approach used in:
Aryan Chauhan's credit-card-fraud-detection repository: https://github.com/itsaryanchauhan/credit-card-fraud-detection

## Repository Structure

```
Credit-Card-Fraud-Detection-ML/
│
├── data/
│    └──creditcard.csv                 # creditcard.csv goes here (download from Kaggle)
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb     # Main analysis and modeling notebook
│
├── output/
│   ├── confusion_matrices.png       # Confusion matrix visualizations
│   ├── results_summary.json         # All metrics/stats, machine-readable
│   └── results_summary.txt          # Same results, human-readable
│
├── report/
│   └── Credit_Card_Fraud_Report.pdf             # Full written report (summary, critical evaluation,
│                                     # featusre engineering analysis, reproducibility,
│                                     # experimental results, conclusions)
│
├── requirements.txt

└── README.md
```

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/Nabiha10/Credit-Card-Fraud-Detection-ML.git
cd Credit-Card-Fraud-Detection-ML
```

**2. Set up a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it inside the `data/` folder.

**5. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open `notebooks/Credit_Card_Fraud_Detection.ipynb` and run all cells in order — the notebook is self-contained: data loading → EDA → feature engineering → model training → evaluation → error analysis → saving outputs.

## Required Libraries

| Library | Purpose |
|---|---|
| pandas | Data manipulation and analysis |
| numpy | Numerical computations |
| matplotlib | Data visualization |
| seaborn | Statistical data visualization |
| scikit-learn | Machine learning models and evaluation |
| xgboost | Gradient boosting model |
| jupyter | Notebook environment |

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Methodology

| Stage | Approach |
|---|---|
| Data Loading | Inspection of shape, types, missing values, duplicates, constant columns |
| EDA | Distributions, IQR-based outlier analysis, temporal (hour-of-day) patterns, Pearson/Spearman/Kendall correlation analysis, class imbalance discussion |
| Feature Engineering | `RobustScaler` applied to `Time` and `Amount` (robust to the extreme values inherent in fraud transactions); PCA components (V1-V28) used as-is |
| Train/Test Split | Stratified 80/20 split to preserve the ~0.17% fraud ratio in both sets |
| Model Training | Random Forest (`class_weight="balanced"`) and XGBoost (`scale_pos_weight`-adjusted), with 5-fold stratified cross-validation |
| Evaluation | Precision, Recall, F1, Matthews Correlation Coefficient (MCC), ROC-AUC, Confusion Matrix — Accuracy intentionally excluded due to class imbalance |
| Error Analysis | False Positive / False Negative case inspection and cybersecurity/financial implications |

## Models Used

**Random Forest** — an ensemble of decision trees whose predictions are combined by majority voting, trained with `class_weight="balanced"` to account for the severe class imbalance.

**XGBoost** — a gradient boosting ensemble trained with `scale_pos_weight` adjusted for the fraud/non-fraud ratio.

## Results

| Model | Precision | Recall | F1 Score | MCC | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 84.21% | 81.63% | 82.90% | 0.829 | 0.975 |
| XGBoost | 89.25% | 84.69% | 86.91% | 0.869 | 0.967 |

Accuracy is intentionally excluded from the results table, as it is not a meaningful metric given the ~0.17% fraud prevalence in the dataset — a model predicting "no fraud" every time would already score above 99%.

## Key Findings

- Both models achieved very low false positive rates (10-15 false alarms out of ~56,860 legitimate test transactions), meaning legitimate traffic was rarely misclassified as fraud.
- XGBoost outperformed Random Forest on Precision, Recall, F1, and MCC, while Random Forest achieved a marginally higher ROC-AUC, indicating slightly better overall ranking of transactions by fraud risk.
- Error analysis showed that most missed frauds (false negatives) had a PCA feature signature closely resembling legitimate transactions, rather than failure being driven primarily by transaction amount. A smaller subset of missed frauds involved unusually large amounts that still received very low predicted fraud probability — the more concerning failure mode operationally.
- Cross-validation confirmed that the single train/test split results were stable rather than a result of a lucky split.
- Full critical discussion of whether the reference paper's claims hold up under closer scrutiny is available in `report/Credit_Card_Fraud_Report.pdf`.



## References

1. Varmedja, D., Karanovic, M., Sladojevic, S., Arsenovic, M., & Anderla, A. (2019). Credit Card Fraud Detection - Machine Learning Methods. 2019 18th International Symposium INFOTEH-JAHORINA (INFOTEH), 1-5.
2. Credit Card Fraud Detection Dataset — Machine Learning Group, Universite Libre de Bruxelles (ULB). https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Chauhan, A. credit-card-fraud-detection (reference implementation). https://github.com/itsaryanchauhan/credit-card-fraud-detection

