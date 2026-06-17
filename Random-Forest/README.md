# Random Forest — Customer Churn Prediction

A complete implementation of the Random Forest algorithm built from the ground up using only NumPy and Pandas — no scikit-learn. Includes a from-scratch Decision Tree (with entropy-based splitting for both categorical and numeric features), bootstrap aggregation, balanced sampling for imbalanced classes, soft voting and a full evaluation suite.

Applied to the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict which customers are likely to cancel their subscription.

## Why build this from scratch?

Libraries like scikit-learn hide the mechanics behind a `.fit()` call. This project implements every core piece manually — entropy, information gain, recursive tree splitting, bootstrap sampling, and majority voting — to demonstrate a real understanding of how Random Forests work internally, not just how to call them.

## Features

* **Decision tree built from first principles** — entropy and information gain calculated manually at every split
* **Handles both categorical and numeric features** — numeric columns (like `tenure`, `MonthlyCharges`) are split using automatically discovered thresholds, no manual binning required
* **Class imbalance handling** — balanced bootstrap sampling ensures each tree sees an equal number of churn/non-churn examples, instead of being overwhelmed by the majority class
* **Soft voting via `predict_proba`** — returns probability scores per class instead of only hard labels, enabling threshold tuning
* **Tunable decision threshold** — adjust the cutoff to trade precision for recall depending on business needs
* **Feature importance** — ranks features by their average information gain across all trees in the forest
* **Full evaluation suite** — confusion matrix, accuracy, precision, recall, and F1 score, since accuracy alone is misleading on imbalanced data

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records from a telecom company, with a binary `Churn` label (Yes/No) and 19 features covering contract type, services subscribed, billing, and tenure.

Class distribution is imbalanced: roughly 73% "No" churn vs 27% "Yes" churn, which is why balanced bootstrap sampling and proper evaluation metrics (rather than raw accuracy) matter here.

## Exploratory Data Analysis (EDA) & Feature Selection
Before building and training the model, a thorough exploratory data analysis was conducted to refine the feature set:
* **Missing Values**: The dataset was verified to have no missing values in any of the columns, ensuring a clean start for structural processing.
* **Class Imbalance**: Bivariate analysis confirmed that the dataset is highly imbalanced toward non-churning customers, highlighting the necessity of internal balanced sampling.
* **Feature Filtering via Bivariate Analysis**: Features were evaluated based on the variance of their distribution relative to the target label (`Churn`). High-difference features and logical business columns were kept, while features that showed negligible difference in churn behavior across their distributions were dropped:
  * **Omitted Features**: `gender` and `PhoneService` were filtered out during EDA because their bivariate correlation with churn showed little to no variance between categories. `customerID` was dropped as an irrelevant identifier, and `TotalCharges` was omitted to avoid multicollinearity issues alongside `MonthlyCharges` and `tenure`.

This reduced the structural model input to 16 key contextual features.

### Features used

| Feature          | Type             |
| ---------------- | ---------------- |
| Contract         | Categorical      |
| InternetService  | Categorical      |
| PaymentMethod    | Categorical      |
| Partner          | Categorical      |
| SeniorCitizen    | Categorical      |
| Dependents       | Categorical      |
| tenure           | Numeric          |
| MultipleLines    | Categorical      |
| OnlineSecurity   | Categorical      |
| OnlineBackup     | Categorical      |
| DeviceProtection | Categorical      |
| TechSupport      | Categorical      |
| StreamingTV      | Categorical      |
| StreamingMovies  | Categorical      |
| PaperlessBilling | Categorical      |
| MonthlyCharges   | Numeric          |
| **Churn**        | **Target label** |

## How it works

### 1. Entropy & Information Gain

Each potential split is scored by how much it reduces entropy (uncertainty) in the target label. For categorical features, entropy is computed per category value. For numeric features, candidate thresholds are tested (midpoints between sorted unique values) and the one minimizing weighted entropy is chosen.

### 2. Decision Tree

Recursively splits the dataset on the feature with the highest information gain, until a stopping condition is hit (pure node, max depth reached, or too few samples to split further).

### 3. Bootstrap Aggregation (Bagging)

Each tree in the forest trains on a different bootstrap sample of the data. With `balance_classes=True`, this sample is drawn with equal counts from each class — this is the key fix for the class imbalance problem in this dataset.

### 4. Random Feature Subsets

At forest-construction time, each tree only considers a random subset of features (default: √total features), reducing correlation between trees and improving generalization.

### 5. Voting

For hard predictions, each tree casts one vote and the majority wins. For probability-based predictions (`predict_proba`), the fraction of trees voting for each class is returned, which can then be thresholded.

## Usage

```python
import pandas as pd
from random_forest import RandomForest, train_test_split

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = df[[
    'Contract', 'InternetService', 'PaymentMethod', 'Partner',
    'SeniorCitizen', 'Dependents', 'tenure', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'MonthlyCharges', 'Churn'
]]

train_df, test_df = train_test_split(df, test_size=0.2)

rf = RandomForest(
    n_trees=100,
    max_depth=10,
    min_samples_split=5,
    balance_classes=True
)

rf.fit(train_df, label_column="Churn")

rf.evaluate(test_df, pos_label="Yes")
```

### Tuning the decision threshold

```python
# Lower threshold catches more churners, at the cost of more false alarms
rf.evaluate(test_df, pos_label="Yes", threshold=0.35)

# Or sweep thresholds to find the F1-optimal cutoff
from random_forest import threshold_sweep
results_df, best_threshold = threshold_sweep(rf, test_df, pos_label="Yes")
```

## Results

Trained on 5,634 customers, evaluated on a held-out test set of 1,409 customers.

```text
Train size : 5634  |  Test size: 1409
Class dist : {'No': 5174, 'Yes': 1869}

                Pred Yes  Pred No
  Actual Yes         281       76
  Actual No          322      730

  Accuracy  : 0.7175
  Precision : 0.4660
  Recall    : 0.7871
  F1 Score  : 0.5854
```

### Feature Importance

| Feature          | Importance |
| ---------------- | ---------- |
| Contract         | 0.1435     |
| OnlineSecurity   | 0.0973     |
| TechSupport      | 0.0928     |
| InternetService  | 0.0807     |
| tenure           | 0.0758     |
| PaymentMethod    | 0.0689     |
| OnlineBackup     | 0.0660     |
| DeviceProtection | 0.0629     |
| StreamingMovies  | 0.0456     |
| StreamingTV      | 0.0452     |
| MonthlyCharges   | 0.0365     |
| PaperlessBilling | 0.0261     |
| Dependents       | 0.0194     |
| Partner          | 0.0170     |
| SeniorCitizen    | 0.0163     |
| MultipleLines    | 0.0014     |

**Contract type is by far the strongest predictor of churn** — consistent with real-world intuition that month-to-month customers churn far more readily than those on one- or two-year contracts. Security and support add-ons (`OnlineSecurity`, `TechSupport`) are the next strongest signals, suggesting customers without these services are more likely to leave.

### Interpreting the results

With the default 0.5 threshold, the model catches **78.7% of customers who actually churn** (recall), while **46.6% of customers flagged as "will churn" are correct** (precision). For a churn-prevention use case, this trade-off is usually intentional — missing a real churner is more costly than a wasted retention outreach to someone who wouldn't have left anyway.

## Project structure

```text
.
├── random_forest.py   # Full implementation: entropy, tree, forest, metrics
└── README.md
```

## Requirements

```text
numpy
pandas
```

## Possible extensions

* Out-of-bag (OOB) scoring for validation without a held-out test set (implemented but currently only supported when `balance_classes=False`)
* Cost-sensitive learning (weight false negatives more heavily during training, not just at prediction time)
* Parallel tree construction for faster training on larger datasets
* Pruning to reduce overfitting beyond just `max_depth`

## License

This project is for educational purposes.  
Dataset sourced from [IBM Telco Customer Churn]([https://www.kaggle.com/datasets/blastchar/telco-customer-churn]) — please refer to their terms of use for data licensing.

