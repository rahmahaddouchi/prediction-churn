# Customer Churn Prediction in Telecommunications

Building and comparing machine learning models to anticipate customer cancellations and identify the main retention levers.

**Author:** Rahma Haddouchi
**Supervisor:** Naceur Ben Della
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 variables
**Stack:** Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP

---

## Overview

In a highly competitive telecom market, keeping an existing customer costs far less than acquiring a new one. This project predicts whether a customer will cancel their subscription from their profile and behavior, and explains *why*, so that retention teams can act on it.

### Objectives

1. Understand the factors that drive churn
2. Run an exploratory data analysis (EDA)
3. Prepare the data for machine learning (cleaning, encoding, scaling)
4. Train several classification models of increasing complexity
5. Compare them using Accuracy, Precision, Recall, F1-score and ROC-AUC
6. Identify the most important variables (feature importance + SHAP)
7. Turn the findings into business recommendations

---

## Repository Structure

```
.
├── churn.ipynb                                   # Full analysis and modeling notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Dataset
├── Prédiction_du_Churn_Client_dans_le_secteur_des_Télécommunications_.pdf   # Project report (French)
└── README.md
```

---

## Dataset

7,043 customers described by 21 variables in four groups:

| Category | Variables |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contract & billing | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Identifier & target | `customerID`, `Churn` (Yes/No) |

The target is imbalanced: about **26.5 %** of customers churned (1,869 out of 7,043).

---

## Methodology

1. **Loading & inspection**: check shape, types and missing values.
2. **Cleaning**: `TotalCharges` is stored as text and converted to numeric. The resulting missing values all belong to customers with `tenure = 0` (new customers not yet billed) and are filled with 0. `customerID` is dropped.
3. **EDA**: churn distribution, churn by contract type, by tenure, monthly charges by churn status, and a correlation matrix.
4. **Preprocessing**
   - Binary variables encoded with `LabelEncoder`
   - Multi-class categorical variables one-hot encoded (`drop_first=True`)
   - Stratified 80/20 train/test split (`random_state=42`)
   - `StandardScaler` on `tenure`, `MonthlyCharges`, `TotalCharges`, fitted on the training set only to avoid data leakage
   - **SMOTE** applied to the training set only, never to the test set
5. **Modeling**: four classifiers
   - Logistic Regression (`max_iter=1000`)
   - Decision Tree (`max_depth=6`)
   - Random Forest (`n_estimators=300`, `max_depth=10`)
   - XGBoost (`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`)
6. **Evaluation**: metrics on the held-out test set, bar chart comparison and ROC curves.
7. **Interpretability**: Random Forest feature importance and SHAP analysis on a 500-customer test sample.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.731 | 0.495 | 0.698 | 0.579 | 0.822 |
| Decision Tree | 0.748 | 0.517 | **0.783** | **0.623** | 0.831 |
| **Random Forest** | **0.752** | **0.523** | 0.749 | 0.616 | **0.840** |
| XGBoost | **0.752** | **0.523** | 0.719 | 0.606 | 0.827 |

- **Random Forest** is selected as the final model (best ROC-AUC: 0.840).
- The **Decision Tree** has the best recall (0.783), valuable when missing an at-risk customer is costly.
- The gap between models is small (under 2 points of ROC-AUC), which suggests that data quality and preparation matter more than the choice of algorithm.

### Key churn drivers

- **Tenure**: the most important variable (importance ≈ 0.185); churn is concentrated in the first months (correlation with churn: −0.35).
- **Charges**: `TotalCharges` and `MonthlyCharges` rank 2nd and 3rd. Churners have a median monthly bill of about $79 versus about $64 for retained customers.
- **Contract type**: month-to-month customers account for most cancellations; `Contract_Two year` has a strong protective effect.
- **Payment method & internet service**: electronic check payment and fiber optic are associated with higher churn.
- **Security & support services**: `OnlineSecurity` and `TechSupport` are less important overall but are associated with better retention.
- **Demographics** (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) play a marginal role.

**Typical at-risk profile:** a recent customer on a month-to-month contract, with high monthly charges, no security or tech support services, paying by electronic check.

---

## Business Recommendations

1. Strengthen onboarding for new customers (first ~5 months)
2. Encourage 1- and 2-year contracts with incentives
3. Review pricing for high-bill customers (tiers, loyalty discounts)
4. Move electronic-check customers to automated payment methods
5. Audit the fiber optic offer (quality, support, pricing)
6. Promote security and tech support services
7. Integrate the Random Forest model into the CRM as an operational churn risk score

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap jupyter
```

### Run

Make sure the CSV file is in the same folder as the notebook, then:

```bash
jupyter notebook churn.ipynb
```

Run all cells in order. A fixed random seed (`RANDOM_STATE = 42`) keeps the results reproducible.

---

## Possible Improvements

- Hyperparameter tuning (grid/random search with cross-validation)
- Decision threshold optimization based on retention campaign costs
- Probability calibration
- Comparing SMOTE with class weights or other resampling strategies
- Model deployment (API or dashboard) for the retention team

---

## Acknowledgements

Dataset: IBM sample data, available on Kaggle as *Telco Customer Churn*.
