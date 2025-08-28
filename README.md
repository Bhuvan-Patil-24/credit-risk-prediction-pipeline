# Credit Risk Prediction for Loan Approvals

End-to-end ML project for classifying loan applicants into **low-risk (0)** vs **high-risk (1)**.
This scaffold includes:
- Data preprocessing & feature engineering
- EDA notebook
- Multiple models (Logistic Regression, Random Forest, XGBoost & Gradient Boost)
- Evaluation (ROC-AUC, confusion matrix)
- Saved pipeline
- Streamlit app for interactive scoring

## Folder Structure
```
credit_risk_project/
├─ app/
│  └─ streamlit_app.py
├─ data/
│  └─ credit_risk_dataset.csv
|  └─ test_credit_risk_dataset.csv
├─ models/
│  └─ gradient_boosting_pipeline.joblib
│  └─ xgboost_pipeline.joblib
├─ notebooks/
│  └─ EDA.ipynb
├─ reports/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ features.py
│  ├─ pipeline.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
└─ requirements.txt
```

## Quickstart

1. **Create a virtual env** (recommended)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Train models** (uses `data/credit_risk_dataset.csv` by default)
```bash
python -m src.train --data_path data/credit_risk_dataset.csv
```

3. **Run the app**
```bash
streamlit run app/streamlit_app.py
```

4. **Use your own data**
- Ensure your CSV has these columns (order not required):

| Column | Description |
|---|---|
| person_age | Applicant age (years) |
| person_income | Annual income |
| person_home_ownership | {RENT, OWN, MORTGAGE, OTHER} |
| person_emp_length | Employment length (years) |
| loan_intent | {PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION} |
| loan_grade | {A, B, C, D, E, F} |
| loan_amnt | Loan amount |
| loan_int_rate | Annual interest rate (%) |
| loan_percent_income | Loan-to-income ratio (loan_amnt / income) |
| cb_person_default_on_file | {Y, N} |
| cb_person_cred_hist_length | Credit history length (years) |
| loan_status | Target: 1 = default/high risk, 0 = repaid/low risk |

## Notes
- The pipeline handles missing values, encodes categoricals, scales numerics, and adds engineered features:
  - `dti` (debt-to-income): `loan_amnt / person_income`
  - `grade_risk` numeric score from `loan_grade` (A=1 ... F=6)
- If `xgboost` is not installed, training continues with the other models.
