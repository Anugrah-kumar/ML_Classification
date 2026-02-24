# Loan Default Prediction

## a. Problem Statement
The goal of this project is to build and evaluate multiple machine learning classification models to predict whether a loan applicant will default on their loan (`loan_status` = 1) or not (`loan_status` = 0). By analyzing demographic and financial data, we aim to identify high-risk applicants.

## b. Dataset Description
The dataset `loan_data.csv` contains 45,000 records of loan applicants.
- **Features**: `person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`, `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`.
- **Target**: `loan_status` (Binary: 0 = Non-Default, 1 = Default).

## c. Models Used and Comparison

We implemented 6 classification models. The evaluation metrics on the test set are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8946 | 0.9527 | 0.7762 | 0.7418 | 0.7586 | 0.6915 |
| **Decision Tree** | 0.9017 | 0.8635 | 0.7719 | 0.7945 | 0.7830 | 0.7196 |
| **kNN** | 0.8938 | 0.9258 | 0.7961 | 0.7050 | 0.7478 | 0.6828 |
| **Naive Bayes** | 0.7364 | 0.9362 | 0.4586 | 0.9980 | 0.6284 | 0.5493 |
| **Random Forest** | 0.9291 | 0.9737 | 0.8938 | 0.7746 | 0.8300 | 0.7887 |
| **XGBoost** | 0.9220 | 0.9709 | 0.8712 | 0.7637 | 0.8139 | 0.7675 |

### Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs well with high AUC, serving as a strong baseline. Balanced precision and recall. |
| **Decision Tree** | Good accuracy but lower AUC compared to ensembles, suggesting some overfitting or lack of probability calibration. |
| **kNN** | Comparable to Logistic Regression. Effective but computationally expensive at inference time. |
| **Naive Bayes** | Very high Recall (99.8%) but poor Precision. It predicts "Default" too often, likely due to distribution assumptions. |
| **Random Forest** | **Best Performer**. Achieved the highest Accuracy (92.9%) and AUC (0.97). Robust handling of features. |
| **XGBoost** | Excellent performance, nearly matching Random Forest. Very efficient and accurate. |