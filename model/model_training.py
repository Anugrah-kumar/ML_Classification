import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib



# Load the dataset
df = pd.read_csv('model/loan_data.csv')

# Define features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

combined_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
combined_df.to_csv("loan_test_data.csv", index=False, encoding="utf-8")

combined_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
combined_df.to_csv("loan_train_data.csv", index=False, encoding="utf-8")

df = pd.read_csv("loan_train_data.csv")

# update person_gender column as male=1 and female=0
df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})
# update person_education 
df['person_education'] = df['person_education'].map({'High School': 0, 'Associate': 1,'Bachelor': 2, 'Master': 3,'Doctorate': 4})

# update previous_loan_defaults_on_file column as Yes=1 and No=0
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

if 'person_home_ownership' in df.columns:

    df['person_home_ownership_MORTGAGE'] = 0
    df['person_home_ownership_OTHER'] = 0
    df['person_home_ownership_OWN'] = 0
    df['person_home_ownership_RENT'] = 0

    for index, row in df.iterrows():
        if row['person_home_ownership'] == 'MORTGAGE':
            df.at[index, 'person_home_ownership_MORTGAGE'] = 1
        elif row['person_home_ownership'] == 'OWN':
            df.at[index, 'person_home_ownership_OWN'] = 1
        elif row['person_home_ownership'] == 'RENT':
            df.at[index, 'person_home_ownership_RENT'] = 1
        elif row['person_home_ownership'] == 'OTHER':
            df.at[index, 'person_home_ownership_OTHER'] = 1

    df.drop(['person_home_ownership'], axis=1, inplace=True)

if 'loan_intent' in df.columns:

    df['loan_intent_DEBTCONSOLIDATION'] = 0
    df['loan_intent_EDUCATION'] = 0
    df['loan_intent_HOMEIMPROVEMENT'] = 0
    df['loan_intent_MEDICAL'] = 0
    df['loan_intent_PERSONAL'] = 0
    df['loan_intent_VENTURE'] = 0
    for index, row in df.iterrows():
            if row['loan_intent'] == 'DEBTCONSOLIDATION':
                df.at[index, 'loan_intent_DEBTCONSOLIDATION'] = 1
            elif row['loan_intent'] == 'EDUCATION':
                df.at[index, 'loan_intent_EDUCATION'] = 1
            elif row['loan_intent'] == 'HOMEIMPROVEMENT':
                df.at[index, 'loan_intent_HOMEIMPROVEMENT'] = 1
            elif row['loan_intent'] == 'MEDICAL':
                df.at[index, 'loan_intent_MEDICAL'] = 1
            elif row['loan_intent'] == 'PERSONAL':
                df.at[index, 'loan_intent_PERSONAL'] = 1
            elif row['loan_intent'] == 'VENTURE':
                df.at[index, 'loan_intent_VENTURE'] = 1

    df.drop(['loan_intent'], axis=1, inplace=True)



target_col = 'loan_status'
# Prepare X and y
X_train = df.drop(target_col, axis=1)
y_train = df[target_col]

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Save the model using joblib
filename = f'model/logistic_regression.joblib'
joblib.dump(log_reg, filename)
print(f"Saved {filename}")


# Initialize Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/decision_tree.joblib'
joblib.dump(dt_classifier, filename)
print(f"Saved {filename}")


# Initialize K-Nearest Neighbors model
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/knn.joblib'
joblib.dump(knn_classifier, filename)
print(f"Saved {filename}")


# Initialize Naive Bayes model
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/naive_bayes.joblib'
joblib.dump(nb_classifier, filename)
print(f"Saved {filename}")



# Initialize Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/random_forest.joblib'
joblib.dump(rf_classifier, filename)
print(f"Saved {filename}")


# Initialize Gradient Boosting model
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gb_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/xgboost.joblib'
joblib.dump(gb_classifier, filename)
print(f"Saved {filename}")



print("All models saved successfully.")