# ==========================================
# Task 3: Customer Churn Prediction
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Churn_Modelling.csv')

print(df.head())

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# Detect target column (Churn/Exited)
target = None
for col in df.columns:
    if 'churn' in col.lower() or 'exit' in col.lower():
        target = col
        break

print("Target:", target)

X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Feature Importance
importances = model.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()