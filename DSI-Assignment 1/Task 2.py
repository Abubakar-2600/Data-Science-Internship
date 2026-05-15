# ==========================================
# Task 2: Loan Default Prediction
# ==========================================

# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🔹 Step 2: Load Dataset
df = pd.read_csv('Loan_default.csv')

print("First 5 rows:\n", df.head())
print("\nColumns:\n", df.columns)

# 🔹 Step 3: Check Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# 🔹 Step 4: Handle Missing Values
df.fillna(method='ffill', inplace=True)

# 🔹 Step 5: Convert Categorical Columns → Numeric
le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

print("\nAfter Encoding:\n", df.head())

# 🔹 Step 6: Define Features (X) and Target (y)

# ⚠️ IMPORTANT: Adjust target column if needed
# Common names: 'Loan_Status', 'Default', 'Loan_Default'
target_column = None

for col in df.columns:
    if 'default' in col.lower() or 'status' in col.lower():
        target_column = col
        break

if target_column is None:
    raise Exception("⚠️ Target column not found. Tell me your column names.")

print("\nTarget Column:", target_column)

X = df.drop(target_column, axis=1)
y = df[target_column]

# 🔹 Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 8: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔹 Step 9: Predictions
y_pred = model.predict(X_test)

# 🔹 Step 10: Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# ✅ Task Completed
# ==========================================
print("\n🎉 Task 2 Completed Successfully!")