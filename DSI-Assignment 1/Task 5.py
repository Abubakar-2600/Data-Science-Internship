# ==========================================
# Task 5: Loan Acceptance Prediction
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('bank.csv')

print(df.head())

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# Detect target (y or loan accepted)
target = 'y' if 'y' in df.columns else df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Visualization
sns.countplot(x=target, data=df)
plt.title("Loan Acceptance Distribution")
plt.show()