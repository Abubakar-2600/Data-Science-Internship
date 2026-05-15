# ================================
# Task 1: Iris Dataset Analysis
# ================================

# 🔹 Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 2: Load Dataset
df = pd.read_csv('Iris.csv')

# 🔹 Step 3: Display Basic Info
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:\n", df.head())

# 🔹 Step 4: Drop unnecessary column (if exists)
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 🔹 Step 5: Data Summary
print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# 🔹 Step 6: Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# ================================
# 🔹 Step 7: Visualizations
# ================================

# 📊 1. Scatter Plot
plt.figure()
sns.scatterplot(
    x='SepalLengthCm',
    y='PetalLengthCm',
    hue='Species',
    data=df
)
plt.title("Sepal Length vs Petal Length")
plt.show()

# 📊 2. Histogram
plt.figure()
df['SepalLengthCm'].hist()
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# 📊 3. Box Plot
plt.figure()
sns.boxplot(
    x='Species',
    y='SepalLengthCm',
    data=df
)
plt.title("Sepal Length by Species")
plt.show()

# 📊 4. Pair Plot (Advanced Visualization)
sns.pairplot(df, hue='Species')
plt.show()

# ================================
# ✅ Task Completed
# ================================
print("\n✅ Task 1 Completed Successfully!")