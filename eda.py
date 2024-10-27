# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and standardize column names
diabetes_data = pd.read_csv('Diabetes.csv')
diabetes_data.columns = diabetes_data.columns.str.lower()

# Drop unnecessary columns
diabetes_data = diabetes_data.drop(columns=['id', 'no_pation'])

# Standardize and clean 'class' column by stripping spaces and converting to uppercase
diabetes_data['class'] = diabetes_data['class'].str.strip().str.upper()

# Check unique values in 'class' to ensure proper encoding
print("Unique values in 'class' column before mapping:", diabetes_data['class'].value_counts(dropna=False))

# Map 'class' to numeric values: N=1, Y=2, P=3
diabetes_data['class'] = diabetes_data['class'].map({'N': 1, 'Y': 2, 'P': 3})

# Encode 'gender' column to numeric values: F=0, M=1
diabetes_data['gender'] = diabetes_data['gender'].map({'F': 0, 'M': 1})

# Handle missing value in 'gender' by filling with the mode
diabetes_data['gender'].fillna(diabetes_data['gender'].mode()[0], inplace=True)

# Confirm there are no remaining missing values
print("\nMissing values after encoding:", diabetes_data.isnull().sum())

# Class Distribution
sns.countplot(x='class', data=diabetes_data, palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1, 2], labels=["Non-Diabetic", "Diabetic", "Predicted-Diabetic"])
plt.show()

# Gender Distribution
sns.countplot(x='gender', data=diabetes_data, palette="coolwarm")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.show()

# Numeric Feature Distributions
numeric_columns = ['age', 'urea', 'cr', 'hba1c', 'chol', 'tg', 'hdl', 'ldl', 'vldl', 'bmi']
for column in numeric_columns:
    sns.histplot(diabetes_data[column], kde=True, color="skyblue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Box Plots for Outliers
for column in numeric_columns:
    sns.boxplot(y=diabetes_data[column], color="salmon")
    plt.title(f"Box Plot of {column}")
    plt.ylabel(column)
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = diabetes_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
