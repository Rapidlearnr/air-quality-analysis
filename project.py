import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

df = pd.read_excel("Census_A1_Clean.xlsx")
# EDA (including missing values check and handling)

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values before handling:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("\nMissing values after handling:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())
# OBJECTIVE 1: Outlier Detection

Q1 = df["Population_Total_Rural"].quantile(0.25)
Q3 = df["Population_Total_Rural"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Population_Total_Rural"] < lower_bound) | (df["Population_Total_Rural"] > upper_bound)]

print("\nOutliers in Population_Total_Rural:")
print(outliers)

print("\nNumber of outliers:")
print(outliers.shape[0])


# OBJECTIVE 2: Visualization Before Removing Outliers and Correlation

# Improving readability by defining target column
target_col = "Population_Total_Rural"

plt.figure(figsize=(8, 5))
sns.histplot(df[target_col], bins=30, kde=True)
plt.title("Distribution of Rural Population Before Removing Outliers")
plt.xlabel(target_col)
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df[target_col])
plt.title("Boxplot of Rural Population Before Removing Outliers")
plt.ylabel(target_col)
plt.show()

corr_cols = [
    "Households_Rural",
    "Population_Total_Rural",
    "Households_Urban",
    "Population_Total_Urban",
    "Pop_per_sq_km_Rural",
    "Pop_per_sq_km_Urban"
]

plt.figure(figsize=(10, 6))
sns.heatmap(df[corr_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
