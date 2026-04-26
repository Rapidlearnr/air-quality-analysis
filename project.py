
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
# EDA: DATA LOADING, CLEANING AND SUMMARY STATISTICS (NOT COUNTED AS OBJECTIVE)

df = pd.read_csv('AirQualityDataset.csv')

df.columns = df.columns.str.strip().str.lower()

df["state"] = df["state"].astype(str).str.strip().str.lower()
df["city"] = df["city"].astype(str).str.strip()

df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
df["pollutant_max"] = pd.to_numeric(df["pollutant_max"], errors="coerce")
df["pollutant_min"] = pd.to_numeric(df["pollutant_min"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

sns.set_style("darkgrid")

print("\n<<--- First 5 Rows --->>")
print(df.head())

print("\n<<--- Dataset Shape --->>")
print(df.shape)

print("\n<<--- Missing Values --->>")
print(df.isnull().sum())

print("\n<<--- Summary Statistics --->>")
print(df[["pollutant_min", "pollutant_max", "pollutant_avg", "latitude", "longitude"]].describe())

print("\n<<--- Pollutant Average Statistics by Pollutant Type --->>")
print(df.groupby("pollutant_id")["pollutant_avg"].describe())

# OBJECTIVE 2: VISUALIZATION OF POLLUTION PATTERNS

state_avg = df.groupby("state")["pollutant_avg"].mean().reset_index()
state_avg = state_avg.sort_values(by="pollutant_avg", ascending=False).head(10)

plt.figure(figsize=(8, 5))
plt.barh(state_avg["state"], state_avg["pollutant_avg"], color="coral")
plt.xlabel("Average Pollution")
plt.ylabel("State")
plt.title("Top 10 Polluted States")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(df["pollutant_avg"].dropna(), bins=20, kde=True, color="purple")
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.title("Pollution Distribution")
plt.tight_layout()
plt.show()

bihar_df = df[df["state"] == "bihar"]

city_avg = bihar_df.groupby("city")["pollutant_avg"].mean().reset_index()
city_avg = city_avg.sort_values(by="pollutant_avg", ascending=False).head(10)

plt.figure(figsize=(9, 4))
plt.bar(city_avg["city"], city_avg["pollutant_avg"], color="gold")
plt.xticks(rotation=45)
plt.xlabel("City")
plt.ylabel("Average Pollution")
plt.title("Top Polluted Cities in Bihar")
plt.tight_layout()
plt.show()

# OBJECTIVE 3: CORRELATION ANALYSIS OF NUMERICAL FEATURES

num_df = df.select_dtypes(include=np.number)
corr_data = num_df.corr()

print("\n<<--- Correlation Matrix --->>")
print(corr_data)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, linecolor="white")
plt.title("Correlation Matrix - All Numerical Features")
plt.tight_layout()
plt.show()


# OBJECTIVE 4: SIMPLE LINEAR REGRESSION TO PREDICT POLLUTANT AVERAGE

reg_df = df.dropna(subset=["pollutant_max", "pollutant_avg"])

X = reg_df[["pollutant_max"]]
y = reg_df["pollutant_avg"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(7, 4))
plt.scatter(reg_df["pollutant_max"], reg_df["pollutant_avg"], color="blue", alpha=0.7)
plt.plot(reg_df["pollutant_max"], y_pred, color="red", linewidth=2)
plt.xlabel("Pollutant Maximum")
plt.ylabel("Pollutant Average")
plt.title("Linear Regression: Pollutant Max vs Pollutant Avg")
plt.tight_layout()
plt.show()

print("\n<<--- Linear Regression Result --->>")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Model Equation: Pollutant Avg =", model.coef_[0], "* Pollutant Max +", model.intercept_)
