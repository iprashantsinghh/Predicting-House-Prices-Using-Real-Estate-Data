import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import joblib

df = pd.read_excel("data/Case Study 1 Data (1).xlsx")
# Handle missing values

for col in df.columns:

    if df[col].dtype == "object":

        df[col].fillna(df[col].mode()[0], inplace=True)

    else:

        try:
            df[col] = pd.to_numeric(df[col])

            df[col].fillna(df[col].median(), inplace=True)

        except:
            pass
print(df.head())
print(df.info())

print(df.isnull().sum())
df.drop("Property ID", axis=1, inplace=True)
df["Date Sold"] = pd.to_datetime(df["Date Sold"])

df["Sale_Year"] = df["Date Sold"].dt.year

df["Sale_Month"] = df["Date Sold"].dt.month

df.drop("Date Sold", axis=1, inplace=True)
le = LabelEncoder()

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())
X = df.drop("Price", axis=1)

y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_train = X_train.fillna(0)

X_test = X_test.fillna(0)

y_train = y_train.fillna(0)

y_test = y_test.fillna(0)
lr = LinearRegression()


lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
print("LINEAR REGRESSION RESULTS")

print("MAE:", mean_absolute_error(y_test, lr_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

print("R2 Score:", r2_score(y_test, lr_pred))

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print("RANDOM FOREST RESULTS")

print("MAE:", mean_absolute_error(y_test, rf_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

print("R2 Score:", r2_score(y_test, rf_pred))
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    nthread=-1
)
# Final NaN check

print(df.isnull().sum())

print(df.dtypes)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
# Remove remaining NaN values

X_train = X_train.fillna(0)

X_test = X_test.fillna(0)
print("XGBOOST RESULTS")


print("MAE:", mean_absolute_error(y_test, xgb_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

print("R2 Score:", r2_score(y_test, xgb_pred))
joblib.dump(rf, "D:/case1/model/best_model.pkl")
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("RANDOM FOREST RESULTS")

print("MAE:", mean_absolute_error(y_test, rf_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

print("R2 Score:", r2_score(y_test, rf_pred))

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    nthread=-1
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("XGBOOST RESULTS")

print("MAE:", mean_absolute_error(y_test, xgb_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

print("R2 Score:", r2_score(y_test, xgb_pred))

# =========================
# VISUALIZATIONS
# =========================

# Price Distribution
plt.figure(figsize=(8,5))

sns.histplot(df["Price"], bins=30, kde=True)

plt.title("Price Distribution")

plt.xlabel("Price")

plt.ylabel("Count")

plt.show()


# Size vs Price
plt.figure(figsize=(8,5))

sns.scatterplot(x=df["Size"], y=df["Price"])

plt.title("Property Size vs Price")

plt.xlabel("Size")

plt.ylabel("Price")

plt.show()


# Bedrooms vs Price
plt.figure(figsize=(8,5))

sns.boxplot(x=df["Bedrooms"], y=df["Price"])

plt.title("Bedrooms vs Price")

plt.show()


# Correlation Heatmap
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()