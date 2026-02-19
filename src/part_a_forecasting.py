# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:27:01 2025

@author: mathaios
"""

#Import necessary libraries
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Resolve project root and data path (so scripts run from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost.csv'
SAMPLE_DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost_sample.csv'
if not DATA_PATH.exists() and SAMPLE_DATA_PATH.exists():
    DATA_PATH = SAMPLE_DATA_PATH


# Step 1: Load the CSV file
MedCost_df = pd.read_csv(DATA_PATH)

# Step 2: Display basic info
print("Medical Cost Dataset Info:")
print(MedCost_df.info())
print(MedCost_df.head())

# Step 3: Clean Data - Drop missing values in 'charges' or 'bmi'
MedCost_df = MedCost_df.dropna(subset=["charges", "bmi"])

# Step 4: Filter data for the most common age group
most_common_age = MedCost_df["age"].value_counts().idxmax()
age_group_df = MedCost_df[MedCost_df["age"] == most_common_age]

# Step 5: Encode categorical variables
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    age_group_df[col] = le.fit_transform(age_group_df[col])

# Step 6: Sort by BMI and reset index for pseudo-time series
age_group_df = age_group_df.sort_values("bmi").reset_index(drop=True)

# Step 7: Visualization - Charges vs BMI
plt.figure(figsize=(14, 6))
plt.plot(age_group_df["bmi"], age_group_df["charges"], marker='o', label="Medical Charges")
plt.title(f"Medical Charges vs. BMI for Age {most_common_age}")
plt.xlabel("BMI")
plt.ylabel("Medical Charges")
plt.grid(True)
plt.legend()
plt.show()

# Step 8: Stationarity Tests
charges_series = age_group_df["charges"]

# ADF Test
adf_result = adfuller(charges_series)
print("\nAugmented Dickey-Fuller Test:")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

# KPSS Test
def kpss_test(series, **kw):
    statistic, p_value, lags, critical_values = kpss(series, **kw)
    print("\nKPSS Test:")
    print(f"KPSS Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    for key, value in critical_values.items():
        print(f"Critical Value ({key}): {value:.4f}")

kpss_test(charges_series, regression='c', nlags="auto")

# Step 9: ACF and PACF
n_obs = len(charges_series)
safe_lags = max(1, (n_obs // 2) - 1)
print(f"\nData points: {n_obs}, Using max lags: {safe_lags}")

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(charges_series, ax=ax[0], lags=safe_lags)
plot_pacf(charges_series, ax=ax[1], lags=safe_lags, method="ywm")
ax[0].set_title("Autocorrelation Function (ACF) - Medical Charges")
ax[1].set_title("Partial Autocorrelation Function (PACF) - Medical Charges")
plt.tight_layout()
plt.show()

# Step 10: Fit ARIMA(1,0,0)
model_arima = ARIMA(charges_series, order=(1, 0, 0))
results_arima = model_arima.fit()
print("\nARIMA(1,0,0) Summary:")
print(results_arima.summary())

# Plot Actual vs Fitted - ARIMA
plt.figure(figsize=(14, 6))
plt.plot(charges_series, label='Actual')
plt.plot(results_arima.fittedvalues, label='Fitted (ARIMA)', linestyle='--')
plt.title("ARIMA(1,0,0) - Actual vs Fitted Medical Charges")
plt.xlabel("Pseudo-time Index (sorted by BMI)")
plt.ylabel("Charges")
plt.legend()
plt.grid(True)
plt.show()

# Step 11: Fit ARX(1,0,0) with Multiple Exogenous Variables
exog_vars = age_group_df[["bmi", "sex", "smoker", "region"]]
model_arx = ARIMA(charges_series, exog=exog_vars, order=(1, 0, 0))
results_arx = model_arx.fit()
print("\nARX(1,0,0) Summary (Multiple Exogenous Variables):")
print(results_arx.summary())

# Plot Actual vs Fitted - ARX
plt.figure(figsize=(14, 6))
plt.plot(charges_series, label='Actual')
plt.plot(results_arx.fittedvalues, label='Fitted (ARX)', linestyle='--')
plt.title("ARX(1,0,0) - Actual vs Fitted Medical Charges (Exog: BMI, Sex, Smoker, Region)")
plt.xlabel("Pseudo-time Index (sorted by BMI)")
plt.ylabel("Charges")
plt.legend()
plt.grid(True)
plt.show()

# Step 12: Compare Models - AIC and RMSE
aic_arima = results_arima.aic
aic_arx = results_arx.aic
rmse_arima = np.sqrt(mean_squared_error(charges_series, results_arima.fittedvalues))
rmse_arx = np.sqrt(mean_squared_error(charges_series, results_arx.fittedvalues))

print("\nModel Comparison:")
print(f"ARIMA AIC: {aic_arima:.2f}")
print(f"ARX AIC:   {aic_arx:.2f}")
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"ARX RMSE:   {rmse_arx:.2f}")