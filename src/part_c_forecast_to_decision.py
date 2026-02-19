# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 14:47:59 2025

@author: mathaios
"""


from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pulp as pl

# Resolve project root and data path (so scripts run from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost.csv'
SAMPLE_DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost_sample.csv'
if not DATA_PATH.exists() and SAMPLE_DATA_PATH.exists():
    DATA_PATH = SAMPLE_DATA_PATH


# -----------------------------
# 0) Configuration (easy to tweak)
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Program costs (per participant)
COST_SMOKING = 500.0
COST_WEIGHT  = 400.0

# Expected savings rates (share of forecasted average charges)
# Choose rates you can justify in the report (these mirror your recent runs)
SAVINGS_RATE_SMOKER = 0.225   # 22.5% of forecasted smoker charges
SAVINGS_RATE_OBESE  = 0.294   # 29.4% of forecasted obese charges

# Total budget and simple capacity assumptions (fractions of eligible pool)
TOTAL_BUDGET = 120_000.0
CAP_FRAC_SMOKER = 0.50  # enroll up to 50% of eligible smokers
CAP_FRAC_OBESE  = 0.20  # enroll up to 20% of eligible obese

# -----------------------------
# 1) Load & prepare data
# -----------------------------
df = pd.read_csv(DATA_PATH).dropna(subset=["charges", "bmi", "smoker"])
df["smoker"] = df["smoker"].str.lower().str.strip()
df["obese"]  = (df["bmi"] >= 30).astype(int)

# Two decision segments: smokers (any BMI) and obese (any smoker status)
seg_smoker = df[df["smoker"] == "yes"].copy()
seg_obese  = df[df["obese"] == 1].copy()

# Eligibility counts
N_smokers = int(len(seg_smoker))
N_obese   = int(len(seg_obese))

# Capacity caps (integers)
cap_smoker = int(max(0, min(N_smokers, round(CAP_FRAC_SMOKER * N_smokers))))
cap_obese  = int(max(0, min(N_obese,   round(CAP_FRAC_OBESE  * N_obese))))

print(f"Eligible smokers: {N_smokers} | capacity cap: {cap_smoker}")
print(f"Eligible obese:   {N_obese}   | capacity cap: {cap_obese}")

# -----------------------------
# 2) Fit ARIMA/ARX per segment and forecast 1-step ahead
# -----------------------------
def fit_arima_arx_forecast(segment_df, value_col="charges", exog_col="bmi"):
    """
    Fit ARIMA(1,0,0) and ARX(1,0,0) (exogenous = BMI) on a BMI-sorted sequence.
    Returns dict with fitted models/values and a 1-step-ahead forecast.
    """
    seg = segment_df.sort_values(exog_col).reset_index(drop=True)
    y = seg[value_col].astype(float)
    X = seg[exog_col].astype(float)

    # ARIMA baseline (no exog)
    arima = ARIMA(y, order=(1, 0, 0))
    arima_res = arima.fit()

    # ARX (ARIMA with exogenous BMI)
    arx = ARIMA(y, exog=X, order=(1, 0, 0))
    arx_res = arx.fit()

    # 1-step-ahead forecasts (use positional indexing to avoid KeyError)
    fc_arima = arima_res.forecast(steps=1).iloc[0]
    last_bmi = float(X.iloc[-1])
    fc_arx   = arx_res.forecast(steps=1, exog=np.array([[last_bmi]])).iloc[0]

    return {
        "seg": seg,
        "y": y, "X": X,
        "arima_res": arima_res,
        "arx_res": arx_res,
        "fitted_arima": arima_res.fittedvalues,
        "fitted_arx": arx_res.fittedvalues,
        "fc_arima": float(fc_arima),
        "fc_arx": float(fc_arx),
        "last_bmi": last_bmi
    }

# Fit models and forecast for Smokers & Obese segments
res_smoker = fit_arima_arx_forecast(seg_smoker, value_col="charges", exog_col="bmi")
res_obese  = fit_arima_arx_forecast(seg_obese,  value_col="charges", exog_col="bmi")

print("\n--- Forecasts (1-step ahead) ---")
print(f"Smokers: ARIMA = {res_smoker['fc_arima']:.2f} | ARX = {res_smoker['fc_arx']:.2f}")
print(f"Obese:   ARIMA = {res_obese['fc_arima']:.2f}  | ARX = {res_obese['fc_arx']:.2f}")

# -----------------------------
# 3) Visualize fitted vs actual (per segment)
# -----------------------------
def plot_segment_fit(seg_res, title_prefix):
    y = seg_res["y"]
    fitted_arima = seg_res["fitted_arima"]
    fitted_arx   = seg_res["fitted_arx"]

    plt.figure(figsize=(10,5))
    plt.plot(y.values, label="Actual (charges)")
    plt.plot(fitted_arima.values, "--", label="Fitted (ARIMA)")
    plt.plot(fitted_arx.values, ":", label="Fitted (ARX exog=BMI)")
    plt.title(f"{title_prefix}: Actual vs Fitted (BMI-sorted pseudo-time)")
    plt.xlabel("Pseudo-time index (sorted by BMI)")
    plt.ylabel("Charges")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_segment_fit(res_smoker, "Smokers")
plot_segment_fit(res_obese,  "Obese")

# -----------------------------
# 4) Forecasted average charges → savings per participant (ARX)
# -----------------------------
fc_smoker_charges = res_smoker["fc_arx"]
fc_obese_charges  = res_obese["fc_arx"]

save_per_smoker = SAVINGS_RATE_SMOKER * fc_smoker_charges
save_per_obese  = SAVINGS_RATE_OBESE  * fc_obese_charges

print("\n--- Decision Inputs (from ARX forecasts) ---")
print(f"Forecasted avg charges (Smokers): {fc_smoker_charges:.2f}  → savings per participant: {save_per_smoker:.2f}")
print(f"Forecasted avg charges (Obese):   {fc_obese_charges:.2f}  → savings per participant: {save_per_obese:.2f}")

# -----------------------------
# 5) Optimization (LP): maximize net savings
# -----------------------------
model = pl.LpProblem("Forecast_to_Decision_Healthcare", pl.LpMaximize)

x_smoke = pl.LpVariable("enroll_smokers", lowBound=0, upBound=N_smokers, cat=pl.LpInteger)
x_obese = pl.LpVariable("enroll_obese",   lowBound=0, upBound=N_obese,   cat=pl.LpInteger)

# Objective
model += (save_per_smoker * x_smoke + save_per_obese * x_obese
          - COST_SMOKING * x_smoke - COST_WEIGHT * x_obese), "Net_Savings"

# Constraints
model += COST_SMOKING * x_smoke + COST_WEIGHT * x_obese <= TOTAL_BUDGET, "Budget"
model += x_smoke <= cap_smoker, "Capacity_Smoker"
model += x_obese <= cap_obese,  "Capacity_Obese"

# Solve
status = model.solve(pl.PULP_CBC_CMD(msg=False))

opt_smoke = int(x_smoke.varValue)
opt_obese = int(x_obese.varValue)
opt_net   = float(pl.value(model.objective))
spent     = COST_SMOKING * opt_smoke + COST_WEIGHT * opt_obese
gross     = save_per_smoker * opt_smoke + save_per_obese * opt_obese

print("\n==== OPTIMAL DECISION (LP) ====")
print("Solver Status:", pl.LpStatus[status])
print(f"Enroll Smokers: {opt_smoke}  | Enroll Obese: {opt_obese}")
print(f"Budget Spent:   {spent:.2f} / {TOTAL_BUDGET:.2f}")
print(f"Gross Savings:  {gross:.2f}")
print(f"Net Savings:    {opt_net:.2f}")

# -----------------------------
# 6) Heuristic comparison (greedy by net-per-cost)
# -----------------------------
programs = [
    {"name": "Smokers", "unit_cost": COST_SMOKING, "unit_save": save_per_smoker, "cap": cap_smoker},
    {"name": "Obese",   "unit_cost": COST_WEIGHT,  "unit_save": save_per_obese,  "cap": cap_obese }
]
for p in programs:
    p["net_per"] = p["unit_save"] - p["unit_cost"]
    p["efficiency"] = p["net_per"] / p["unit_cost"]

budget_left = TOTAL_BUDGET
alloc = {"Smokers": 0, "Obese": 0}
for p in sorted(programs, key=lambda d: d["efficiency"], reverse=True):
    if p["net_per"] <= 0:
        continue
    can = int(min(p["cap"], budget_left // p["unit_cost"]))
    alloc[p["name"]] = can
    budget_left -= can * p["unit_cost"]

heuristic_gross = sum(d["unit_save"] * alloc[d["name"]] for d in programs)
heuristic_cost  = sum(d["unit_cost"] * alloc[d["name"]] for d in programs)
heuristic_net   = heuristic_gross - heuristic_cost
gap = (opt_net - heuristic_net) / opt_net * 100 if opt_net else 0.0

print("\n==== HEURISTIC (Greedy) ====")
print(f"Enroll Smokers: {alloc['Smokers']}  | Enroll Obese: {alloc['Obese']}")
print(f"Budget Spent:   {heuristic_cost:.2f} / {TOTAL_BUDGET:.2f}")
print(f"Gross Savings:  {heuristic_gross:.2f}")
print(f"Net Savings:    {heuristic_net:.2f}")
print(f"Relative Net-Savings Gap: {gap:.2f}%")

# -----------------------------
# 7) Tables & Visuals for the report
# -----------------------------
# Forecast table
fc_table = pd.DataFrame({
    "Segment": ["Smokers", "Obese"],
    "Forecasted Avg Charges (ARX)": [fc_smoker_charges, fc_obese_charges],
    "Savings Rate": [SAVINGS_RATE_SMOKER, SAVINGS_RATE_OBESE],
    "Savings per Participant": [save_per_smoker, save_per_obese]
})
print("\n--- Forecast-to-Decision Inputs ---")
print(fc_table)

# Optimal allocation summary
summary_df = pd.DataFrame({
    "Program": ["Smoking Cessation", "Weight Reduction"],
    "Enrolled": [opt_smoke, opt_obese],
    "Cost per Participant": [COST_SMOKING, COST_WEIGHT],
    "Total Cost": [COST_SMOKING*opt_smoke, COST_WEIGHT*opt_obese],
    "Gross Savings": [save_per_smoker*opt_smoke, save_per_obese*opt_obese]
})
summary_df["Net Savings"] = summary_df["Gross Savings"] - summary_df["Total Cost"]
print("\n--- Optimal Allocation Summary ---")
print(summary_df)

# Bar chart: LP vs Heuristic (net savings)
plt.figure(figsize=(7,5))
plt.bar(["Optimal LP", "Heuristic"], [opt_net, heuristic_net])
plt.ylabel("Net Savings")
plt.title("Net Savings: Optimal LP vs Heuristic")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Sensitivity Analysis (±10% on budget & savings rates)
# -----------------------------
def solve_lp_with(b_mult=1.0, s_mult=1.0):
    m = pl.LpProblem("Sensitivity_F2D", pl.LpMaximize)
    xs = pl.LpVariable("enroll_smokers", lowBound=0, upBound=N_smokers, cat=pl.LpInteger)
    xo = pl.LpVariable("enroll_obese",   lowBound=0, upBound=N_obese,   cat=pl.LpInteger)

    s_sm = s_mult * save_per_smoker
    s_ob = s_mult * save_per_obese

    m += (s_sm * xs + s_ob * xo - COST_SMOKING * xs - COST_WEIGHT * xo)
    m += COST_SMOKING * xs + COST_WEIGHT * xo <= b_mult * TOTAL_BUDGET
    m += xs <= cap_smoker
    m += xo <= cap_obese

    m.solve(pl.PULP_CBC_CMD(msg=False))
    return float(pl.value(m.objective))

sens_rows = []
for b in [0.9, 1.0, 1.1]:
    for s in [0.9, 1.0, 1.1]:
        val = solve_lp_with(b, s)
        sens_rows.append((b, s, val))

sens_df = pd.DataFrame(sens_rows, columns=["Budget Multiplier", "Savings Multiplier", "Net Savings"])
print("\n--- Sensitivity Analysis (±10% Budget × ±10% Savings) ---")
print(sens_df)

# Optional: pivot-like print to see a grid
pivot = sens_df.pivot(index="Budget Multiplier", columns="Savings Multiplier", values="Net Savings")
print("\nSensitivity Pivot (Net Savings):")
print(pivot.round(2))