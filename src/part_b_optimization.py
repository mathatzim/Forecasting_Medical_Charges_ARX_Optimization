# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:15:22 2025

@author: mathaios
"""

# Import libraries
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp

# Resolve project root and data path (so scripts run from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost.csv'
SAMPLE_DATA_PATH = PROJECT_ROOT / 'data/raw/medical_cost_sample.csv'
if not DATA_PATH.exists() and SAMPLE_DATA_PATH.exists():
    DATA_PATH = SAMPLE_DATA_PATH


# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

# Define "obese" as BMI >= 30
df["obese"] = (df["bmi"] >= 30).astype(int)

# Extract key groups
n_smokers = df["smoker"].value_counts().get("yes", 0)
n_obese = df["obese"].sum()

avg_smoker_charges = df[df["smoker"] == "yes"]["charges"].mean()
avg_obese_charges = df[df["obese"] == 1]["charges"].mean()

print(f"Smokers in dataset: {n_smokers} | Obese in dataset: {n_obese}")
print(f"Avg charges - smokers: {avg_smoker_charges:.2f} | obese: {avg_obese_charges:.2f}")

# -------------------------------
# 2. Parameters
# -------------------------------
budget = 120000
cost_smoking_program = 500
cost_weight_program = 400

savings_smoker = avg_smoker_charges * 0.225  # ~22.5% reduction
savings_obese = avg_obese_charges * 0.294    # ~29.4% reduction

max_smoking_capacity = 137   # chosen capacity
max_weight_capacity = 128    # chosen capacity

# -------------------------------
# 3. LP Model
# -------------------------------
model = pulp.LpProblem("Health_Optimization", pulp.LpMaximize)

# Decision variables
x_smokers = pulp.LpVariable("Enroll_Smokers", lowBound=0, upBound=n_smokers, cat="Integer")
x_obese = pulp.LpVariable("Enroll_Obese", lowBound=0, upBound=n_obese, cat="Integer")

# Objective: maximize net savings
model += (x_smokers * (savings_smoker - cost_smoking_program) +
          x_obese * (savings_obese - cost_weight_program)), "Net_Savings"

# Constraints
model += x_smokers * cost_smoking_program + x_obese * cost_weight_program <= budget, "Budget"
model += x_smokers <= max_smoking_capacity, "Capacity_Smoking"
model += x_obese <= max_weight_capacity, "Capacity_Weight"

# Solve
model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n==== OPTIMAL LP SOLUTION ====")
print("Solver Status:", pulp.LpStatus[model.status])
print("Enroll in Smoking Cessation:", int(x_smokers.value()))
print("Enroll in Weight Program:   ", int(x_obese.value()))
budget_spent = int(x_smokers.value())*cost_smoking_program + int(x_obese.value())*cost_weight_program
gross_savings = int(x_smokers.value())*savings_smoker + int(x_obese.value())*savings_obese
net_savings = pulp.value(model.objective)
print(f"Budget Spent:                {budget_spent:.2f} / {budget:.2f}")
print(f"Gross Savings:               {gross_savings:.2f}")
print(f"Net Savings (Objective):     {net_savings:.2f}")

# -------------------------------
# 4. Heuristic (Greedy)
# -------------------------------
def greedy_allocation(budget):
    options = []
    options.append(("smoker", savings_smoker - cost_smoking_program, cost_smoking_program, max_smoking_capacity))
    options.append(("obese", savings_obese - cost_weight_program, cost_weight_program, max_weight_capacity))
    options.sort(key=lambda x: x[1]/x[2], reverse=True)

    budget_left = budget
    enrolled = {"smoker": 0, "obese": 0}
    net = 0
    for group, net_benefit, cost, cap in options:
        can_enroll = min(cap, budget_left // cost)
        enrolled[group] = int(can_enroll)
        net += can_enroll * net_benefit
        budget_left -= can_enroll * cost
    return enrolled, net

greedy_enrolled, greedy_net = greedy_allocation(budget)
print("\n==== HEURISTIC (Greedy) SOLUTION ====")
print("Enroll in Smoking Cessation :", greedy_enrolled["smoker"])
print("Enroll in Weight Program    :", greedy_enrolled["obese"])
print(f"Net Savings (Heuristic):     {greedy_net:.2f}")
print(f"Relative Gap: {(net_savings - greedy_net)/net_savings*100:.2f}%")

# -------------------------------
# 5. Sensitivity Analysis
# -------------------------------
print("\n==== SENSITIVITY (±10% Budget & Savings) ====")
sensitivity_results = []
for b_factor in [0.9, 1.0, 1.1]:
    for s_factor in [0.9, 1.0, 1.1]:
        model = pulp.LpProblem("Sensitivity", pulp.LpMaximize)
        xs = pulp.LpVariable("Enroll_Smokers", lowBound=0, upBound=n_smokers, cat="Integer")
        xo = pulp.LpVariable("Enroll_Obese", lowBound=0, upBound=n_obese, cat="Integer")

        model += xs * (s_factor*savings_smoker - cost_smoking_program) + \
                 xo * (s_factor*savings_obese - cost_weight_program)

        model += xs*cost_smoking_program + xo*cost_weight_program <= b_factor*budget
        model += xs <= max_smoking_capacity
        model += xo <= max_weight_capacity

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        net = pulp.value(model.objective)
        sensitivity_results.append((b_factor, s_factor, net))
        print(f"Budget x{b_factor}, Savings x{s_factor} => Net Savings: {net:.2f}")

# Convert sensitivity results to DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results, columns=["Budget Multiplier", "Savings Multiplier", "Net Savings"])

# -------------------------------
# 6. Reporting Tables & Plots
# -------------------------------
# Summary table
summary_df = pd.DataFrame({
    "Program": ["Smoking Cessation", "Weight Reduction"],
    "Enrolled": [int(x_smokers.value()), int(x_obese.value())],
    "Cost per Participant": [cost_smoking_program, cost_weight_program],
    "Total Cost": [int(x_smokers.value())*cost_smoking_program,
                   int(x_obese.value())*cost_weight_program],
    "Gross Savings": [int(x_smokers.value())*savings_smoker,
                      int(x_obese.value())*savings_obese]
})
summary_df["Net Savings"] = summary_df["Gross Savings"] - summary_df["Total Cost"]
print("\n--- Optimal Allocation Summary Table ---")
print(summary_df)

# Sensitivity table
print("\n--- Sensitivity Analysis Table ---")
print(sensitivity_df)

# Visualization: LP vs Heuristic
plt.figure(figsize=(7,5))
plt.bar(["Optimal LP", "Heuristic"], [net_savings, greedy_net], color=["steelblue","orange"])
plt.ylabel("Net Savings")
plt.title("Comparison of LP vs Heuristic Allocation")
plt.grid(axis="y")
plt.show()