# Forecasting Medical Charges (ARIMA/ARX) + Optimization (LP)

Coursework project combining **predictive** analytics (time-series forecasting) and **prescriptive** analytics (linear programming)
on the Kaggle *Medical Cost* dataset.

This repo reproduces the workflow described in the report:
- **Part A:** ARIMA vs ARX forecasting on a BMI-sorted “pseudo-time” series for the most common age group (age 18).
- **Part B:** Budget allocation between a smoking-cessation program and a weight-reduction program to maximize **net savings**.
- **Part C:** A forecast-to-decision pipeline that uses **ARX forecasts** as inputs to the optimization.

## Key results (from the report)

### Part A (Age 18, BMI-sorted pseudo-time)
- ARIMA(1,0,0): **AIC 1474.21**, **RMSE 10,099.25**
- ARX(1,0,0) with exogenous variables (BMI, sex, smoker, region): **RMSE ~5,281** and improved AIC (see report)

### Part B (Optimization, static averages)
- Budget: **$120,000**
- Costs: **$500** per smoker program participant, **$400** per obese program participant
- Reported optimal allocation: **137 smokers**, **128 obese**, spend **$119,700**
- Reported savings (Part B section): gross **~$1.57M**, net **~$1.45M**

### Part C (Forecast-to-decision)
- Uses **ARX(1,0,0) with BMI as exogenous** to produce one-step-ahead forecasts for smokers and obese subgroups.
- Reported optimal allocation remains **137 / 128**, with higher projected savings (see report’s Executive Summary).

## How to run

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Run each part
```bash
python -m src.part_a_forecasting
python -m src.part_b_optimization
python -m src.part_c_forecast_to_decision
```

## Data
- `data/raw/medical_cost_sample.csv (sample for repo)` — Kaggle “Medical Cost Dataset” (see report for citation/source).

## Repo structure
- `src/part_a_forecasting.py` — ARIMA vs ARX on age-18 BMI-sorted pseudo-time series
- `src/part_b_optimization.py` — LP + heuristic on static averages
- `src/part_c_forecast_to_decision.py` — ARX subgroup forecasts → LP decision
- `docs/ForecastingMedicalCharges.docx` — original report

## Notes
- This dataset is cross-sectional; the “time series” is a **pseudo-time ordering** (BMI-sorted), used for demonstrating ARIMA/ARX mechanics.
- Some reported values (e.g., AIC) can vary slightly by `statsmodels` version and default settings.


### Getting the full dataset
To reproduce the exact report results, download the *Medical Cost Dataset* from Kaggle (see report) and place it at `data/raw/medical_cost.csv`.
