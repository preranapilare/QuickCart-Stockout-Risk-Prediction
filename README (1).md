# QuickCart Warehouse Inventory Stockout Risk Prediction

## Overview
An end-to-end 3-class supervised machine-learning project for predicting daily stockout risk at Store + SKU + Day level.

Classes:
- Safe
- At-Risk
- Imminent

The main business objective is early identification of **Imminent** stockout cases.

## Dataset
Five CSV tables are used:
- `dim_stores.csv`
- `dim_skus.csv`
- `dim_suppliers.csv`
- `dim_events.csv`
- `fact_inventory_daily.csv`

The dataset contains 21,600 records covering October 1–30, 2026.

## Target Distribution
- Safe: 14,131 (65.42%)
- At-Risk: 5,186 (24.01%)
- Imminent: 2,283 (10.57%)

## Data Preparation
The workflow includes data loading, table merging, supplier-reliability cleaning, missing-value handling, categorical encoding, numerical scaling, and feature engineering.

Important engineered features:
- `reorder_gap`
- `days_of_cover_ratio`
- `supplier_reliability_clean`
- `is_recent_reorder`
- `day_of_month`
- `days_since_festival_start`

`lead_time_days_actual` was excluded from predictive features because it is largely missing and can create leakage concerns.

## Train/Test Split
Chronological split:
- Training: October 1–23, 2026 — 16,560 records
- Testing: October 24–30, 2026 — 5,040 records

## Models
1. Majority-class baseline
2. Logistic Regression
3. Random Forest
4. Gradient Boosting

## Model Results

| Model | Accuracy | Balanced Accuracy | Imminent Recall | Imminent Precision | Imminent F1 |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 62.28% | 33.33% | 0.00% | 0.00% | 0.00% |
| Logistic Regression | 89.35% | 84.15% | **98.45%** | 62.23% | 76.26% |
| Random Forest | 94.13% | 88.48% | 72.52% | 87.95% | 79.49% |
| Gradient Boosting | **95.62%** | **91.44%** | 80.00% | **90.78%** | **85.05%** |

## Model Interpretation
There is a clear trade-off between overall performance and Imminent-class recall.

- Logistic Regression has the highest Imminent recall: **98.45%**.
- Gradient Boosting has the highest accuracy, balanced accuracy, Imminent precision, and Imminent F1.

Because the project emphasizes **Imminent recall**, Logistic Regression is documented as the recall-focused final model, while Gradient Boosting is retained as the higher-overall-performance comparison model.

## Gradient Boosting Confusion Matrix

| Actual / Predicted | At-Risk | Imminent | Safe |
|---|---:|---:|---:|
| At-Risk | 1,063 | 63 | 0 |
| Imminent | 155 | 620 | 0 |
| Safe | 3 | 0 | 3,136 |

For Imminent: 620 of 775 cases were detected, giving 80.00% recall.

## Feature Importance
The leading Gradient Boosting features were:
1. `days_of_cover`
2. `reorder_gap`
3. `days_of_cover_ratio`
4. `supplier_reliability_clean`
5. `lead_time_variance_days`

## Business Insights
The model can support inventory teams by identifying high-risk Store + SKU combinations early. Possible actions include prioritizing imminent-risk SKUs for replenishment, monitoring low-stock products, considering supplier reliability, and using stock coverage/reorder gaps as early-warning signals.

## Project Structure

```text
QuickCart_Stockout_Risk/
├── data/
│   ├── dim_stores.csv
│   ├── dim_skus.csv
│   ├── dim_suppliers.csv
│   ├── dim_events.csv
│   └── fact_inventory_daily.csv
├── notebooks/
│   └── QuickCart_Stockout_Risk.ipynb
├── models/
│   ├── quickcart_stockout_final_model.pkl
│   └── quickcart_gradient_boosting_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── README.md
└── requirements.txt
```

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Jupyter Notebook / Google Colab.

## Conclusion
The project demonstrates an end-to-end machine-learning workflow for warehouse stockout-risk prediction, with chronological evaluation and explicit attention to the Imminent class. The results show that model choice depends on whether the priority is maximum Imminent recall or stronger overall classification performance.
