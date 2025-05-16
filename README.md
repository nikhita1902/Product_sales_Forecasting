# Product_sales_Forecasting
Forecasting product sales using ML and time series models

---

## 🧠 Problem Statement

To predict daily **sales** of various stores based on features such as store type, location, region, discounts, holidays, and number of orders. The model should generalize well on unseen test data and serve as a foundation for business intelligence dashboards.

---

## 🔍 Exploratory Data Analysis (EDA)

Conducted in `eda.ipynb`, covering:
- Sales distribution
- Trends over time
- Store-wise and region-wise performance
- Holiday and discount effects

---

## ⚙️ Feature Engineering

Performed in `feature_engineering.ipynb`:
- Log transformations on skewed features (`Sales_log`, `Order_log`)
- RobustScaler used to handle outliers
- Cleaned dataset saved for modeling

---

## 🤖 Model Training & Selection

Models tried:
- Linear Regression
- Random Forest
- XGBoost (final model)

Final training in `final_model_training.ipynb`. Model saved as `xgboost_model.pkl`.

---

## 📈 Evaluation

Evaluation metrics:
- RMSE
- MAE
- R² Score

Predictions are inverse-transformed after scaling. Final results saved in `final_submission.csv`.

---

## 📊 Tableau Dashboard

An interactive Tableau Dashboard was created with the following insights:

1. **Sales Performance Dashboard**
   - Time series sales trends
   - Store and location type comparisons

2. **Regional Sales Analysis**
   - Region-wise sales, orders, and average order size

3. **Promotional Impact Analysis**
   - Discounts vs non-discount days
   - Holiday vs normal day sales

4. **Operational Insights**
   - Daily orders vs sales scatter plots
   - Stock behavior trends

5. **Forecast Evaluation**
   - Predicted vs actual sales
   - MAE and RMSE over time

6. **Custom Filters**
   - Store Type
   - Region
   - Holiday / Discount days
   - Date Range

---

## 🚀 Deployment

- Streamlit web application to upload new data and generate predictions
- Deployed locally and can be extended to platforms like Streamlit Cloud or Hugging Face Spaces


---

## 📂 Requirements

```bash
Python 3.10+
pandas
numpy
scikit-learn
xgboost
matplotlib / seaborn
pickle
streamlit 

