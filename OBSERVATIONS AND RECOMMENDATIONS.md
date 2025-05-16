# OBSERVATIONS: 

**Key Observations from EDA:**

* Dataset contains 188,340 records and 10 columns with no missing or duplicate values.
* Numerical columns: `Sales` and `#Order` are right-skewed and contain outliers.
* Sales and Order are highly correlated (correlation coefficient: 0.94).
* Sales are generally higher without discounts and lower on holidays.
* Peak sales occurred in May 2018; lowest sales in February 2019.
* 365 stores, 4 store types, 5 location types, and 4 region codes are present.

**Key Observations from Hypothesis Testing:**

* Sales significantly vary across:

  * Holidays
  * Discount offerings
  * Store types
  * Location types
  * Region codes

**Key Observations from Feature Engineering:**

* Applied log transformation on `Sales` and `Order` to reduce skewness and handle outliers.
* Extracted time-based features: Year, Month, Day, Day of week, Is\_weekend.
* Created new features:

  * `Discount_offered` (binary flag)
  * `Sales_per_order`
* Encoded categorical variables using:

  * Label Encoding for `Discount`
  * One-Hot Encoding for `Store_Type`, `Location_Type`, and `Region_Code`
* Final dataset contained 26 features used for modeling.

**Key Observations from Model Selection (Initial Attempt):**

* Used `Order` (highly correlated with `Sales`) as a predictor, leading to unrealistically high performance.
* Realized mistake and restarted modeling without using `Order` to avoid data leakage.

**Key Observations from Final Model Training:**

* Models tested: Linear Regression, Random Forest, XGBoost.
* Model Performance:

  * **Linear Regression:** RMSE = 0.6423, MAE = 0.4405, R² = 0.5148
  * **Random Forest:** RMSE = 0.6739, MAE = 0.4635, R² = 0.4660
  * **XGBoost:** RMSE = 0.6188, MAE = 0.4233, R² = 0.5496
* XGBoost performed the best and was selected as the final model.

**Modeling Notes:**

* ARIMA and LSTM were not used because:

  * Dataset is multi-store and includes multiple features beyond time.
  * ARIMA cannot handle categorical variables.
  * LSTM is time-consuming and not required for current complexity.

**Deployment Preparation:**

* XGBoost model saved using Pickle.
* Sales predictions were scaled during modeling; inverse transformed using saved scaler for submission.
* Final test data was feature-engineered with the same process as training data.
* Generated final submission CSV with `ID` and predicted `Sales`.

**Deployment**

* Streamlit web application to upload new data and generate predictions
* Deployed locally and can be extended to platforms like Streamlit Cloud or Hugging Face Spaces


# Recommendations:

1. **Avoid data leakage:** Do not use highly correlated target-like features (`Order`) during modeling.
2. **Ensure consistent feature engineering:** Always apply identical transformations on both train and test datasets.
3. **Validate model assumptions:** Use residual plots and feature importance graphs to validate model behavior.
4. **Use tree-based models for tabular data:** XGBoost provided the best results and should be preferred over linear models for similar tasks.
5. **Avoid complex time series models when unnecessary:** For non-sequential, multi-feature data, XGBoost or Random Forest are more efficient.
6. **Monitor outliers and skewness:** Log transformations and feature scaling significantly improve model performance.
7. **Preserve reproducibility:** Save all pre-processing steps (scalers, encoders, models) for consistent predictions during deployment.


