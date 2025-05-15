import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open('C:\Product_Sale_Forecasting\py_flask_app (deployment)\models(deployment)\xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the RobustScaler used to scale sales
with open('C:\Product_Sale_Forecasting\py_flask_app (deployment)\models(deployment)\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 19 features the model expects (same order!)
feature_names = [
    'Store_id', 'Holiday', 'Discount', 'Year', 'Month', 'Day',
    'DayOfWeek', 'Is_Weekend', 'Discount_Offered', 
    'Store_Type_S2', 'Store_Type_S3', 'Store_Type_S4',
    'Location_Type_L2', 'Location_Type_L3', 'Location_Type_L4', 'Location_Type_L5',
    'Region_Code_R2', 'Region_Code_R3', 'Region_Code_R4'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract the input values in correct order
        input_data = [data[feature] for feature in feature_names]
        input_array = np.array(input_data).reshape(1, -1)

        # Predict (sales in scaled form)
        scaled_prediction = model.predict(input_array)

        # Inverse transform to get actual sales
        actual_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        return jsonify({
            'predicted_sales': round(actual_prediction, 2)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing feature in input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
