from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler once when the server starts
with open('C:\Product_Sale_Forecasting\py_flask_app (deployment)\models(deployment)\xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('C:\Product_Sale_Forecasting\py_flask_app (deployment)\models(deployment)\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Product Sales Forecasting API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # get JSON input
        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply scaler transformation (if needed)
        input_scaled = scaler.transform(input_df)
        
        # Predict using the model
        prediction = model.predict(input_scaled)
        
        # Format the prediction as a list to return as JSON
        prediction_list = prediction.tolist()
        
        return jsonify({'prediction': prediction_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
