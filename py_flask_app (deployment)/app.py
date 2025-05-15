import os
import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Define base directory (where this app.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler and model using relative paths
scaler_path = os.path.join(BASE_DIR, 'models(deployment)', 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'models(deployment)', 'xgboost_model.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract features in the expected order
    features = [
        data.get('Store_id'),
        data.get('Store_Type'),
        data.get('Location_Type'),
        data.get('Region_Code'),
        data.get('Holiday'),
        data.get('Discount')
    ]

    # Convert to numpy array and reshape for scaler/model
    features_array = np.array(features).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features_array)

    # Predict
    prediction = model.predict(features_scaled)

    # Return prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
