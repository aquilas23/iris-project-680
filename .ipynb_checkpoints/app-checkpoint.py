from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Default homepage
@app.route('/')
def home():
    return "Iris Model API is running! Use POST /predict to get predictions."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify({"message": "Prediction endpoint is working", "received_data": data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
