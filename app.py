from flask import Flask, send_from_directory, jsonify, request
import os
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

label_dict = {0: "Iris-setosa", 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Load model
model = joblib.load("iris_model.pkl")

# Serve index.html at root URL
@app.route('/')
def serve_index():
    return send_from_directory("frontend", "index.html")

# Serve static files (if you have any CSS/JS)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory("frontend", path)

# Prediction API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if 'features' not in data or not isinstance(data['features'], list) or len(data['features']) != 4:
            return jsonify({"error": "Invalid input. Expected 'features' list with exactly 4 values"}), 400
        
        prediction = label_dict[model.predict([data['features']])[0]]
        return jsonify({"message": "Prediction successful", "received_data": data, "prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
