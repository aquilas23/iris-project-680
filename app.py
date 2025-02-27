from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([np.array(data['features'])])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
