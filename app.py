from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Enable Cross-Origin Resource Sharing (CORS)

# Load the model
with open("iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Ensure all fields exist
        if not all(k in data for k in ("sepal_length", "sepal_width", "petal_length", "petal_width")):
            return jsonify({"error": "Missing input data"}), 400

        # Convert input into NumPy array
        features = np.array([[data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]])

        # Make prediction
        prediction = model.predict(features)

        # Return response
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

