from flask import Flask, send_from_directory, jsonify, request
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")

#  Serve index.html at root URL
@app.route('/')
def serve_index():
    return send_from_directory("frontend", "index.html")

#  Serve static files (if you have any CSS/JS)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory("frontend", path)

#  Prediction API Endpoint
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
