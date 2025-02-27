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
    data = request.get_json()
    return jsonify({"message": "Prediction endpoint is working", "received_data": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

