"""
app.py — Flask Web Application for the AI Importer Discovery Engine
Run with: python app.py
"""
import os
import sys

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from main import run_importer_discovery_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/discover", methods=["POST"])
def discover():
    data = request.get_json()
    product = data.get("product", "").strip()
    country = data.get("country", "").strip()
    top_n = int(data.get("top_n", 10))

    if not product or not country:
        return jsonify({"error": "Product and Country are required."}), 400

    try:
        results = run_importer_discovery_pipeline(product, country, top_n)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
