"""
app.py — Flask Web Application for the AI Importer Discovery Engine
Run with: python app.py
"""
import os
import sys
import json
import re

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def _slugify(text: str) -> str:
    """Lowercase, collapse whitespace to underscores, strip non-alnum."""
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text

def _find_cached(product: str, country: str):
    """
    Return (data, filename) if a matching cached JSON exists, else (None, None).
    Matching strategy (in order):
      1. Exact slug match  →  {product}_{country}_importers.json
      2. Filename contains both product-slug and country-slug
    """
    if not os.path.isdir(OUTPUTS_DIR):
        return None, None

    p_slug = _slugify(product)
    c_slug = _slugify(country)
    files  = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".json")]

    # 1. exact
    exact = f"{p_slug}_{c_slug}_importers.json"
    if exact in files:
        path = os.path.join(OUTPUTS_DIR, exact)
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), exact

    # 2. fuzzy – both slugs appear somewhere in the filename
    for fname in files:
        stem = fname.replace("_importers.json", "")
        if p_slug in stem and c_slug in stem:
            path = os.path.join(OUTPUTS_DIR, fname)
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), fname

    return None, None


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cached", methods=["POST"])
def cached():
    """Return pre-stored results for a product+country without hitting any API."""
    body    = request.get_json() or {}
    product = body.get("product", "").strip()
    country = body.get("country", "").strip()
    top_n   = int(body.get("top_n", 10))

    if not product or not country:
        return jsonify({"error": "Product and Country are required."}), 400

    data, fname = _find_cached(product, country)
    if data is None:
        return jsonify({"error": f"No cached results found for '{product}' in '{country}'. "
                                  "Try one of the available cached searches."}), 404

    return jsonify(data[:top_n])


@app.route("/api/list-cached", methods=["GET"])
def list_cached():
    """Return a list of all pre-stored query combinations available offline."""
    if not os.path.isdir(OUTPUTS_DIR):
        return jsonify([])

    entries = []
    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        if not fname.endswith("_importers.json"):
            continue
        stem = fname[: -len("_importers.json")]          # e.g. ceramic_tiles_germany
        # split on the last occurrence of a known country slug (heuristic: last word)
        parts     = stem.split("_")
        country   = parts[-1].replace("_", " ").title()  # simple heuristic
        product   = " ".join(parts[:-1]).replace("_", " ").title()
        entries.append({"product": product, "country": country, "file": fname})

    return jsonify(entries)


@app.route("/api/discover", methods=["POST"])
def discover():
    """Live discovery pipeline – requires valid API keys."""
    # Import lazily so the app starts even if keys are missing
    try:
        from main import run_importer_discovery_pipeline
    except Exception as exc:
        return jsonify({"error": f"Pipeline unavailable: {exc}"}), 503

    data    = request.get_json() or {}
    product = data.get("product", "").strip()
    country = data.get("country", "").strip()
    top_n   = int(data.get("top_n", 10))

    if not product or not country:
        return jsonify({"error": "Product and Country are required."}), 400

    try:
        results = run_importer_discovery_pipeline(product, country, top_n)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

