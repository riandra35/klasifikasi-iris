from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# ======================================================
# Load Model (sekali saja saat aplikasi dijalankan)
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knn_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Label kelas
SPECIES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


# ======================================================
# Home
# ======================================================

@app.route("/")
def index():
    return render_template("index.html")


# ======================================================
# Prediction API
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]])

        prediction = model.predict(features)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0]
            confidence = round(np.max(probability) * 100, 2)

        return jsonify({
            "success": True,
            "species": SPECIES[int(prediction)],
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ======================================================
# Health Check
# ======================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "loaded"
    })


# ======================================================
# Local Development Only
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
