from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = np.array([[
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"])
    ]])

    pred = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0].max() * 100
    else:
        prob = None

    return jsonify({
        "species": species[int(pred)],
        "confidence": round(prob,2) if prob else None
    })


if __name__ == "__main__":
    app.run(debug=True)
