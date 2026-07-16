from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model KNN
# Menggunakan os.path untuk memastikan Vercel dapat menemukan file model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'knn_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input dari form HTML
        sepal_length = float(request.form['SepalLengthCm'])
        sepal_width = float(request.form['SepalWidthCm'])
        petal_length = float(request.form['PetalLengthCm'])
        petal_width = float(request.form['PetalWidthCm'])
        
        # Konversi input menjadi array numpy 2D (sesuai format X_test)
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Melakukan prediksi
        prediction = model.predict(features)
        
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        return render_template('index.html', error=str(e))

# Penting untuk Vercel (namun tidak dijalankan di environment serverless)
if __name__ == '__main__':
    app.run(debug=True)
