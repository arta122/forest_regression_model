import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve input values
            Temperature = float(request.form.get('temperature'))
            Rh = float(request.form.get('rh'))
            Ws = float(request.form.get('ws'))
            Rain = float(request.form.get('rain'))
            FFMC = float(request.form.get('ffmc'))
            DMC = float(request.form.get('dmc'))
            ISI = float(request.form.get('isi'))

            # Check whether 'classes' or 'Region' should be included
            # Assuming 'Region' was not part of training data, we remove it
            classes = float(request.form.get('classes'))  # Ensure numeric
            # Region = float(request.form.get('region'))  # REMOVE this if not in training

            # Check feature count before scaling
            input_features = np.array([[Temperature, Rh, Ws, Rain, FFMC, DMC, ISI, classes]])
            print(f"Input shape: {input_features.shape}")  # Debugging

            # Scale input data
            new_data_scaled = scaler_model.transform(input_features)

            # Predict using the model
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=result[0])

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
