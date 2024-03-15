from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.externals import joblib  # For scikit-learn <= 0.23
# from joblib import load  # For scikit-learn >= 0.24

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    cgpa = float(request.form.get('cgpa'))
    iq = float(request.form.get('iq'))
    profile_score = float(request.form.get('profile_score'))

    # Create input array
    input_query = np.array([[cgpa, iq, profile_score]])

    # Make prediction
    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
