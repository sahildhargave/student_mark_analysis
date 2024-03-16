from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form.get('cgpa'))  # Convert to float
    iq = float(request.form.get('iq'))      # Convert to float
    profile_score = float(request.form.get('profile_score'))  # Convert to float

    input_query = np.array([[cgpa, iq, profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
