from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the model only once when the application starts
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # Handle missing or invalid input gracefully
    try:
        cgpa = float(request.form.get('cgpa'))  # Convert to float
        iq = float(request.form.get('iq'))      # Convert to float
        profile_score = float(request.form.get('profile_score'))  # Convert to float
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide numeric values for cgpa, iq, and profile_score.'}), 400

    input_query = np.array([[cgpa, iq, profile_score]])

    # Perform prediction
    try:
        result = model.predict(input_query)[0]
    except Exception as e:
        return jsonify({'error': 'An error occurred while predicting. Please try again later.'}), 500

    return jsonify({'placement': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
