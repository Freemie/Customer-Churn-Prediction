from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
print("Loading churn prediction model...")
with open('churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input array
        input_data = []
        for feature in features:
            input_data.append(data.get(feature, 0))
        
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scale numerical features
        numerical_features = ['tenure', 'monthly_charges', 'total_charges', 'charge_per_month']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk': 'high' if probability > 0.7 else 'medium' if probability > 0.3 else 'low'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return '''
    <h1>Customer Churn Prediction API</h1>
    <p>Send POST request to /predict with customer data</p>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    
    