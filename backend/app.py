import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the credit card dataset
def load_dataset():
    return pd.read_csv('codes/creditcard.csv')

# Load pre-trained anomaly detection model
def load_model():
    with open("codes/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Global variables to avoid reloading
df = load_dataset()
model = load_model()

@app.route('/')
def index():
    # Get basic dataset statistics for the dashboard
    total_transactions = len(df)
    fraud_count = df[df['Class'] == 1].shape[0]
    non_fraud_count = df[df['Class'] == 0].shape[0]
    fraud_percentage = (fraud_count / total_transactions) * 100
    
    stats = {
        'total': total_transactions,
        'fraud': fraud_count,
        'non_fraud': non_fraud_count,
        'fraud_percentage': round(fraud_percentage, 2)
    }
    
    # Pass the first few rows for the data preview table
    preview_data = df.head(5).to_dict('records')
    
    return render_template('index.html', stats=stats, preview_data=preview_data)

@app.route('/detect', methods=['POST'])
def detect_anomaly():
    # Get data from request
    data = request.get_json()
    row_index = data.get('row_index', 0)
    
    # Get the row from the dataset
    if row_index < 0 or row_index >= len(df):
        return jsonify({'error': 'Invalid row index'}), 400
    
    row = df.iloc[row_index]
    
    # Prepare features (exclude Class column)
    features = row.drop('Class').values.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert prediction (-1 for anomaly, 1 for normal) to result
    result = "Anomaly (Fraud)" if prediction[0] == -1 else "Normal (Not Fraud)"
    
    # Get actual class for comparison
    actual_class = "Fraud" if row['Class'] == 1 else "Not Fraud"
    
    return jsonify({
        'row_data': row.to_dict(),
        'prediction': result,
        'actual_class': actual_class
    })

@app.route('/data')
def get_data():
    # Return dataset statistics for charts
    fraud_by_amount = df.groupby('Class')['Amount'].agg(['mean', 'median', 'max']).to_dict()
    
    # Get top correlated features with Class
    correlations = df.corr()['Class'].sort_values(ascending=False).to_dict()
    
    return jsonify({
        'fraud_by_amount': fraud_by_amount,
        'correlations': correlations
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)