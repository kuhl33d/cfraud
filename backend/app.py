import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
from werkzeug.utils import secure_filename
from flask import session, flash

# Add these imports at the top of your file
import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
from werkzeug.utils import secure_filename
from flask import session, flash

from flask import after_this_request
from datetime import datetime
import functools



# Add these constants after your existing imports
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limit
LOG_FOLDER = 'logs'

# Create uploads and logs directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.urandom(24)  # Required for session management

# Configure logging
log_file_path = os.path.join(LOG_FOLDER, 'app.log')
handler = RotatingFileHandler(log_file_path, maxBytes=10485760, backupCount=10)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Application startup')

# Dictionary to store custom datasets in memory
custom_datasets = {}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the credit card dataset
def load_dataset():
    return pd.read_csv('codes/creditcard.csv')

# Load pre-trained anomaly detection model
def load_model():
    with open("codes/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Add this function to your app.py
def json_serialize(obj):
    """Custom JSON serializer to handle NaN, Infinity, and -Infinity"""
    if isinstance(obj, float):
        if np.isnan(obj):
            return None  # Convert NaN to null
        if np.isinf(obj):
            if obj > 0:
                return "Infinity"  # Convert positive infinity to string
            else:
                return "-Infinity"  # Convert negative infinity to string
    return obj


# Global variables to avoid reloading
df = load_dataset()
model = load_model()


# Add this decorator function to log responses
def log_response(f):
    """Decorator to log response details"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the response from the route function
        response = f(*args, **kwargs)
        
        # Log response details
        if isinstance(response, tuple) and len(response) >= 2:
            # Handle (response, status_code) tuple returns
            resp_data, status_code = response[0], response[1]
        else:
            # Handle direct response returns
            resp_data, status_code = response, 200
        
        # Log response status and data
        app.logger.info(f'Response Status: {status_code}')
        
        # Try to log response data if it's JSON
        if hasattr(resp_data, 'get_json'):
            try:
                resp_json = resp_data.get_json()
                # Truncate large responses to avoid huge log files
                resp_str = json.dumps(resp_json)
                if len(resp_str) > 1000:
                    resp_str = resp_str[:1000] + '... [truncated]'
                app.logger.info(f'Response Data: {resp_str}')
            except Exception as e:
                app.logger.warning(f'Failed to log response data: {str(e)}')
        
        return response
    return decorated_function


@app.route('/')
@log_response
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
@log_response
def detect_anomaly():
    # Get data from request
    data = request.get_json()
    row_index = data.get('row_index', 0)
    
    # Convert row_index to integer
    try:
        row_index = int(row_index)
    except (ValueError, TypeError):
        return jsonify({'error': 'Row index must be a valid integer'}), 400
    
    # Get the row from the dataset
    if row_index < 0 or row_index >= len(df):
        return jsonify({'error': 'Invalid row index'}), 400
    
    row = df.iloc[row_index]
    
    # Prepare features (exclude Class column)
    features = row.drop('Class').values.reshape(1, -1)  # Use .values to remove feature names
    
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
@log_response
def get_data():
    # Enhanced transaction amount statistics
    fraud_df = df[df['Class'] == 1].copy()  # Add .copy() here
    non_fraud_df = df[df['Class'] == 0].copy()  # Add .copy() here
    
    # Transaction amount statistics by class
    amount_stats = {
        'fraud': {
            'mean': round(fraud_df['Amount'].mean(), 2),
            'median': round(fraud_df['Amount'].median(), 2),
            'min': round(fraud_df['Amount'].min(), 2),
            'max': round(fraud_df['Amount'].max(), 2),
            'std': round(fraud_df['Amount'].std(), 2),
            'quartiles': [
                round(fraud_df['Amount'].quantile(0.25), 2),
                round(fraud_df['Amount'].quantile(0.5), 2),
                round(fraud_df['Amount'].quantile(0.75), 2)
            ]
        },
        'non_fraud': {
            'mean': round(non_fraud_df['Amount'].mean(), 2),
            'median': round(non_fraud_df['Amount'].median(), 2),
            'min': round(non_fraud_df['Amount'].min(), 2),
            'max': round(non_fraud_df['Amount'].max(), 2),
            'std': round(non_fraud_df['Amount'].std(), 2),
            'quartiles': [
                round(non_fraud_df['Amount'].quantile(0.25), 2),
                round(non_fraud_df['Amount'].quantile(0.5), 2),
                round(non_fraud_df['Amount'].quantile(0.75), 2)
            ]
        }
    }
    
    # Time-based statistics
    time_stats = {
        'fraud': {
            'mean': round(fraud_df['Time'].mean(), 2),
            'min': round(fraud_df['Time'].min(), 2),
            'max': round(fraud_df['Time'].max(), 2)
        },
        'non_fraud': {
            'mean': round(non_fraud_df['Time'].mean(), 2),
            'min': round(non_fraud_df['Time'].min(), 2),
            'max': round(non_fraud_df['Time'].max(), 2)
        }
    }
    
    # Enhanced feature correlations with fraud
    correlations = df.corr()['Class'].sort_values(ascending=False).to_dict()
    
    # Top positive and negative correlations
    top_positive = {k: round(v, 4) for k, v in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]}
    top_negative = {k: round(v, 4) for k, v in sorted(correlations.items(), key=lambda x: x[1])[:10]}
    
    # Amount distribution by bins
    bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    bin_labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000+']
    fraud_df['amount_bin'] = pd.cut(fraud_df['Amount'], bins=bins, labels=bin_labels)
    non_fraud_df['amount_bin'] = pd.cut(non_fraud_df['Amount'], bins=bins, labels=bin_labels)
    
    fraud_amount_dist = fraud_df['amount_bin'].value_counts().sort_index().to_dict()
    non_fraud_amount_dist = non_fraud_df['amount_bin'].value_counts().sort_index().to_dict()
    
    return jsonify({
        'amount_stats': amount_stats,
        'time_stats': time_stats,
        'correlations': {
            'top_positive': top_positive,
            'top_negative': top_negative,
            'all': correlations
        },
        'amount_distribution': {
            'fraud': fraud_amount_dist,
            'non_fraud': non_fraud_amount_dist
        }
    })

@app.route('/paginated_data')
@log_response
def get_paginated_data():
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Validate and limit parameters
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:  # Limit max items per page
        per_page = 10
    
    # Calculate start and end indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get total number of records
    total_records = len(df)
    total_pages = (total_records + per_page - 1) // per_page  # Ceiling division
    
    # Get the data for the current page
    page_data = df.iloc[start_idx:end_idx].to_dict('records')
    
    # Add row index to each record
    for i, record in enumerate(page_data):
        record['index'] = start_idx + i
    
    # Return paginated data with metadata
    return jsonify({
        'data': page_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_records': total_records,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    })

@app.route('/filter_data')
@log_response
def filter_data():
    # Get filter parameters
    class_filter = request.args.get('class', None, type=int)  # 0 for non-fraud, 1 for fraud
    min_amount = request.args.get('min_amount', None, type=float)
    max_amount = request.args.get('max_amount', None, type=float)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Apply filters
    filtered_df = df.copy()
    
    if class_filter is not None:
        filtered_df = filtered_df[filtered_df['Class'] == class_filter]
    
    if min_amount is not None:
        filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
    
    if max_amount is not None:
        filtered_df = filtered_df[filtered_df['Amount'] <= max_amount]
    
    # Validate and limit pagination parameters
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 10
    
    # Calculate start and end indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get total number of filtered records
    total_records = len(filtered_df)
    total_pages = (total_records + per_page - 1) // per_page
    
    # Get the data for the current page
    page_data = filtered_df.iloc[start_idx:end_idx].to_dict('records')
    
    # Add row index to each record
    for i, record in enumerate(page_data):
        record['index'] = filtered_df.index[start_idx + i]  # Use original index from the dataframe
    
    # Return filtered and paginated data with metadata
    return jsonify({
        'data': page_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_records': total_records,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        },
        'filters': {
            'class': class_filter,
            'min_amount': min_amount,
            'max_amount': max_amount
        }
    })

@app.route('/upload', methods=['POST'])
@log_response
def upload_file():
    app.logger.info('File upload initiated')
    if 'file' not in request.files:
        app.logger.warning('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.warning('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        app.logger.warning(f'Invalid file type: {file.filename}')
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    # Generate a unique ID for this upload
    dataset_id = str(uuid.uuid4())
    app.logger.info(f'Generated dataset ID: {dataset_id}')
    
    # Save the file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}.csv")
    file.save(file_path)
    app.logger.info(f'File saved to {file_path}')
    
    try:
        # Try to load the dataset
        custom_df = pd.read_csv(file_path)
        app.logger.info(f'Successfully loaded CSV with {len(custom_df)} rows')
        
        # Validate the dataset schema
        required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'Amount']
        
        # Check if the dataset has the required columns
        missing_columns = [col for col in required_columns if col not in custom_df.columns]
        if missing_columns:
            app.logger.warning(f'Missing required columns: {missing_columns}')
            os.remove(file_path)  # Clean up the file
            return jsonify({
                'error': f"Missing required columns: {', '.join(missing_columns)}"
            }), 400
        
        # Add Class column if not present (default to 0)
        if 'Class' not in custom_df.columns:
            app.logger.info('Adding missing Class column with default value 0')
            custom_df['Class'] = 0
        
        # Store the dataset in memory
        custom_datasets[dataset_id] = custom_df
        
        # Store the dataset ID in the session
        session['current_dataset_id'] = dataset_id
        
        # Return basic stats about the dataset
        total_transactions = len(custom_df)
        fraud_count = custom_df[custom_df['Class'] == 1].shape[0]
        non_fraud_count = custom_df[custom_df['Class'] == 0].shape[0]
        fraud_percentage = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
        
        app.logger.info(f'Upload successful. Dataset stats: {total_transactions} total, {fraud_count} fraud')
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'stats': {
                'total': total_transactions,
                'fraud': fraud_count,
                'non_fraud': non_fraud_count,
                'fraud_percentage': round(fraud_percentage, 2)
            },
            'preview': custom_df.head(5).to_dict('records')
        })
    
    except Exception as e:
        # Log the full exception with traceback
        app.logger.error(f'Error processing uploaded file: {str(e)}\n{traceback.format_exc()}')
        # Clean up the file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 400

@app.route('/dataset/<dataset_id>/detect', methods=['POST'])
@log_response
def detect_anomaly_custom_dataset(dataset_id):
    # Get data from request
    data = request.get_json()
    row_index = data.get('row_index', 0)
    
    # Convert row_index to integer
    try:
        row_index = int(row_index)
    except (ValueError, TypeError):
        return jsonify({'error': 'Row index must be a valid integer'}), 400
    
    # Check if the dataset exists
    if dataset_id not in custom_datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    custom_df = custom_datasets[dataset_id]
    
    # Get the row from the dataset
    if row_index < 0 or row_index >= len(custom_df):
        return jsonify({'error': 'Invalid row index'}), 400
    
    row = custom_df.iloc[row_index]
    
    # Prepare features (exclude Class column)
    features = row.drop('Class').values.reshape(1, -1)  # Use .values to remove feature names
    
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

@app.route('/dataset/<dataset_id>/analyze', methods=['GET'])
@log_response
def analyze_dataset(dataset_id):
    app.logger.info(f'Analyzing dataset: {dataset_id}')
    # Check if the dataset exists
    if dataset_id not in custom_datasets:
        app.logger.warning(f'Dataset not found: {dataset_id}')
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        custom_df = custom_datasets[dataset_id]
        app.logger.info(f'Dataset loaded with {len(custom_df)} records')
        
        # Prepare features (exclude Class column if present)
        X = custom_df.drop(columns=['Class'])
        
        # Remove feature names to avoid the sklearn warning
        X_no_names = X.values
        
        # Make predictions
        app.logger.info('Running model predictions')
        predictions = model.predict(X_no_names)
        
        # Convert predictions from -1 to 1 (anomaly) and 1 to 0 (normal)
        predictions = [1 if x == -1 else 0 for x in predictions]
        
        # Count anomalies detected
        anomalies_count = sum(predictions)
        app.logger.info(f'Detected {anomalies_count} anomalies')
        
        # Add predictions to the dataframe
        custom_df['Prediction'] = predictions
        
        # Calculate accuracy if Class column is present
        accuracy = None
        confusion_matrix = None
        if 'Class' in custom_df.columns:
            true_labels = custom_df['Class'].values
            accuracy = (predictions == true_labels).mean()
            
            # Create confusion matrix
            tp = sum((predictions == 1) & (true_labels == 1))
            fp = sum((predictions == 1) & (true_labels == 0))
            tn = sum((predictions == 0) & (true_labels == 0))
            fn = sum((predictions == 0) & (true_labels == 1))
            
            confusion_matrix = {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn)
            }
        
        # Calculate transaction amount statistics
        fraud_pred_df = custom_df[custom_df['Prediction'] == 1].copy()  # Add .copy() here
        non_fraud_pred_df = custom_df[custom_df['Prediction'] == 0].copy()  # Add .copy() here
        
        # Transaction amount statistics by predicted class
        amount_stats = {
            'fraud': {
                'mean': round(fraud_pred_df['Amount'].mean(), 2) if not fraud_pred_df.empty else 0,
                'median': round(fraud_pred_df['Amount'].median(), 2) if not fraud_pred_df.empty else 0,
                'min': round(fraud_pred_df['Amount'].min(), 2) if not fraud_pred_df.empty else 0,
                'max': round(fraud_pred_df['Amount'].max(), 2) if not fraud_pred_df.empty else 0,
                'std': round(fraud_pred_df['Amount'].std(), 2) if not fraud_pred_df.empty else 0,
                'quartiles': [
                    round(fraud_pred_df['Amount'].quantile(0.25), 2) if not fraud_pred_df.empty else 0,
                    round(fraud_pred_df['Amount'].quantile(0.5), 2) if not fraud_pred_df.empty else 0,
                    round(fraud_pred_df['Amount'].quantile(0.75), 2) if not fraud_pred_df.empty else 0
                ]
            },
            'non_fraud': {
                'mean': round(non_fraud_pred_df['Amount'].mean(), 2) if not non_fraud_pred_df.empty else 0,
                'median': round(non_fraud_pred_df['Amount'].median(), 2) if not non_fraud_pred_df.empty else 0,
                'min': round(non_fraud_pred_df['Amount'].min(), 2) if not non_fraud_pred_df.empty else 0,
                'max': round(non_fraud_pred_df['Amount'].max(), 2) if not non_fraud_pred_df.empty else 0,
                'std': round(non_fraud_pred_df['Amount'].std(), 2) if not non_fraud_pred_df.empty else 0,
                'quartiles': [
                    round(non_fraud_pred_df['Amount'].quantile(0.25), 2) if not non_fraud_pred_df.empty else 0,
                    round(non_fraud_pred_df['Amount'].quantile(0.5), 2) if not non_fraud_pred_df.empty else 0,
                    round(non_fraud_pred_df['Amount'].quantile(0.75), 2) if not non_fraud_pred_df.empty else 0
                ]
            }
        }
        
        # Calculate feature correlations with prediction
        correlations = {}
        if 'Prediction' in custom_df.columns:
            # Add Prediction column to calculate correlations
            correlations = custom_df.corr()['Prediction'].sort_values(ascending=False).to_dict()
            
            # Top positive and negative correlations
            top_positive = {k: round(v, 4) for k, v in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]}
            top_negative = {k: round(v, 4) for k, v in sorted(correlations.items(), key=lambda x: x[1])[:10]}
        
        # Get anomalies only
        anomalies_df = custom_df[custom_df['Prediction'] == 1]
        anomalies_preview = anomalies_df.head(10).to_dict('records') if not anomalies_df.empty else []
        
        return jsonify(json.loads(json.dumps({
            'success': True,
            'dataset_id': dataset_id,
            'total_records': len(custom_df),
            'anomalies_detected': int(anomalies_count),
            'anomalies_percentage': round((anomalies_count / len(custom_df)) * 100, 2),
            'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
            'confusion_matrix': confusion_matrix,
            'preview_with_predictions': custom_df.head(10).to_dict('records'),
            'anomalies_preview': anomalies_preview,
            'amount_stats': amount_stats,
            'correlations': {
                'top_positive': top_positive if 'Prediction' in custom_df.columns else {},
                'top_negative': top_negative if 'Prediction' in custom_df.columns else {},
                'all': correlations
            }
        }, default=json_serialize)))
    except Exception as e:
        # Log the full exception with traceback
        app.logger.error(f'Error analyzing dataset: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500

@app.route('/datasets', methods=['GET'])
@log_response
def list_datasets():
    datasets = []
    for dataset_id, df in custom_datasets.items():
        datasets.append({
            'id': dataset_id,
            'total_records': len(df),
            'upload_time': dataset_id  # Using UUID as a proxy for time
        })
    
    return jsonify({
        'success': True,
        'datasets': datasets
    })

@app.route('/dataset/<dataset_id>', methods=['DELETE'])
@log_response
def delete_dataset(dataset_id):
    # Check if the dataset exists
    if dataset_id not in custom_datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Remove from memory
    del custom_datasets[dataset_id]
    
    # Remove file if it exists
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}.csv")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify({
        'success': True,
        'message': 'Dataset deleted successfully'
    })

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the exception
    app.logger.error(f'Unhandled exception: {str(e)}\n{traceback.format_exc()}')
    # Return JSON instead of HTML for HTTP errors
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    # app.run(debug=True, port=port)
    app.run(debug=True,port=int(os.environ.get('PORT', 80)))

# Add this after app configuration
@app.before_request
def log_request_info():
    """Log request details before processing"""
    # Don't log static file requests
    if request.path.startswith('/static/'):
        return
    
    # Log basic request info
    app.logger.info(f'Request: {request.method} {request.path}')
    app.logger.info(f'Headers: {dict(request.headers)}')
    
    # Log request body for appropriate methods
    if request.method in ['POST', 'PUT', 'PATCH']:
        if request.is_json:
            # For JSON requests
            try:
                body = request.get_json()
                app.logger.info(f'Request JSON: {json.dumps(body)}')
            except Exception as e:
                app.logger.warning(f'Failed to parse JSON request: {str(e)}')
        elif request.form:
            # For form data
            app.logger.info(f'Request Form: {dict(request.form)}')
        elif request.files:
            # For file uploads, just log file names
            files = {k: v.filename for k, v in request.files.items()}
            app.logger.info(f'Request Files: {files}')
