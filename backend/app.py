"""
No-Code ML Pipeline Builder - Flask Backend
A REST API for handling ML pipeline operations
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import uuid
import json
import time
from werkzeug.utils import secure_filename

# Serve React build
FRONTEND_BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'build')

app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for pipeline sessions
pipeline_sessions = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session(session_id):
    """Get or create a pipeline session"""
    if session_id not in pipeline_sessions:
        pipeline_sessions[session_id] = {
            'dataset': None,
            'original_dataset': None,
            'preprocessing': [],
            'target_column': None,
            'train_test_split': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'model': None,
            'model_trained': False,
            'results': None
        }
    return pipeline_sessions[session_id]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'ML Pipeline API is running'})


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create a new pipeline session"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    return jsonify({'session_id': session_id})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return dataset info"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    session_id = request.form.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload CSV or Excel files only.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Get session and store dataset
        session = get_session(session_id)
        session['dataset'] = df.copy()
        session['original_dataset'] = df.copy()
        
        # Get dataset info
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            unique_count = int(df[col].nunique())
            
            # Determine column type
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                col_type = 'numeric'
            else:
                col_type = 'categorical'
            
            column_info.append({
                'name': col,
                'dtype': dtype,
                'type': col_type,
                'null_count': null_count,
                'unique_count': unique_count,
                'sample_values': df[col].dropna().head(3).tolist()
            })
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_info': column_info,
            'preview': df.head(10).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400


@app.route('/api/set-target', methods=['POST'])
def set_target():
    """Set the target column for the ML model"""
    data = request.json
    session_id = data.get('session_id')
    target_column = data.get('target_column')
    
    if not session_id or not target_column:
        return jsonify({'error': 'Session ID and target column are required'}), 400
    
    session = get_session(session_id)
    
    if session['dataset'] is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    
    if target_column not in session['dataset'].columns:
        return jsonify({'error': 'Target column not found in dataset'}), 400
    
    session['target_column'] = target_column
    
    # Get target column info
    target_data = session['dataset'][target_column]
    unique_values = target_data.unique().tolist()
    
    return jsonify({
        'success': True,
        'target_column': target_column,
        'unique_values': unique_values[:20],  # Limit to 20 values
        'num_classes': len(unique_values)
    })


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Apply preprocessing to the dataset"""
    data = request.json
    session_id = data.get('session_id')
    method = data.get('method')  # 'standardize' or 'normalize'
    columns = data.get('columns', [])  # Columns to apply preprocessing to
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    session = get_session(session_id)
    
    if session['dataset'] is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    
    if not session['target_column']:
        return jsonify({'error': 'Please set target column first'}), 400
    
    try:
        df = session['dataset'].copy()
        
        # Get numeric columns if none specified
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column from preprocessing
            if session['target_column'] in columns:
                columns.remove(session['target_column'])
        
        if not columns:
            return jsonify({'error': 'No numeric columns to preprocess'}), 400
        
        # Apply preprocessing
        if method == 'standardize':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            session['preprocessing'].append({'method': 'StandardScaler', 'columns': columns})
        elif method == 'normalize':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            session['preprocessing'].append({'method': 'MinMaxScaler', 'columns': columns})
        else:
            return jsonify({'error': 'Invalid preprocessing method'}), 400
        
        session['dataset'] = df
        
        return jsonify({
            'success': True,
            'method': method,
            'columns_processed': columns,
            'preview': df.head(10).to_dict(orient='records'),
            'preprocessing_steps': session['preprocessing']
        })
    
    except Exception as e:
        return jsonify({'error': f'Preprocessing error: {str(e)}'}), 400


@app.route('/api/reset-preprocessing', methods=['POST'])
def reset_preprocessing():
    """Reset preprocessing to original dataset"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    session = get_session(session_id)
    
    if session['original_dataset'] is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    
    session['dataset'] = session['original_dataset'].copy()
    session['preprocessing'] = []
    session['train_test_split'] = None
    session['X_train'] = None
    session['X_test'] = None
    session['y_train'] = None
    session['y_test'] = None
    
    return jsonify({
        'success': True,
        'message': 'Dataset reset to original',
        'preview': session['dataset'].head(10).to_dict(orient='records')
    })


@app.route('/api/split', methods=['POST'])
def split_data():
    """Perform train-test split"""
    data = request.json
    session_id = data.get('session_id')
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    session = get_session(session_id)
    
    if session['dataset'] is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    
    if not session['target_column']:
        return jsonify({'error': 'Please set target column first'}), 400
    
    try:
        df = session['dataset'].copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Prepare features and target
        X = df.drop(columns=[session['target_column']])
        y = df[session['target_column']]
        
        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)
        
        # Check if target is continuous (for classification we need discrete classes)
        is_continuous = False
        if y.dtype in ['float64', 'float32']:
            # Check if it has too many unique values (likely continuous)
            unique_ratio = len(y.unique()) / len(y)
            if unique_ratio > 0.1 or len(y.unique()) > 20:
                is_continuous = True
        
        # Convert continuous target to categorical using binning
        if is_continuous:
            # Use quantile-based binning for better distribution
            y = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            session['target_binned'] = True
        else:
            session['target_binned'] = False
        
        # Encode target if categorical/object
        if y.dtype == 'object' or hasattr(y, 'cat'):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            session['label_encoder'] = le
        
        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Store in session
        session['X_train'] = X_train
        session['X_test'] = X_test
        session['y_train'] = y_train
        session['y_test'] = y_test
        session['train_test_split'] = {
            'test_size': test_size,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1]
        }
        
        response_data = {
            'success': True,
            'split_info': session['train_test_split'],
            'train_ratio': round((1 - test_size) * 100),
            'test_ratio': round(test_size * 100)
        }
        
        if is_continuous:
            response_data['warning'] = 'Target column had continuous values. Converted to 3 categories (Low, Medium, High) for classification.'
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Split error: {str(e)}'}), 400


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the selected model with comprehensive metrics"""
    data = request.json
    session_id = data.get('session_id')
    model_type = data.get('model_type')  # 'logistic_regression' or 'decision_tree'
    model_params = data.get('model_params', {})
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    session = get_session(session_id)
    
    if session['X_train'] is None:
        return jsonify({'error': 'Please perform train-test split first'}), 400
    
    try:
        start_time = time.time()
        
        # Select model
        if model_type == 'logistic_regression':
            max_iter = model_params.get('max_iter', 1000)
            model = LogisticRegression(max_iter=max_iter, random_state=42, solver='lbfgs')
            model_name = 'Logistic Regression'
        elif model_type == 'decision_tree':
            max_depth = model_params.get('max_depth', None)
            if max_depth == 0:
                max_depth = None
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_split=2, min_samples_leaf=1)
            model_name = 'Decision Tree Classifier'
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Train model
        train_start = time.time()
        model.fit(session['X_train'], session['y_train'])
        train_time = time.time() - train_start
        
        # Make predictions
        y_pred_train = model.predict(session['X_train'])
        y_pred_test = model.predict(session['X_test'])
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(session['X_test'])
        except:
            y_pred_proba = None
        
        # Calculate metrics
        train_accuracy = accuracy_score(session['y_train'], y_pred_train)
        test_accuracy = accuracy_score(session['y_test'], y_pred_test)
        
        # Additional metrics
        precision = precision_score(session['y_test'], y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(session['y_test'], y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(session['y_test'], y_pred_test, average='weighted', zero_division=0)
        
        # Cross-validation scores (if enough samples)
        cv_scores = None
        cv_mean = None
        cv_std = None
        if len(session['X_train']) >= 10:
            try:
                cv_folds = min(5, len(session['X_train']) // 2)
                cv_scores = cross_val_score(model, session['X_train'], session['y_train'], cv=cv_folds, scoring='accuracy')
                cv_mean = round(cv_scores.mean() * 100, 2)
                cv_std = round(cv_scores.std() * 100, 2)
                cv_scores = [round(s * 100, 2) for s in cv_scores.tolist()]
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(session['y_test'], y_pred_test)
        
        # Get unique class labels
        unique_labels = sorted(list(set(session['y_train'].tolist()) | set(session['y_test'].tolist())))
        
        # Feature importance (for decision tree)
        feature_importance = None
        if model_type == 'decision_tree':
            importance = model.feature_importances_
            feature_names = session['X_train'].columns.tolist()
            feature_importance = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 features
            feature_importance = [{'feature': f, 'importance': round(float(i), 4)} for f, i in feature_importance]
        
        # For logistic regression, get coefficients as feature importance
        if model_type == 'logistic_regression':
            try:
                coefficients = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0])
                feature_names = session['X_train'].columns.tolist()
                feature_importance = sorted(
                    zip(feature_names, coefficients),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                feature_importance = [{'feature': f, 'importance': round(float(i), 4)} for f, i in feature_importance]
            except:
                pass
        
        total_time = time.time() - start_time
        
        # Store results
        session['model'] = model
        session['model_trained'] = True
        session['results'] = {
            'model_name': model_name,
            'model_type': model_type,
            'train_accuracy': round(train_accuracy * 100, 2),
            'test_accuracy': round(test_accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm.tolist(),
            'class_labels': unique_labels,
            'feature_importance': feature_importance,
            'train_samples': len(session['X_train']),
            'test_samples': len(session['X_test']),
            'total_features': session['X_train'].shape[1],
            'training_time': round(train_time, 3),
            'total_time': round(total_time, 3),
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        return jsonify({
            'success': True,
            'results': session['results']
        })
    
    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 400


@app.route('/api/pipeline-status', methods=['POST'])
def pipeline_status():
    """Get current pipeline status"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    session = get_session(session_id)
    
    status = {
        'dataset_uploaded': session['dataset'] is not None,
        'target_set': session['target_column'] is not None,
        'preprocessing_applied': len(session['preprocessing']) > 0,
        'data_split': session['train_test_split'] is not None,
        'model_trained': session['model_trained'],
        'target_column': session['target_column'],
        'preprocessing_steps': session['preprocessing'],
        'split_info': session['train_test_split'],
        'results': session['results']
    }
    
    return jsonify(status)


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the entire pipeline session"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    if session_id in pipeline_sessions:
        del pipeline_sessions[session_id]
    
    get_session(session_id)  # Create fresh session
    
    return jsonify({'success': True, 'message': 'Session reset successfully'})


# Serve React Frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "="*60)
    print("🚀 ML Pipeline Builder - Production Server")
    print("="*60)
    print(f"📍 Server running at: http://localhost:{port}")
    print(f"📁 Serving frontend from: {FRONTEND_BUILD}")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
