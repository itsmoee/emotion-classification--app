import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify, session, redirect, url_for
from datetime import datetime
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import logging
import threading
import time
import yaml
import shutil
from model import EmotionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
CORS(app)  # Allow cross-origin requests for API usage

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load emotion mapping
with open('emotion_mapping.json', 'r') as f:
    emotion_mapping = json.load(f)
valid_emotions = list(set(emotion_mapping.values()))

# Path to the interaction log
LOG_FILE = config['app']['log_file']
USERS_FILE = config['app']['users_file']
BATCH_RETRAIN_INTERVAL = config['model']['batch_retrain_interval_hours'] * 3600  # Convert to seconds

# Initialize the log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'timestamp',
        'user_id',
        'input_text',
        'predicted_emotion',
        'confidence',
        'user_feedback',
        'corrected_emotion'
    ]).to_csv(LOG_FILE, index=False)

# Initialize users file if it doesn't exist
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({
            "admin": {
                "password": generate_password_hash("admin"),
                "role": "admin"
            }
        }, f)

# Initialize the emotion classifier
model = EmotionClassifier()

# Global variables to track retraining status
last_retrain_timestamp = None
retraining_in_progress = False


# Authentication functions
def load_users():
    """Load users from JSON file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


def login_required(f):
    """Decorator for routes that require login"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator for routes that require admin privileges"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))

        users = load_users()
        if session['user_id'] not in users or users[session['user_id']]['role'] != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403

        return f(*args, **kwargs)

    return decorated_function


# Scheduled background tasks
def schedule_batch_retraining():
    """Background thread for scheduled model retraining"""
    global last_retrain_timestamp, retraining_in_progress
    while True:
        time.sleep(BATCH_RETRAIN_INTERVAL)
        try:
            logger.info("Starting scheduled batch retraining")
            retraining_in_progress = True
            model.batch_retrain_from_logs(sample_size=config['model']['batch_sample_size'])
            last_retrain_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            retraining_in_progress = False
            logger.info("Scheduled batch retraining completed")
        except Exception as e:
            retraining_in_progress = False
            logger.error(f"Error in scheduled batch retraining: {e}")


# Start background task if enabled
if config['model']['enable_scheduled_retraining']:
    retraining_thread = threading.Thread(target=schedule_batch_retraining, daemon=True)
    retraining_thread.start()
    logger.info(f"Scheduled batch retraining started (interval: {BATCH_RETRAIN_INTERVAL / 3600} hours)")


# Web routes
@app.route('/')
def index():
    """Main page"""
    user_id = session.get('user_id', None)
    return render_template('index.html',
                           valid_emotions=valid_emotions,
                           logged_in=(user_id is not None),
                           user_id=user_id)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        user_id = data.get('user_id')
        password = data.get('password')

        users = load_users()

        if user_id in users and check_password_hash(users[user_id]['password'], password):
            session['user_id'] = user_id
            session['role'] = users[user_id]['role']

            if request.is_json:
                return jsonify({'success': True, 'redirect': '/'})
            return redirect('/')

        if request.is_json:
            return jsonify({'error': 'Invalid credentials'}), 401
        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.pop('user_id', None)
    session.pop('role', None)
    return redirect('/')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if config['app']['enable_registration'] == False:
        return jsonify({'error': 'Registration is disabled'}), 403

    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        user_id = data.get('user_id')
        password = data.get('password')

        users = load_users()

        if user_id in users:
            if request.is_json:
                return jsonify({'error': 'User already exists'}), 400
            return render_template('register.html', error='User already exists')

        users[user_id] = {
            'password': generate_password_hash(password),
            'role': 'user'
        }

        save_users(users)
        session['user_id'] = user_id
        session['role'] = 'user'

        if request.is_json:
            return jsonify({'success': True, 'redirect': '/'})
        return redirect('/')

    return render_template('register.html')


# API routes
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get user_id from session or anonymous
        user_id = session.get('user_id', 'anonymous')

        # Predict emotion with confidence
        predicted_emotion, confidence = model.predict(text, return_confidence=True)

        # Debug log
        logger.info(
            f"Predicted emotion: {predicted_emotion}, Type: {type(predicted_emotion)}, Value: {predicted_emotion}")

        # If predicted_emotion is an integer or can be converted to one
        if isinstance(predicted_emotion, (int, np.integer)) or (
                isinstance(predicted_emotion, str) and predicted_emotion.isdigit()):

            # Convert to integer if it's a string
            idx = int(predicted_emotion) if isinstance(predicted_emotion, str) else predicted_emotion

            # First try the idx_to_emotion mapping
            if idx in model.idx_to_emotion:
                predicted_emotion = model.idx_to_emotion[idx]
            else:
                # Try to find the key in emotion_mapping that has this value
                found = False
                for emotion, emotion_idx in model.emotion_mapping.items():
                    if emotion_idx == idx:
                        predicted_emotion = emotion
                        found = True
                        break

                # If we still haven't found it, try to get it from the inverse_mapping
                if not found and idx in model.inverse_mapping:
                    predicted_emotion = model.inverse_mapping[idx]

        # Final check: is predicted_emotion a valid emotion name?
        valid_emotions = list(set(model.emotion_mapping.values()))
        if isinstance(predicted_emotion, (int, np.integer)) or predicted_emotion.isdigit():
            # Still a number - get from emotion_mapping.json directly
            with open('emotion_mapping.json', 'r') as f:
                emotion_dict = json.load(f)
                # Reverse lookup: find key for this value
                idx = int(predicted_emotion)
                for emotion, emotion_idx in emotion_dict.items():
                    if emotion_idx == idx:
                        predicted_emotion = emotion
                        break

        logger.info(f"Final predicted emotion to return: {predicted_emotion}")

        return jsonify({
            'predicted_emotion': predicted_emotion,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Error in /api/predict: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for feedback"""
    data = request.get_json()
    input_text = data.get('input_text', '')
    predicted_emotion = data.get('predicted_emotion', '')
    user_feedback = data.get('feedback', '')
    confidence = data.get('confidence', 0.0)

    # Get user_id from session or anonymous
    user_id = session.get('user_id', 'anonymous')

    # Validate feedback
    if user_feedback.lower() == 'pass':
        corrected_emotion = predicted_emotion
        retraining_needed = False
    else:
        corrected_emotion = user_feedback.lower()
        if corrected_emotion not in valid_emotions:
            return jsonify({
                'error': f'Invalid emotion: {corrected_emotion}. Valid emotions are: {valid_emotions}'
            }), 400
        retraining_needed = True

    # Log the interaction
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'user_id': user_id,
        'input_text': input_text,
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'user_feedback': user_feedback,
        'corrected_emotion': corrected_emotion
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # Retrain the model if feedback is not 'pass' and incremental training is enabled
    message = f'Feedback received: {user_feedback}.'
    if retraining_needed and config['model']['enable_incremental_training']:
        model.retrain(input_text, corrected_emotion)
        message += ' Model updated.'
    else:
        message += ' Model not updated.'

    return jsonify({'message': message})


@app.route('/api/stats')
@login_required
def api_stats():
    """Get emotion classification statistics"""
    # Read the log file
    if not os.path.exists(LOG_FILE):
        return jsonify({'error': 'No log data available'}), 404

    df = pd.read_csv(LOG_FILE)

    # Basic stats
    total_entries = len(df)
    unique_users = len(df['user_id'].unique())
    feedback_given = len(df[df['user_feedback'] != 'pass'])

    # Calculate accuracy based on feedback
    corrected_df = df[df['user_feedback'] != 'pass']
    if len(corrected_df) > 0:
        accuracy = (df['predicted_emotion'] == df['corrected_emotion']).mean()
    else:
        accuracy = None

    # Calculate emotion distribution
    emotion_counts = df['predicted_emotion'].value_counts().to_dict()

    # Get recent entries
    recent_entries = df.tail(10).to_dict('records')

    # Get patterns of disagreement
    conflict_patterns = model.get_conflicting_predictions(LOG_FILE)

    return jsonify({
        'total_entries': total_entries,
        'unique_users': unique_users,
        'feedback_given': feedback_given,
        'accuracy': accuracy,
        'emotion_distribution': emotion_counts,
        'recent_entries': recent_entries,
        'conflict_patterns': conflict_patterns
    })


@app.route('/api/retrain', methods=['POST'])
@admin_required
def api_retrain():
    """Force model retraining from logs"""
    global last_retrain_timestamp, retraining_in_progress
    # Optional sample size from request
    data = request.get_json() or {}
    sample_size = data.get('sample_size', None)

    try:
        retraining_in_progress = True
        success = model.batch_retrain_from_logs(sample_size=sample_size)
        last_retrain_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        retraining_in_progress = False
        if success:
            return jsonify({'message': 'Model successfully retrained from logs'})
        else:
            return jsonify({'message': 'No new data available for retraining'})
    except Exception as e:
        retraining_in_progress = False
        logger.error(f"Error in manual retraining: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/backup', methods=['POST'])
@admin_required
def api_backup():
    """Create a backup of the current model and logs"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'backups/{timestamp}'

    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)

        # Backup model
        model.backup_model()

        # Backup logs
        shutil.copy(LOG_FILE, f'{backup_dir}/interaction_log.csv')

        return jsonify({
            'message': 'Backup created successfully',
            'backup_dir': backup_dir
        })
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotions')
def api_emotions():
    """Return valid emotions"""
    return jsonify({
        'emotions': valid_emotions,
        'count': len(valid_emotions)
    })


@app.route('/api/system_status', methods=['GET'])
def api_system_status():
    """Return the current system status"""
    return jsonify({
        'status': 'retraining' if retraining_in_progress else 'idle'
    })


@app.route('/api/retrain_status', methods=['GET'])
def api_retrain_status():
    """Return the last retrain timestamp"""
    return jsonify({
        'last_retrained': last_retrain_timestamp
    })


@app.route('/download_log')
@admin_required
def download_log():
    """Download the interaction log"""
    return send_file(LOG_FILE, as_attachment=True)


# Legacy routes for backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    """Legacy prediction endpoint"""
    return api_predict()


@app.route('/feedback', methods=['POST'])
def feedback():
    """Legacy feedback endpoint"""
    return api_feedback()


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('model/backups', exist_ok=True)
    os.makedirs('backups', exist_ok=True)

    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = config['app'].get('debug', False)
    app.run(host='0.0.0.0', port=port, debug=debug)