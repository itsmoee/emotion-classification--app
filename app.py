import os
import json
import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify
from datetime import datetime
from model import EmotionClassifier

app = Flask(__name__)

# Load emotion mapping
with open('emotion_mapping.json', 'r') as f:
    emotion_mapping = json.load(f)
valid_emotions = list(set(emotion_mapping.values()))

# Initialize the emotion classifier
model = EmotionClassifier()

# Path to the interaction log
LOG_FILE = 'interaction_log.csv'

# Initialize the log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'input_text', 'predicted_emotion', 'user_feedback', 'corrected_emotion']).to_csv(LOG_FILE, index=False)

@app.route('/')
def index():
    return render_template('index.html', valid_emotions=valid_emotions)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Predict emotion
    predicted_emotion = model.predict(text)
    return jsonify({'predicted_emotion': predicted_emotion})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    input_text = data.get('input_text', '')
    predicted_emotion = data.get('predicted_emotion', '')
    user_feedback = data.get('feedback', '')

    # Validate feedback
    if user_feedback.lower() == 'pass':
        corrected_emotion = predicted_emotion
    else:
        corrected_emotion = user_feedback.lower()
        if corrected_emotion not in valid_emotions:
            return jsonify({'error': f'Invalid emotion: {corrected_emotion}. Valid emotions are: {valid_emotions}'}), 400

    # Log the interaction
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'input_text': input_text,
        'predicted_emotion': predicted_emotion,
        'user_feedback': user_feedback,
        'corrected_emotion': corrected_emotion
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # Retrain the model if feedback is not 'pass'
    if user_feedback.lower() != 'pass':
        model.retrain(input_text, corrected_emotion)
        return jsonify({'message': f'Feedback received: {user_feedback}. Model updated.'})
    return jsonify({'message': f'Feedback received: {user_feedback}. No retraining needed.'})

@app.route('/download_log')
def download_log():
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))