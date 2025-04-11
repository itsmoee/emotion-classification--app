import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import logging
import yaml

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


class EmotionClassifier:
    def __init__(self, model_dir='model/emotion_model', device=None):
        # Load emotion mapping
        with open('emotion_mapping.json', 'r') as f:
            self.emotion_mapping = json.load(f)

        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create inverse mapping for convenience
        self.inverse_mapping = {v: k for k, v in self.emotion_mapping.items()}

        # Get unique emotions
        self.emotions = list(set(self.emotion_mapping.values()))
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        logger.info(f"Emotion to Index Mapping: {self.emotion_to_idx}")
        logger.info(f"Index to Emotion Mapping: {self.idx_to_emotion}")

        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load model
        try:
            logger.info(f"Loading model from {model_dir}")
            self.model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.emotions)
            )
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base BERT model")
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(self.emotions)
            )
            self.model.to(self.device)

        # Initialize metrics tracking
        self.metrics = {
            'accuracy': [],
            'f1_scores': [],
            'training_loss': [],
            'validation_loss': []
        }

        # Load metrics if they exist
        metrics_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")

        # Initialize feedback buffer for batch retraining
        self.feedback_buffer = []
        self.MAX_BUFFER_SIZE = config['model'].get('max_feedback_buffer_size', 10)  # Use config value

        # Evaluation mode by default
        self.model.eval()

    def preprocess(self, text):
        """
        Preprocess text for BERT model
        """
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def predict(self, text, return_confidence=False):
        """
        Predict emotion from text with optional confidence score
        """
        try:
            self.model.eval()
            inputs = self.preprocess(text)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()

            # Get the actual emotion name using inverse_mapping
            # The values in emotion_mapping are indices, so we need to map back
            emotion_name = None

            # First try getting from idx_to_emotion mapping
            if predicted_idx in self.idx_to_emotion:
                emotion_name = self.idx_to_emotion[predicted_idx]

            # If that doesn't work, check if any key in inverse_mapping maps to this index
            if emotion_name is None:
                for emotion, idx in self.emotion_mapping.items():
                    if idx == predicted_idx:
                        emotion_name = emotion
                        break

            # Last resort fallback
            if emotion_name is None:
                emotion_name = str(predicted_idx)

            if return_confidence:
                return emotion_name, confidence
            return emotion_name
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def retrain(self, text, corrected_emotion):
        """
        Retrain the model with new feedback
        """
        try:
            self.feedback_buffer.append((text, corrected_emotion))

            if len(self.feedback_buffer) >= self.MAX_BUFFER_SIZE:
                logger.info("Feedback buffer full. Initiating retraining...")
                self.batch_retrain()
                self.feedback_buffer = []
            else:
                logger.info(
                    f"Feedback added to buffer. Current buffer size: {len(self.feedback_buffer)}/{self.MAX_BUFFER_SIZE}")
        except Exception as e:
            logger.error(f"Error in retraining: {e}")
            raise

    def batch_retrain(self):
        """
        Perform batch retraining with feedback buffer
        """
        if not self.feedback_buffer:
            logger.info("No feedback available for retraining")
            return False

        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

            for text, corrected_emotion in self.feedback_buffer:
                inputs = self.preprocess(text)
                labels = torch.tensor([self.emotion_to_idx[corrected_emotion]]).to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save the updated model
            self.model.save_pretrained('model/emotion_model')

            # Update metrics
            self.metrics['training_loss'].append(loss.item())
            self.save_metrics()

            logger.info("Batch retraining completed successfully")
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Error in batch retraining: {e}")
            self.model.eval()
            raise

    def batch_retrain_from_logs(self, sample_size=None):
        """
        Batch retrain the model using interaction logs
        """
        try:
            df = pd.read_csv('interaction_log.csv')
            if len(df) == 0:
                logger.info("No data in logs for retraining")
                return False

            # Filter entries with feedback
            feedback_df = df[df['user_feedback'].notnull() & (df['user_feedback'] != 'pass')]
            if len(feedback_df) == 0:
                logger.info("No feedback data in logs for retraining")
                return False

            # Sample data if sample_size is specified
            if sample_size:
                feedback_df = feedback_df.sample(n=min(sample_size, len(feedback_df)), random_state=42)

            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

            for _, row in feedback_df.iterrows():
                text = row['input_text']
                corrected_emotion = row['corrected_emotion']

                if corrected_emotion not in self.emotion_to_idx:
                    continue

                inputs = self.preprocess(text)
                labels = torch.tensor([self.emotion_to_idx[corrected_emotion]]).to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save the updated model
            self.model.save_pretrained('model/emotion_model')

            # Update metrics
            self.metrics['training_loss'].append(loss.item())
            self.save_metrics()

            logger.info("Batch retraining from logs completed successfully")
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Error in batch retraining from logs: {e}")
            self.model.eval()
            raise

    def save_metrics(self):
        """
        Save metrics to file
        """
        metrics_path = os.path.join('model/emotion_model', 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)

    def backup_model(self):
        """
        Create a backup of the current model
        """
        import shutil
        import datetime

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'model/backups/emotion_model_{timestamp}'
        shutil.copytree('model/emotion_model', backup_path)
        logger.info(f"Model backed up to {backup_path}")

    def get_conflicting_predictions(self, log_file):
        """
        Analyze logs for conflicting predictions
        """
        try:
            df = pd.read_csv(log_file)
            conflicts = df[df['user_feedback'].notnull() & (df['user_feedback'] != 'pass')]

            if len(conflicts) == 0:
                return {}

            # Group by predicted vs corrected emotion
            conflict_patterns = conflicts.groupby(['predicted_emotion', 'corrected_emotion']).agg({
                'input_text': ['count', lambda x: list(x)[:3]]
            }).reset_index()

            conflict_patterns.columns = ['predicted_emotion', 'corrected_emotion', 'count', 'examples']
            total_conflicts = conflict_patterns['count'].sum()

            result = {}
            for _, row in conflict_patterns.iterrows():
                pattern = f"{row['predicted_emotion']} -> {row['corrected_emotion']}"
                result[pattern] = {
                    'count': row['count'],
                    'ratio': row['count'] / total_conflicts,
                    'examples': row['examples']
                }

            return result
        except Exception as e:
            logger.error(f"Error analyzing conflicting predictions: {e}")
            return {}