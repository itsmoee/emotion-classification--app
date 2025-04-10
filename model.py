import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class EmotionClassifier:
    def __init__(self):
        # Load emotion mapping
        with open('emotion_mapping.json', 'r') as f:
            self.emotion_mapping = json.load(f)
        self.emotions = list(set(self.emotion_mapping.values()))
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'model\emotion_model',
            num_labels=len(self.emotions)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, text):
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return {key: val.to(self.device) for key, val in encoding.items()}

    def predict(self, text):
        self.model.eval()
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_idx = torch.argmax(logits, dim=1).item()
        return self.idx_to_emotion[predicted_idx]

    def retrain(self, text, corrected_emotion):
        # Prepare the new data
        corrected_idx = self.emotion_to_idx[corrected_emotion]
        inputs = self.preprocess(text)
        labels = torch.tensor([corrected_idx]).to(self.device)

        # Create a small dataset for retraining
        dataset = TensorDataset(
            inputs['input_ids'],
            inputs['attention_mask'],
            labels
        )
        dataloader = DataLoader(dataset, batch_size=1)

        # Set model to training mode
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Train for one step
        for batch in dataloader:
            input_ids, attention_mask, label = batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save the updated model
        self.model.save_pretrained('model/emotion_model')