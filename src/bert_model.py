import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class BertSentimentClassifier:
    def __init__(self,model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=3)
    def predict(self,text,):
        inputs = self.tokenizer(text,add_special_tokens=True,padding=True,truncation=True,max_length=128,return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions
