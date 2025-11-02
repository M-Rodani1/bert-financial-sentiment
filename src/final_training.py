import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from data.data_loader import load_data
from data.bert_model import BertSentimentClassifier
from sklearn.metrics import classification_report
import numpy as np

class EarlyStopping:
    def __init__(self, patience=3,min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    def __call__(self, val_loss, model):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            return False
        if val_loss < self.best_loss -self.min_delta:
            # Improvement!
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            print(f" validation loss improved to {val_loss:.4f}")
            return False
        else:
            # No improvement
            self.counter += 1
            print(f"NO improvement for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered! best loss was {self.best_loss:.4f}")
                return True
            return False
    def save_checkpoint(self, model):
        """Save model weights"""
        import copy
        self.best_model = copy.deepcopy(model.state_dict())
    def load_best_model(self,model):
        """Load the best model weights"""
        if self.best_model is not  None:
            model.load_state_dict(self.best_model)
            print(f"Best model loaded with validation loss {self.best_loss:.4f}")
def calculate_class_weights(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(unique_labels)
    weights = total_samples / (num_classes * counts)
    return torch.FloatTensor(weights)

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def test_trained_model(model, tokenizer):
    print("\nTesting trained model on new examples:")
    test_sentences = [
        "Tesla stock surged 15% after beating earnings expectations",
        "Company announces massive layoffs amid financial struggles",
        "Market remains stable with modest trading volumes",
        "Apple shares plummet following disappointing iPhone sales",
        "Investors show strong confidence in renewable energy sector"
    ]

    model.eval()
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Text: {sentence}")
        print(f"Prediction: {sentiment_map[prediction]}")
        print()


def train_bert():
    print("Starting BERT training for financial sentiment analysis!")

    # Step 1: Load and prepare data
    df = load_data()
    sentences = df["Sentence"].tolist()
    labels = df["label"].tolist()

    # Step 2: Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_labels)
    print(f"\nClass weights: Negative={class_weights[0]:.4f}, Neutral={class_weights[1]:.4f}, Positive={class_weights[2]:.4f}")

    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")

    # Step 3: Load model + tokenizer from bert_model.py
    classifier = BertSentimentClassifier("bert-base-uncased")
    tokenizer = classifier.tokenizer
    model = classifier.model

    # Step 4: Create datasets and dataloaders
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    # Create weighted loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create early stopping
    early_stopping = EarlyStopping(patience=3,min_delta=0.001)

    # Calculate total training steps
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10
    print(f"total training steps: {total_steps}")
    print(f"warmup steps: {warmup_steps}")


    # Step 5: Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Step 6: Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            loss = criterion(outputs.logits, batch["label"])
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        model.eval()
        all_predictions = []
        all_labels = []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = criterion(outputs.logits, batch["label"])
                val_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                true_labels = batch['label']
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.2e}")
        print("\nclassification report:")
        print(classification_report(all_labels, all_predictions,target_names=["negative", "neutral", "positive"],digits=4))
        if early_stopping(avg_val_loss, model):
            break


    print("Training complete!")
    early_stopping.load_best_model(model)
    test_trained_model(model, tokenizer)

if __name__ == "__main__":
    train_bert()
