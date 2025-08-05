import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from data_loader import load_data
import torch.nn.functional as F


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

        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
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
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Text: {sentence}")
        print(f"Prediction: {sentiment_map[prediction]}")
        print()

def train_bert():
    print("Starting BERT training for financial sentiment analysis!")


    print("Loading data...")
    df = load_data()
    sentences = df['Sentence'].tolist()
    labels = df['label'].tolist()

    # Step 2: Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")

    # Step 3: Create tokenizer and model
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Step 4: Create datasets and dataloaders
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print("Data preparation complete!")
    print("Ready to start training...")

    # Step 5: Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Step 6: Training loop
    print("Starting training...")
    for epoch in range(2):  # Train for 2 epochs
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    print("Training complete!")
    test_trained_model(model, tokenizer)



if __name__ == "__main__":
    train_bert()
