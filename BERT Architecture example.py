import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

sentences = [
    "I absolutely loved this movie, it was fantastic!",
    "The acting was terrible and the plot was boring.",
    "The best experience of my life, highly recommended.",
    "I hated every minute of this show.",
    "A true masterpiece of modern cinema.",
    "Waste of time and money, do not watch."
]
labels = [1, 0, 1, 0, 1, 0]

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

dataset = SentimentDataset(sentences, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

print("Fine-tuning BERT for Sentiment Analysis...")
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

test_review = "It was a really great film with amazing visuals."
result = predict_sentiment(test_review)
print(f"\nReview: {test_review}")
print(f"Predicted Sentiment: {result}")