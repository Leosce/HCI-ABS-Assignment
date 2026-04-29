import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

sentences = [
    "I absolutely loved this movie, it was fantastic!",
    "The acting was terrible and the plot was boring.",
    "The best experience of my life, highly recommended.",
    "I hated every minute of this show.",
    "A true masterpiece of modern cinema.",
    "Waste of time and money, do not watch."
]
labels = np.array([1, 0, 1, 0, 1, 0])

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def encode_sentences(texts, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
        
    return np.array(input_ids), np.array(attention_masks)

input_ids, attention_masks = encode_sentences(sentences, tokenizer)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print("Fine-tuning BERT with TensorFlow...")
model.fit(
    [input_ids, attention_masks], 
    labels, 
    batch_size=2, 
    epochs=3
)

def predict_sentiment(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    
    logits = model(encoded['input_ids'], attention_mask=encoded['attention_mask']).logits
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    return "Positive" if prediction == 1 else "Negative"

test_review = "It was a really great film with amazing visuals."
result = predict_sentiment(test_review)
print(f"\nReview: {test_review}")
print(f"Predicted Sentiment: {result}")