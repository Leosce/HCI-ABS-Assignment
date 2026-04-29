import torch
import torch.nn as nn
import torch.optim as optim

class BERTFromScratch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, num_classes, max_seq_len):
        super(BERTFromScratch, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        x = self.transformer_blocks(x)
        
        cls_token_output = x[:, 0, :]
        
        return self.classifier(cls_token_output)

VOCAB_SIZE = 100
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_CLASSES = 2
MAX_SEQ_LEN = 10

model = BERTFromScratch(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, NUM_CLASSES, MAX_SEQ_LEN)

train_data = torch.cat([
    torch.randint(0, 50, (10, MAX_SEQ_LEN)),
    torch.randint(50, 100, (10, MAX_SEQ_LEN))
])
train_labels = torch.tensor([0]*10 + [1]*10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the BERT architecture from scratch...")
model.train()
for epoch in range(1, 21):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/20 - Loss: {loss.item():.4f}")

model.eval()
test_input = torch.randint(60, 100, (1, MAX_SEQ_LEN)) 
with torch.no_grad():
    logits = model(test_input)
    prediction = torch.argmax(logits, dim=1).item()
    print(f"\nTest Input (Tokens): {test_input.tolist()[0]}")
    print(f"Predicted Class: {prediction} (Expected: 1)")
