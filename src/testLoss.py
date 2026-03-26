import torch
import transformerLogic
import dataLogic
import pickle
from torch.utils.data import DataLoader
from transformerLogic import (
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, EMBED_DIM,
    NUM_HEADS, NUM_LAYERS, FF_DIM
)

with open("./data/remap_dict", "rb") as f:
    remapping = pickle.load(f)
newVocabSize = len(remapping)

with open("./data/sequences.pkl", "rb") as f:
    data = pickle.load(f)

dataset = dataLogic.MidiDataset(data, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = transformerLogic.transformer(newVocabSize, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
model.load_state_dict(torch.load("./checkpoints/model_weights_bestLoss.pt"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
mask = transformerLogic.create_causal_mask(SEQ_LEN, device)

total_loss = 0
num_batches = 0
with torch.no_grad():
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, mask)
        loss = criterion(logits.view(-1, newVocabSize), y.view(-1))
        total_loss += loss.item()
        num_batches += 1

print(f"Loss: {total_loss / num_batches:.4f}")