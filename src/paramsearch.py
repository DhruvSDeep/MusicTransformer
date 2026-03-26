import dataLogic
import transformerLogic
import torch
import pickle
import random

from transformerLogic import (
    VOCAB_SIZE, SEQ_LEN, EMBED_DIM,
    NUM_HEADS, NUM_LAYERS, FF_DIM
)

with open("./data/remap_dict", "rb") as f:
    remapping = pickle.load(f)
newVocabSize = len(remapping)
reverseReMap = dataLogic.reverse_remap()

model = transformerLogic.transformer(newVocabSize, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
model.load_state_dict(torch.load("./checkpoints/model_weights_bestLoss.pt"))

with open("./data/sequences.pkl", "rb") as f:
    data = pickle.load(f)

song = random.choice(data)
seed = song[:100]

settings = [
    (0.3, 5),
    (0.4, 10),
    (0.5, 10),
    (0.5, 20),
    (0.6, 15),
    (0.6, 25),
    (0.7, 15),
    (0.7, 30),
    (0.8, 20),
    (0.8, 40),
    (0.9, 30),
    (1.0, 50),
]

for temp, k in settings:
    output = transformerLogic.creation(model, seed, max_length=1024, temperature=temp, topK=k)
    mapped = [reverseReMap[t] for t in output]
    tokens = dataLogic.intToToken(mapped)
    dataLogic.detokenize_midi(tokens, f"./outputs/t{temp}_k{k}.mid")
    print(f"Generated t{temp}_k{k}.mid")

# Save reference
ref = [reverseReMap[t] for t in song]
ref_tokens = dataLogic.intToToken(ref)
dataLogic.detokenize_midi(ref_tokens, "./outputs/reference.mid")
print("Generated reference.mid")