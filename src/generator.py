import dataLogic
import transformerLogic
import torch
import pickle
from transformerLogic import (
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, EMBED_DIM, 
    NUM_HEADS, NUM_LAYERS, FF_DIM, LEARNING_RATE, EPOCHS
)
with open("./data/remap_dict", "rb") as f:
    remapping = pickle.load(f)

newVocabSize = len(remapping)
reverseReMap = dataLogic.reverse_remap()

import random

with open("./data/sequences.pkl", "rb") as f:
    data = pickle.load(f)

song = random.choice(data)
seed = song[:100]


model = transformerLogic.transformer(newVocabSize, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
model.load_state_dict(torch.load("./checkpoints/model_weights_bestLoss.pt")) 

# Generate continuation from seed
output = transformerLogic.creation(model, seed, max_length=1024, temperature=0.75, topK=45)

newSeed = output[650:]

output = transformerLogic.creation(model, newSeed, max_length=1024, temperature=0.75, topK=45)


for i in range(len(output)):
    output[i] = reverseReMap[output[i]]
tokens = dataLogic.intToToken(output)



dataLogic.detokenize_midi(tokens, "./outputs/trial1.midi")

