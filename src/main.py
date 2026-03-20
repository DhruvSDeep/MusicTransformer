import transformerLogic
import dataLogic
import glob
import os
from torch.utils.data import DataLoader
from torch import save
import pickle
from elim_tokens import tokensToEliminate, reMap

from transformerLogic import (
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, EMBED_DIM, 
    NUM_HEADS, NUM_LAYERS, FF_DIM, LEARNING_RATE, EPOCHS
)

negListDefault = []
og = False

if not (os.path.exists("./data/sequences.pkl")):
    og = True
    dataPaths =glob.glob("./data/**/*.midi", recursive=True)
    data = []
    for i in range(len(dataPaths)):
        try:
            data.append(dataLogic.tokenToInt(dataLogic.tokenize_midi(dataPaths[i]), negListDefault))
        except:
            print('1 file failed')

    with open("./data/sequences.pkl", "wb") as f:
        pickle.dump(data, f)

negList = tokensToEliminate(6)

remapping = reMap(VOCAB_SIZE, negList)
with open("./data/remap_dict", "wb") as f:
    pickle.dump(remapping, f)



if og == True:
    with open("./data/sequences.pkl", "rb") as f:
        data = pickle.load(f)
    for j in range(len(data)):
        data[j] = [remapping[t] for t in data[j] if t in remapping]
    with open("./data/sequences.pkl", "wb") as f:
        pickle.dump(data, f)

with open("./data/sequences.pkl", "rb") as f:
    data = pickle.load(f)

newVocabSize = len(remapping)

# Debug check
max_id = max(max(seq) for seq in data)
print(f"Max token ID in data: {max_id}")
print(f"Vocab size: {newVocabSize}")


dataset = dataLogic.MidiDataset(data, seq_len=512)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)



model = transformerLogic.transformer(newVocabSize, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
transformerLogic.train(model, dataloader, EPOCHS, newVocabSize)
save(model.state_dict(), "./checkpoints/model_weights.pt")

