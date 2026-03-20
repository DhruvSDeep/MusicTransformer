import pickle
from transformerLogic import VOCAB_SIZE
import os
'''
total_tokens = len(list(data.keys()))

for l in range(1, 20):
        
    runsum = 0
    sum=0
    negToken = 0

    for i in list(data.keys()):
        if data[i] > l:
            runsum += int(data[i])
        else:
            negToken+=1
        sum+=int(data[i])

    print('thresh = ', l, '- covers in % : ', (runsum*100/sum), ' tokens perc eliminated : ', (negToken/total_tokens))'''

def create_occurence_dict():
    with open("./data/sequences.pkl", "rb") as f:
        data = pickle.load(f)
    masterDict={}
    for i in range(VOCAB_SIZE):
        masterDict[i] = 0
    for seq in data:
        for i in seq:
            masterDict[i] += 1
    with open("./data/occurence_dictionary.pkl", "wb") as f:
        pickle.dump(masterDict, f)


def tokensToEliminate(threshold):

    if not os.path.exists("./data/occurence_dictionary.pkl"):
        create_occurence_dict()

    negList = []
    with open("./data/occurence_dictionary.pkl", 'rb') as f:
        data = pickle.load(f)
    for i in list(data.keys()):
        if data[i] <= threshold:
            negList.append(i)
    return(negList)

def reMap(vocab, negList):
    negset = set(negList)
    negset.discard(0)  # always keep pad, bos and eos
    negset.discard(1)  
    negset.discard(2) 
    reMapping ={}
    idx = 0
    for i in range (vocab):
        if i not in negset:
            reMapping[i] = idx
            idx +=1
    return(reMapping)
