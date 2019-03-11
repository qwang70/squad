import json

idx2word = {}
with open("data/word2idx.json") as f:
    word2idx = json.load(f)
    for key in word2idx:
        idx2word[word2idx[key]] = key
with open('data/idx2word.json', 'w') as outfile:  
    json.dump(idx2word, outfile)


idx2char = {}
with open("data/char2idx.json") as f:
    char2idx = json.load(f)
    for key in char2idx:
        idx2char[char2idx[key]] = key
with open('data/idx2char.json', 'w') as outfile:  
    json.dump(idx2char, outfile)