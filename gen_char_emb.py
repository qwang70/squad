from ujson import dump
from ujson import load as json_load
import numpy as np

with open('data/word2idx.json', 'r') as fh:
    word2idx = json_load(fh)
with open('data/char2idx.json', 'r') as fh:
    char2idx = json_load(fh)
with open('data/char_emb.json', 'r') as fh:
    char_emb = json_load(fh)
char_emb_for_word = {}
for word in word2idx:
    if word not in char_emb_for_word:
        wid = word2idx[word]
        emb = None
        for c in word:
            if emb is None:
                emb = np.array(char_emb[char2idx[c]])
            else:
                emb += np.array(char_emb[char2idx[c]])
        char_emb_for_word[int(wid)] = emb
with open("data/char_emb_for_word.json", "w") as ujson_file:  
    dump(char_emb_for_word, ujson_file)
