import numpy as np
import spacy
import json

# load spacy model
nlp = spacy.load('en_core_web_sm')

# build dictionary
poses = ['POS', 'PUNCT', 'SYM', 'ADJ', 'CCONJ', 'NUM', 'DET', 'ADV', 'ADP', 'X', 'VERB', 'NOUN', 'PROPN', 'PART', 'INTJ', 'PRON', '']
pos_dict = {}
for i, pos in enumerate(poses):
    pos_dict[pos] = i + 1

ners = [ 'PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT', 'LAW', 'WORK_OF_ART','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
ner_dict = {}
for i, pos in enumerate(ners):
    ner_dict[pos] = i + 1

def find_overlapping_span(token_span, span):
    start = -1
    end = -1
    for idx, s in enumerate(span):
        if start == -1:
            if s[0] <= token_span[0] and token_span[0] <= s[0]:
                start = idx
        if start != -1:
            if s[1] <= token_span[1] and token_span[1] <= s[1]:
                end = idx
                break
    return (start, end)
eval_file = 'data/test_eval.json' 
with open(eval_file) as f:
    data = json.load(f)
    size = len(data)
    pos = np.zeros((size, 400, len(pos_dict)))
    ner = np.zeros((size, 400, len(ner_dict)))

    for num in data:
        idx = int(num) - 1
        if idx % 1000 == 0:
            print(idx)
        context = data[num]['context']
        span = data[num]['spans']
        doc = nlp(context)
        # pos
        for token in doc:
            if token.pos_ not in poses:
                continue
            pos_idx = pos_dict[token.pos_]
            # print(token.text, token.pos_, token.pos)
            token_span = (token.idx, token.idx + len(token.text))
            # print(context)
            # print("token span", token_span)
            start, end = find_overlapping_span(token_span, span)
            # print(start, end)
            if start >= 0 and end >= 0 and start < 400 and end < 400:
                pos[idx, start:(end+1), pos_idx] = 1
            else:
                print("pos out", token.text, token.pos_, token_span, start, end)
        # ner
        for ent in doc.ents:
            token_span = (ent.start_char, ent.end_char)
            ner_idx = ner_dict[ent.label_]
            # print(ent.text, ent.label_, ner_idx)
            
            start, end = find_overlapping_span(token_span, span)
            if start >= 0 and end >= 0 and start < 400 and end < 400:
                ner[idx, start:(end+1), ner_idx] = 1
            else:
                print("ner out",ent.text, ent.label_, token_span, start, end)


