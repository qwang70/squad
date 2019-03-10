import numpy as np
import spacy
import json
from collections import Counter

# load spacy model
nlp = spacy.load('en')

# data_paths = ['train_tiny.npz', 'data/train.npz', 'data/test.npz', 'data/dev.npz']
data_paths = ['data/test.npz']
output_file = 'data/frequent.json'


def compute_top_question_words(question_idxs, output_file, num_top = 20):
    word_count = Counter()
    for question in question_idxs:
        for word in question:
            if word != 0:
                word_count.update([word])

    mc = word_count.most_common(num_top)

    outlist = [int(item[0]) for item in mc] + [0]
    outlist.sort()
    maxidx = outlist[-1]


    old_idx = outlist + [i for i in range(maxidx + 1) if i not in outlist]
    assert len(old_idx) == maxidx + 1
    outfile = open(output_file, "w")
    json.dump(old_idx, outfile)
    outfile.close()

    new_idx = [0 for i in range(maxidx + 1)]
    for idx in old_idx:
        new_idx[old_idx[idx]] = idx
    return new_idx




def convert_to_new(idxs, new_idx):
    l = len(new_idx)
    for row in idxs:
        for j, idx in enumerate(row):
            if idx < l:
                row[j] = new_idx[idx]







with open("data/idx2word.json") as f:
    idx2word = json.load(f)
    for data_path in data_paths:
        dataset = np.load(data_path)
        context_idxs = dataset['context_idxs']
        question_idxs = dataset['ques_idxs']

        new_idx = compute_top_question_words(question_idxs,  data_path.split('.')[0] + '_word_dict.json')
        print(question_idxs[2][:10])
        convert_to_new(context_idxs, new_idx)
        convert_to_new(question_idxs, new_idx)
        print(question_idxs[2][:10])
        






        # init EM mat
        em_indicators = np.zeros(context_idxs.shape)
        for idx, row in enumerate(context_idxs):
            if idx % 1000 == 0:
                print(idx)
            for j, word in enumerate(row):
                if word != 0:
                    if word in question_idxs[idx]:
                        em_indicators[idx, j] = 1
                    else:
                        em_indicators[idx, j] = -1



        # init lemma mat
        lemma_indicators = np.zeros(context_idxs.shape)
        for idx, row in enumerate(question_idxs):
            lemma_list = []
            # get all the lemma word in the question
            for word_id in row:
                if int(word_id) > 1:
                    word = idx2word[str(word_id)]
                    tokens = nlp.tokenizer(word)
                    for token in tokens:
                        lemma_list.append(token.lemma_.lower())

            # match the lemma word in the answer
            context = context_idxs[idx]
            for col_idx, word_id in enumerate(context):
                if int(word_id) != 0:
                    word = idx2word[str(word_id)]
                    tokens = nlp.tokenizer(word)
                    for token in tokens:
                        if token.lemma_ not in ''',.''' and token.lemma_.lower() in lemma_list:
                            lemma_indicators[idx, col_idx] = 1
                            break
                        else:
                            lemma_indicators[idx, col_idx] = -1
        outfile = '{}_features.npz'.format(data_path.split('.')[0])
        np.savez(outfile, context_idxs=dataset['context_idxs'],\
            context_char_idxs = dataset['context_char_idxs'],\
            ques_idxs = dataset['ques_idxs'],\
            ques_char_idxs=dataset['ques_char_idxs'],\
            y1s=dataset['y1s'],\
            y2s=dataset['y2s'],\
            ids=dataset['ids'],\
            em_indicators=em_indicators,\
            lemma_indicators=lemma_indicators)


