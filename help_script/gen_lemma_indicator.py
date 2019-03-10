import numpy as np
import spacy
import json
from collections import Counter

# load spacy model
nlp = spacy.load('en')

data_paths = ['train_tiny.npz']#, 'data/train.npz', 'data/test.npz', 'data/dev.npz']
output_file = 'data/frequent.json'


def compute_top_question_words(question_idxs, output_file, num_top = 20):
    word_count = Counter()
    for question in question_idxs:
        for word in question:
            if word != 0:
                word_count.update([word])
    mc = word_count.most_common(num_top)
    with open(output_file, 'w') as outfile:  
        json.dump([item[0] for item in mc].sorted(), outfile)



with open("data/idx2word.json") as f:
    idx2word = json.load(f)
    for data_path in data_paths:
        dataset = np.load(data_path)
        context_idxs = dataset['context_idxs']
        question_idxs = dataset['ques_idxs']

        compute_top_question_words(question_idxs, output_file)




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


