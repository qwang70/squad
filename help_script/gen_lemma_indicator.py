import numpy as np
import spacy
import json
from collections import Counter
import zipfile
import io

def saveCompressed(fh, **namedict):
     with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                          allowZip64=True) as zf:
         for k, v in namedict.items():
             with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                 np.lib.npyio.format.write_array(buf,
                                                 np.asanyarray(v),
                                                 allow_pickle=False)
# load spacy model
nlp = spacy.load('en')

# data_paths = ['train_tiny.npz']#, 'data/train.npz', 'data/test.npz', 'data/dev.npz']
data_paths = ['data/dev.npz', 'data/train.npz', 'data/test.npz']
output_file = 'data/frequent.json'


# build dictionary
poses = ['POS', 'PUNCT', 'SYM', 'ADJ', 'CCONJ', 'NUM', 'DET', 'ADV', 'ADP', 'X', 'VERB', 'NOUN', 'PROPN', 'PART', 'INTJ', 'PRON', '']
pos_dict = {}
for i, pos in enumerate(poses):
    pos_dict[pos] = i

ners = [ 'PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT', 'LAW', 'WORK_OF_ART','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
ner_dict = {}
for i, pos in enumerate(ners):
    ner_dict[pos] = i

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

def get_pos_ner(eval_file, ids_set, limit, limit2):
    with open(eval_file) as f:
        data = json.load(f)
        c_pos = np.zeros((len(ids_set), limit, len(pos_dict)))
        c_ner = np.zeros((len(ids_set), limit, len(ner_dict)))
        q_pos = np.zeros((len(ids_set), limit2, len(pos_dict)))
        q_ner = np.zeros((len(ids_set), limit2, len(ner_dict)))

        offset = 1
        for num in data:
            if int(num) not in ids_set:
                print("not in dict")
                offset += 1
                continue
            idx = int(num) - offset
            if idx % 1000 == 0:
                print(idx)
            context = data[num]['context']
            span = data[num]['spans']
            question = data[num]['question']
            ques_span = data[num]['ques_spans']
            # context 
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
                if start >= 0 and end >= 0 and start < limit and end < limit:
                    c_pos[idx, start:(end+1), pos_idx] = 1
                else:
                    print("pos out", num, token.text, token.pos_, token_span, start, end)
            # ner
            for ent in doc.ents:
                if ent.label_ not in ners:
                    continue
                token_span = (ent.start_char, ent.end_char)
                ner_idx = ner_dict[ent.label_]
                # print(ent.text, ent.label_, ner_idx)
                
                start, end = find_overlapping_span(token_span, span)
                if start >= 0 and end >= 0 and start < limit and end < limit:
                    c_ner[idx, start:(end+1), ner_idx] = 1
                else:
                    print("ner out",num, ent.text, ent.label_, token_span, start, end)

            # question
            doc = nlp(question)
            # pos
            for token in doc:
                if token.pos_ not in poses:
                    continue
                pos_idx = pos_dict[token.pos_]
                # print(token.text, token.pos_, token.pos)
                token_span = (token.idx, token.idx + len(token.text))
                # print(context)
                # print("token span", token_span)
                start, end = find_overlapping_span(token_span, ques_span)
                # print(start, end)
                if start >= 0 and end >= 0 and start < limit2 and end < limit2:
                    q_pos[idx, start:(end+1), pos_idx] = 1
                else:
                    print("q pos out", num, token.text, token.pos_, token_span, start, end)
            # ner
            for ent in doc.ents:
                if ent.label_ not in ners:
                    continue
                token_span = (ent.start_char, ent.end_char)
                ner_idx = ner_dict[ent.label_]
                # print(ent.text, ent.label_, ner_idx)
                
                start, end = find_overlapping_span(token_span, ques_span)
                if start >= 0 and end >= 0 and start < limit2 and end < limit2:
                    q_ner[idx, start:(end+1), ner_idx] = 1
                else:
                    print("ner out",num, ent.text, ent.label_, token_span, start, end)
    return np.concatenate((c_pos, c_ner), axis=2),  np.concatenate((q_pos, q_ner), axis=2)


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

#        new_idx = compute_top_question_words(question_idxs,  data_path.split('.')[0] + '_word_dict.json')
#        convert_to_new(context_idxs, new_idx)
#        convert_to_new(question_idxs, new_idx)

        # pos, ner
        print("pos, ner...")
        eval_file = '{}_eval.json'.format(data_path.split('.')[0])
        context_posner, question_posner = get_pos_ner(eval_file, {*dataset['ids']}, context_idxs.shape[1], question_idxs.shape[1])

        # init EM mat
        print("EM init...")
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
        print("lemma init...")
        lemma_indicators = np.zeros(context_idxs.shape)
        pos_num = np.zeros(context_idxs.shape)
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
                        pos_num[idx, col_idx] = pos_dict[token.pos_]
                        if token.lemma_ not in ''',.''' and token.lemma_.lower() in lemma_list:
                            lemma_indicators[idx, col_idx] = 1
                            break
                        else:
                            lemma_indicators[idx, col_idx] = -1
        # features = np.concatenate([em_indicators, posner], axis=2)
        outfile = '{}_features'.format(data_path.split('.')[0])
        print(outfile, "saving...")
        np.savez_compressed(outfile, context_idxs=dataset['context_idxs'],\
            context_char_idxs = dataset['context_char_idxs'],\
            ques_idxs = dataset['ques_idxs'],\
            ques_char_idxs=dataset['ques_char_idxs'],\
            y1s=dataset['y1s'],\
            y2s=dataset['y2s'],\
            ids=dataset['ids'],\
            #features = features
            em_indicators=em_indicators,\
            lemma_indicators=lemma_indicators,\
            c_posner=context_posner, \
            q_posner=question_posner
            )


