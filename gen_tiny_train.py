from numpy import load, savez

data = load('data/train.npz')
savez("train_tiny.npz", context_idxs=data['context_idxs'][:100],\
    context_char_idxs = data['context_char_idxs'][:100],\
    ques_idxs = data['ques_idxs'][:100],\
    ques_char_idxs=data['ques_char_idxs'][:100],\
    y1s=data['y1s'][:100],\
    y2s=data['y2s'][:100],\
    ids=data['ids'][:100])
