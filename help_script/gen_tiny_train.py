from numpy import load, savez_compressed

data = load('data/train_features.npz')
savez_compressed("train_tiny_features.npz", context_idxs=data['context_idxs'][:100],\
    context_char_idxs = data['context_char_idxs'][:100],\
    ques_idxs = data['ques_idxs'][:100],\
    ques_char_idxs=data['ques_char_idxs'][:100],\
    y1s=data['y1s'][:100],\
    y2s=data['y2s'][:100],\
    ids=data['ids'][:100],\
    em_indicators=data['em_indicators'][:100],\
    lemma_indicators=data['lemma_indicators'][:100],\
    c_posner=data['c_posner'][:100], \
    q_posner=data['q_posner'][:100]
    )
