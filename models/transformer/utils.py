import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import json


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out
def get_embedding_matrix(tokenize_level):
    if os.path.isfile('./embedding_matrix_pre_build_%s.npy' % tokenize_level) == False:
        print('Build new embedding matrix and save to embedding_matrix_pre_build_%s.npy! \n ...' % tokenize_level)
        if tokenize_level == 'syllable':
            f =  open('./word2vec_vi_%ss_300dims.txt' % tokenize_level, encoding='utf-8')
        elif tokenize_level == 'word':
            f =  open('./word2vec_vi_%ss_300dims.txt' % tokenize_level, encoding='utf-8')
        word_dict = []
        embeddings_index = {}
        embedding_dim = 300
        max_feature = len(embeddings_index) + 2
        for line in f:
            values = line.split(' ')
            word = values[0]
            word_dict.append(word)
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        
        if tokenize_level == 'syllable':
            with open('wtoi_%s.json' % tokenize_level, 'r') as f:
                wtoi = json.load(f)
            num_words = len(wtoi)
        elif tokenize_level == 'word':
            with open('wtoi_%s.json' % tokenize_level, 'r') as f:
                wtoi = json.load(f)
            num_words = len(wtoi)

        embedding_dim = 300
        embedding_matrix = np.zeros((num_words, embedding_dim))

        for word,i in wtoi.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.randn(embedding_dim)
        np.save('embedding_matrix_pre_build_%s.npy' % tokenize_level, embedding_matrix) # save
    else:
        print('Load embedding matrix %s level pre build from file!' % tokenize_level)
        embedding_matrix = np.load('./embedding_matrix_pre_build_%s.npy' % tokenize_level)

    embedding_matrix = torch.tensor(embedding_matrix).type(torch.FloatTensor)
    expand_dim = nn.Linear(in_features=300,out_features=512)
    embedding_matrix = expand_dim(embedding_matrix)
    print('Embedding data loaded\n')
    return embedding_matrix
    


def get_pretrained_encoding(path):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(path)
    weights = torch.FloatTensor(model.vectors)
    expand_dim = nn.Linear(in_features=300,out_features=512)
    weights = expand_dim(weights)
    return weights

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out