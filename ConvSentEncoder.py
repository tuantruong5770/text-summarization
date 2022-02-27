
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    def __init__(self, n_hidden: int, emb_dim: int, embedding_model):
        '''
        The initiator of sentence encoder is to prepare tokenized sentences and
        Word2Vec model.
        arguments:
        n_hidden: The output size of convolutionary network window.
        embedding_model: the word2vec model used for
        '''
        super().__init__()

        # word embedding argument
        self.embedding_model = embedding_model

        # sentence encodding argument
        self.n_hidden = n_hidden
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])

    def forward(self, _input):
        '''encode the input sentence'''
        sent_words = [w.lower() for w in _input]
        sent_words_vec = [self.embedding_model.wv[w] for w in sent_words]
        sent_length = len(sent_words_vec)
        emb_dim = len(sent_words_vec[0])
        conv_in = torch.tensor(sent_words_vec).reshape(1, emb_dim, sent_length)
        temp = []
        for conv in self._convs:
            temp.append(conv(conv_in))
        output = torch.cat([F.relu(res).max(dim=2)[0]
                            for res in temp], dim=1).reshape((3 * self.n_hidden))
        return output

def main():
    '''sample input'''
    emb_dim = 10
    n_hidden = 20
    wd = 10
    sg = 0

    text_path = "alice.txt"
    with open(text_path) as f:
        text = f.read()

    # break article into sentences
    article = text.replace('\n', ' ')
    sent_list = sent_tokenize(article)

    # break sentences into words
    tokenized_sents = []
    for sent in sent_list:
        temp = []
        for w in word_tokenize(sent):
            temp.append(w.lower())
        tokenized_sents.append(temp)

    # encode words using word2vec
    encoded_words = Word2Vec(tokenized_sents, min_count=1,
                             vector_size=emb_dim, window=wd, sg=sg)

    s = SentenceEncoder(n_hidden, encoded_words)
    print(s.forward(tokenized_sents[0]))


