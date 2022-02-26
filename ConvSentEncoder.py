
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder:
    def __init__(self, article: str, emb_dim: int, n_hidden: int, wd: int, sg=0):
        '''
        The initiator of sentence encoder is to prepare tokenized sentences and
        Word2Vec model.
        arguments:
        article: The string of the article that needs sentence encoding.
        emb_dim: The size of a word vector.
        n_hidden: The output size of convolutionary network window.
        wd: The window size of the Word2Vec model.
        sg: If sg=0, Word2Vec is using cbow model. Otherwise, Word2Vec is using skip gram model.
        '''
        self.emb_dim = emb_dim
        self.n_hidden = n_hidden
        # need adjust
        self._convs_wd_size = range(3, 6)
        # nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i) for i in range(3, 6)])

        # break article into sentences
        article = article.replace('\n', ' ')
        self.sent_list = sent_tokenize(article)

        # break sentences into words
        self.tokenized_sents = []
        for sent in self.sent_list:
            temp = []
            for w in word_tokenize(sent):
                temp.append(w.lower())
            self.tokenized_sents.append(temp)

        # encode words using word2vec
        self.encoded_words = Word2Vec(self.tokenized_sents, min_count=1, vector_size=emb_dim, window=wd, sg=sg)

        # make a list of sentence represented by word vector
        self.tokenized_sents_wv = [[self.encoded_words.wv[w] for w in sent] for sent in self.tokenized_sents]

    def encode(self):
        '''encode each sentence in the article'''
        self.encoded_sents = []
        for sent in self.tokenized_sents_wv:
            conv_in = torch.tensor(sent).reshape(1, self.emb_dim, len(sent))
            temp = []
            for size in self._convs_wd_size:
                if len(sent) < size:
                    size = len(sent)
                module = nn.Conv1d(self.emb_dim, self.n_hidden, size)
                temp.append(module(conv_in))
            output = torch.cat([F.relu(res).max(dim=2)[0]
                                for res in temp], dim=1)
            self.encoded_sents.append(output)

        self.encoded_sents = torch.cat(self.encoded_sents, dim=0)
        return self.encoded_sents

    def get_encoded_sents(self):
        return self.encoded_sents


def main():
    '''sample input'''
    text = "string of an article"
    # The size of a word vector for word embedding
    emb_dim = 10
    # The output size for each CNN window
    # The sentence vector will be 3*n_hidden
    n_hidden = 20
    # The window size to do the word embedding
    wd = 10
    # Indicating to use cbow mode while 0, use skip gram mode while 1
    sg = 0

    # make a sentence encoder and encode
    s = SentenceEncoder(text, emb_dim, n_hidden, wd, sg)
    encoded_sentences = s.encode()
    print(encoded_sentences[0])


