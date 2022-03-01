from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from word2vec_helper import Word2VecHelper

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    def __init__(self, emb_weights, emb_dim, n_hidden, kernel, dropout, training=True):
        """
        Convolutional neural network for sentence encoding.

        :param emb_weights: pretrained weights of the Word2Vec model
        :param emb_dim: dimension of word embeddings
        :param n_hidden: number of hidden nodes
        :param kernel: list of kernel sizes
        :param dropout: prob. out zeroing word_vector[i-th]
        """
        super().__init__()
        self._embedding = nn.Embedding.from_pretrained(emb_weights)
        self.training = training
        self._dropout = dropout
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i) for i in kernel])


    def forward(self, _input):
        """
        Encode the input sentence
        :param _input: List of words in a sentence
        :return: Tensor([1, n_hidden * len(KERNEL)])
        """
        # Pad dimension and transpose from (1, sent_len, emb_dim) to (1, emb_dim, sent_len)
        emb_input = self._embedding(_input).unsqueeze(0).transpose(1, 2)

        # Apply dropout on each word vector element based on dropout prob.
        # Only enable in training
        conv_in = F.dropout(emb_input, self._dropout, training=self.training)

        # Each output in conv_out is of dimension (1, n_hidden, sent_len - (kernel_size - 1))
        conv_out = [conv(conv_in) for conv in self._convs]

        # The output in conv_out is then max over dim2 to create a vector of dim (1, n_hidden)
        # The resulting vector is concatenated over dim1, creating vector of dim (1, n_hidden * 3)
        output = torch.cat([F.relu(res).max(dim=2)[0] for res in conv_out], dim=1)
        return output


if __name__ == "__main__":
    emb_dim = 10
    n_hidden = 20
    wd = 10
    sg = 0
    kernel_list = [3, 4, 5]

    with open("alice.txt") as f:
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

    model = Word2Vec(sentences=tokenized_sents, min_count=1, vector_size=emb_dim, window=wd, sg=sg)
    encoder = SentenceEncoder(torch.FloatTensor(model.wv.vectors), emb_dim, n_hidden, kernel_list, 0.0)
    sent_ids = Word2VecHelper.text_to_id(tokenized_sents, model)
    print(model.wv.key_to_index)
    # Test on first sentence of the text
    print(encoder.forward(sent_ids[0]))
