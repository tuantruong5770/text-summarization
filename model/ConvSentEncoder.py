from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from word2vec_helper import Word2VecHelper, PAD

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_hidden, kernel, dropout, training=True):
        """
        Convolutional neural network for sentence encoding.

        :param vocab_size: size of vocab in word2v
        :param emb_dim: dimension of word embeddings
        :param n_hidden: number of hidden nodes
        :param kernel: list of kernel sizes
        :param dropout: prob. out zeroing word_vector[i-th]
        """
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.training = training
        self._dropout = dropout
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i) for i in kernel])


    def forward(self, _input, predict=False):
        """
        Encode the input sentence
        :param _input: tensor of vectors of word ids in the embedding keymap
        :param predict: if True, _input must be a tensor word vectors
        :return: Tensor([1, n_hidden * len(KERNEL)])
        """
        # Transpose from (1, sent_len, emb_dim) to (1, emb_dim, sent_len)
        if predict:
            _input = _input.transpose(1, 2)
        else:
            _input = self._embedding(_input).transpose(1, 2)

        # Apply dropout on each word vector element based on dropout prob.
        # Only enable in training
        conv_in = F.dropout(_input, self._dropout, training=self.training)

        # Each output in conv_out is of dimension (1, n_hidden, sent_len - (kernel_size - 1))
        conv_out = [conv(conv_in) for conv in self._convs]

        # The output in conv_out is then max over dim2 to create a vector of dim (1, n_hidden)
        # The resulting vector is concatenated over dim1, creating vector of dim (1, n_hidden * 3)
        output = torch.cat([F.relu(res).max(dim=2)[0] for res in conv_out], dim=1)
        return output


    def set_embedding(self, embedding):
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


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

    model = Word2Vec(sentences=tokenized_sents, min_count=1, vector_size=emb_dim, window=wd)
    encoder = SentenceEncoder(torch.FloatTensor(model.wv.vectors), emb_dim, n_hidden, kernel_list, 0.0).to(device)
    sent_ids = Word2VecHelper.text_to_id(tokenized_sents, model, 0)
    # Test on first sentence of the text
    print(sent_ids[:100].reshape(100*202, 1).size())
    print(encoder.forward(sent_ids).size())
