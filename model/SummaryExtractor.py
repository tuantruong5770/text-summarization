from model.ConvSentEncoder import SentenceEncoder
from model.LSTMEncoder import LSTMEncoder
from model.LSTMDecoder import LSTMDecoder
from word2vec_helper import Word2VecHelper
from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize

import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SummaryExtractorHyperParameters:
    def __init__(self,
                 word2vec_model,
                 conv_sent_encoder_n_hidden=100,
                 conv_sent_encoder_kernel=(3, 4, 5),
                 conv_sent_encoder_dropout=0.0,
                 conv_sent_encoder_training=True,
                 lstm_encoder_n_hidden=256,
                 lstm_encoder_n_layer=1,
                 lstm_encoder_dropout=0.0,
                 lstm_decoder_n_hidden=256,
                 lstm_decoder_n_layer=1,
                 lstm_decoder_context_vec_size=512,
                 lstm_decoder_pointer_net_n_hidden=512,
                 lstm_decoder_dropout=0.0):
        """
        Hyper-parameters for the sentence extraction model.

        :param word2vec_model: word2vec model
        :param conv_sent_encoder_kernel: list of kernel size for convolutional neural network
        :param conv_sent_encoder_dropout: dropout probability for convolutional sentence encoder input
        :param conv_sent_encoder_training: True if the model is in training phase else False
        :param lstm_encoder_n_layer: number of layers of the LSTMEncoder (multiplied by 2 for bidirectional)
        :param lstm_encoder_dropout: dropout probability for LSTMEncoder
        :param lstm_decoder_n_hidden: number of hidden feature for LSTMDecoder
        :param lstm_decoder_n_layer: number of layers for LSTMDecoder
        :param lstm_decoder_context_vec_size: size of context vector e_t (output of the Glimpse operation)
        :param lstm_decoder_pointer_net_n_hidden: number of hidden feature for the Pointer Network
        :param lstm_decoder_dropout: dropout probability for LSTMDecoder
        """
        self.word2vec_model = word2vec_model

        self.conv_sent_encoder_n_hidden = conv_sent_encoder_n_hidden
        self.conv_sent_encoder_output_dim = conv_sent_encoder_n_hidden * len(conv_sent_encoder_kernel)
        self.conv_sent_encoder_kernel = conv_sent_encoder_kernel
        self.conv_sent_encoder_dropout = conv_sent_encoder_dropout
        self.conv_sent_encoder_training = conv_sent_encoder_training

        self.lstm_encoder_n_hidden = lstm_encoder_n_hidden
        self.lstm_encoder_n_layer = lstm_encoder_n_layer
        self.lstm_encoder_output_dim = lstm_encoder_n_hidden * 2
        self.lstm_encoder_dropout = lstm_encoder_dropout

        self.lstm_decoder_n_hidden = lstm_decoder_n_hidden
        self.lstm_decoder_n_layer = lstm_decoder_n_layer
        self.lstm_decoder_context_vec_size = lstm_decoder_context_vec_size
        self.lstm_decoder_pointer_net_n_hidden = lstm_decoder_pointer_net_n_hidden
        self.lstm_decoder_dropout = lstm_decoder_dropout

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SummaryExtractor(nn.Module):
    def __init__(self, parameters: SummaryExtractorHyperParameters):
        """
        Combine all the sub-models to create a full summary extraction model.
        Input list of word ids vectors -> output tensor of sentence extraction probability vectors.

        Word ID vector is vector of word ID according to the given word2vec model.
        List of word ids is list(IntTensor()), each element in list is a tensor of word id in that sentence.

        Output is a tensor of probability vectors, where each vector has size num_sentence.
        Each vector is a probability distribution of each sentence being the sentence chosen for summary.
        Size of output tensor will be the same as number of sentences in the summary.

        :param parameters: SentenceExtractionHyperParameters object for parameters filling
        """
        super().__init__()
        self.hyper_params = parameters
        self._sentence_encoder = SentenceEncoder(vocab_size=len(Word2VecHelper.get_weights(parameters.word2vec_model)),
                                                 emb_dim=parameters.word2vec_model.vector_size,
                                                 n_hidden=parameters.conv_sent_encoder_n_hidden,
                                                 kernel=parameters.conv_sent_encoder_kernel,
                                                 dropout=parameters.conv_sent_encoder_dropout,
                                                 training=parameters.conv_sent_encoder_training)
        self._sentence_encoder.set_embedding(Word2VecHelper.get_weights(parameters.word2vec_model))

        self._lstm_encoder = LSTMEncoder(input_dim=parameters.conv_sent_encoder_output_dim,
                                         n_hidden=parameters.lstm_encoder_n_hidden,
                                         n_layer=parameters.lstm_encoder_n_layer,
                                         dropout=parameters.lstm_encoder_dropout)

        self._lstm_decoder = LSTMDecoder(encoder_dim=parameters.lstm_encoder_output_dim,
                                         lstm_dim=parameters.lstm_decoder_n_hidden,
                                         num_layer=parameters.lstm_decoder_n_layer,
                                         context_size=parameters.lstm_decoder_context_vec_size,
                                         pointer_size=parameters.lstm_decoder_pointer_net_n_hidden,
                                         dropout=parameters.lstm_decoder_dropout)

    def initialize(self, _input, predict=False):
        sentence_vec = self._sentence_encoder(_input, predict=predict)
        context_aware_sentence_vec = self._lstm_encoder(sentence_vec)
        # Get the first sentence
        hidden_states = self._lstm_decoder.init_hidden()
        coverage_g = self._lstm_decoder.init_coverage_glimpse(len(context_aware_sentence_vec))
        coverage_p = self._lstm_decoder.init_coverage_pointer(len(context_aware_sentence_vec))
        next_sent, hidden_states, coverage_g, coverage_p = self._lstm_decoder(None, hidden_states,
                                                                              context_aware_sentence_vec, coverage_g,
                                                                              coverage_p, start_token=True)
        return context_aware_sentence_vec, next_sent, hidden_states, coverage_g, coverage_p

    def forward(self, selected_sent, hidden_states, context_aware_sentence_vec, coverage_g, coverage_p):
        """
        :param selected_sent: context aware sentence representation torch.size(encoder_dim) extracted in prev time-step
        :param hidden_states: previous time-step hidden states of LSTMDecoder
        :param context_aware_sentence_vec: outputs of LSTMDecoder
        :param coverage_g: coverage vector for glimpse network in previous time-step
        :param coverage_p: coverage vector for pointer network in previous time-step
        :return: output: extract probability vector of each sentence, hidden_states, coverage_g, coverage_p
        """
        output, hidden_states, coverage_g, coverage_p = self._lstm_decoder(selected_sent, hidden_states,
                                                                           context_aware_sentence_vec,
                                                                           coverage_g, coverage_p)
        return output, hidden_states, coverage_g, coverage_p


if __name__ == '__main__':
    # with open("alice.txt") as f:
    #     text = f.read()
    #
    # # break article into sentences
    # article = text.replace('\n', ' ')
    # sent_list = sent_tokenize(article)
    #
    # # break sentences into words
    # tokenized_sents = []
    # for sent in sent_list:
    #     temp = []
    #     for w in word_tokenize(sent):
    #         temp.append(w.lower())
    #     tokenized_sents.append(temp)
    #
    # word2v = Word2Vec(sentences=tokenized_sents, vector_size=20, min_count=1)
    # model = SummaryExtractor(SummaryExtractorHyperParameters(word2v)).to(device)
    # sent_ids = Word2VecHelper.text_to_id(tokenized_sents, word2v, 0)
    # print(model(sent_ids, teacher_forcing=True, target=[1, 4, 6]))
    # print(model(sent_ids, summary_length=5))

    pass

    # a = torch.rand(5, 10)
    # b = torch.ones(10)
    # print(a)
    # a[0] = b
    # print(a)

