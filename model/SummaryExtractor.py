from ConvSentEncoder import SentenceEncoder
from LSTMEncoder import LSTMEncoder
from LSTMDecoder import LSTMDecoder
from word2vec_helper import Word2VecHelper

import torch.nn as nn


class SummaryExtractorHyperParameters:
    def __init__(self,
                 word2vec_model,
                 conv_sent_encoder_n_hidden=20,
                 conv_sent_encoder_kernel=(3, 4, 5),
                 conv_sent_encoder_dropout=0.0,
                 conv_sent_encoder_training=True,
                 lstm_encoder_n_hidden=30,
                 lstm_encoder_n_layer=1,
                 lstm_encoder_dropout=0.0,
                 lstm_decoder_n_hidden=30,
                 lstm_decoder_n_layer=1,
                 lstm_decoder_context_vec_size=30,
                 lstm_decoder_pointer_net_n_hidden=30,
                 lstm_decoder_dropout=0.0):
        """
        Hyper-parameters for the sentence extraction model.

        :param word2vec_model: word2vec pretrained model name
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
        self._sentence_encoder = SentenceEncoder(emb_weights=Word2VecHelper.get_weights(parameters.word2vec_model),
                                                 emb_dim=parameters.word2vec_model.vector_size,
                                                 n_hidden=parameters.conv_sent_encoder_n_hidden,
                                                 kernel=parameters.conv_sent_encoder_kernel,
                                                 dropout=parameters.conv_sent_encoder_dropout,
                                                 training=parameters.conv_sent_encoder_training)

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


    def forward(self, _input):

