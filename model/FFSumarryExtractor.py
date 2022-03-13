from model.ConvSentEncoder import SentenceEncoder
from model.LSTMEncoder import LSTMEncoder
from model.FFNetwork import FFNetwork
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FFSummaryExtractorHyperParameters:
    def __init__(self,
                 word_embedding_dim=128,
                 conv_sent_encoder_vocab_size=30000,
                 conv_sent_encoder_n_hidden=100,
                 conv_sent_encoder_kernel=(3, 4, 5),
                 conv_sent_encoder_dropout=0.0,
                 conv_sent_encoder_training=True,
                 lstm_encoder_n_hidden=256,
                 lstm_encoder_n_layer=1,
                 lstm_encoder_dropout=0.0,
                 ff_network_hidden_dim=512):
        """
        Hyper-parameters for the feed forward sentence extraction model.
        """
        self.word_embedding_dim = word_embedding_dim

        self.conv_sent_encoder_vocab_size = conv_sent_encoder_vocab_size + 2    # +2 for pad index and unknown index
        self.conv_sent_encoder_n_hidden = conv_sent_encoder_n_hidden
        self.conv_sent_encoder_output_dim = conv_sent_encoder_n_hidden * len(conv_sent_encoder_kernel)
        self.conv_sent_encoder_kernel = conv_sent_encoder_kernel
        self.conv_sent_encoder_dropout = conv_sent_encoder_dropout
        self.conv_sent_encoder_training = conv_sent_encoder_training

        self.lstm_encoder_n_hidden = lstm_encoder_n_hidden
        self.lstm_encoder_n_layer = lstm_encoder_n_layer
        self.lstm_encoder_output_dim = lstm_encoder_n_hidden * 2
        self.lstm_encoder_dropout = lstm_encoder_dropout

        self.ff_network_hidden_dim = ff_network_hidden_dim


class FFSummaryExtractor(nn.Module):
    def __init__(self):
        """
        Combine all the sub-models to create a full feedforward summary extraction model.
        Input list of word ids vectors -> output tensor of sentence extraction probability vectors.

        Word ID vector is vector of word ID according to the given word2vec model.
        List of word ids is list(IntTensor()), each element in list is a tensor of word id in that sentence.

        Output is a probability vector tensor with fixed size of ff_network_hidden_dim, representing probability
        of sentences being extracted.
        """
        super().__init__()
        self.hyper_params = FFSummaryExtractorHyperParameters()

        self._sentence_encoder = SentenceEncoder(vocab_size=self.hyper_params.conv_sent_encoder_vocab_size,
                                                 emb_dim=self.hyper_params.word_embedding_dim,
                                                 n_hidden=self.hyper_params.conv_sent_encoder_n_hidden,
                                                 kernel=self.hyper_params.conv_sent_encoder_kernel,
                                                 dropout=self.hyper_params.conv_sent_encoder_dropout,
                                                 training=self.hyper_params.conv_sent_encoder_training)

        self._lstm_encoder = LSTMEncoder(input_dim=self.hyper_params.conv_sent_encoder_output_dim,
                                         n_hidden=self.hyper_params.lstm_encoder_n_hidden,
                                         n_layer=self.hyper_params.lstm_encoder_n_layer,
                                         dropout=self.hyper_params.lstm_encoder_dropout)

        self._ff_network = FFNetwork(encoder_dim=self.hyper_params.lstm_encoder_output_dim,
                                     hidden_dim=self.hyper_params.ff_network_hidden_dim)


    def forward(self, _input):
        """
        Output probability vector of the selected sentences.

        :param _input: word ids 2-d tensor
        :return: output: extract probability vector of each sentence, hidden_states, coverage_g, coverage_p
        """
        document_size_lim = self.hyper_params.ff_network_hidden_dim
        sentence_vec = self._sentence_encoder(_input)
        context_aware_sentence_vec = self._lstm_encoder(sentence_vec)
        document_size = context_aware_sentence_vec.size(0)

        # Fixing input size for feedforward network
        if document_size > document_size_lim:
            context_aware_sentence_vec = context_aware_sentence_vec[:document_size_lim]
        else:
            context_aware_sentence_vec = torch.cat((context_aware_sentence_vec,
                                                    torch.zeros(document_size_lim - document_size,
                                                                context_aware_sentence_vec.size(1)).to(device)), dim=0)
        # Output is size ff_network_hidden_dim
        output = self._ff_network(context_aware_sentence_vec)
        return output


    def set_embedding(self, word_embedding):
        self._sentence_encoder.set_embedding(word_embedding)


if __name__ == '__main__':
    pass
