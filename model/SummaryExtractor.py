from model.ConvSentEncoder import SentenceEncoder
from model.LSTMEncoder import LSTMEncoder
from model.LSTMDecoder import LSTMDecoder
import torch
import torch.nn as nn


class SummaryExtractorHyperParameters:
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
                 lstm_decoder_n_hidden=256,
                 lstm_decoder_n_layer=1,
                 lstm_decoder_context_vec_size=512,
                 lstm_decoder_pointer_net_n_hidden=512,
                 lstm_decoder_dropout=0.0):
        """
        Hyper-parameters for the sentence extraction model.
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

        self.lstm_decoder_n_hidden = lstm_decoder_n_hidden
        self.lstm_decoder_n_layer = lstm_decoder_n_layer
        self.lstm_decoder_context_vec_size = lstm_decoder_context_vec_size
        self.lstm_decoder_pointer_net_n_hidden = lstm_decoder_pointer_net_n_hidden
        self.lstm_decoder_dropout = lstm_decoder_dropout


class SummaryExtractor(nn.Module):
    def __init__(self):
        """
        Combine all the sub-models to create a full summary extraction model.
        Input list of word ids vectors -> output tensor of sentence extraction probability vectors.

        Word ID vector is vector of word ID according to the given word2vec model.
        List of word ids is list(IntTensor()), each element in list is a tensor of word id in that sentence.

        Output is a tensor of probability vectors, where each vector has size num_sentence.
        Each vector is a probability distribution of each sentence being the sentence chosen for summary.
        Size of output tensor will be the same as number of sentences in the summary.
        """
        super().__init__()
        self.hyper_params = SummaryExtractorHyperParameters()

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

        self._lstm_decoder = LSTMDecoder(encoder_dim=self.hyper_params.lstm_encoder_output_dim,
                                         lstm_dim=self.hyper_params.lstm_decoder_n_hidden,
                                         num_layer=self.hyper_params.lstm_decoder_n_layer,
                                         context_size=self.hyper_params.lstm_decoder_context_vec_size,
                                         pointer_size=self.hyper_params.lstm_decoder_pointer_net_n_hidden,
                                         dropout=self.hyper_params.lstm_decoder_dropout)


    def initialize(self, _input):
        sentence_vec = self._sentence_encoder(_input)
        context_aware_sentence_vec = self._lstm_encoder(sentence_vec)
        # Get the first sentence
        hidden_states = self._lstm_decoder.init_hidden()
        coverage_g = self._lstm_decoder.init_coverage_glimpse(len(context_aware_sentence_vec))
        coverage_p = self._lstm_decoder.init_coverage_pointer(len(context_aware_sentence_vec))
        output, hidden_states, coverage_g, coverage_p = self._lstm_decoder(None, hidden_states,
                                                                           context_aware_sentence_vec, coverage_g,
                                                                           coverage_p, start_token=True)
        return context_aware_sentence_vec, output, hidden_states, coverage_g, coverage_p


    def forward(self, selected_sent, hidden_states, context_aware_sentence_vec, coverage_g, coverage_p):
        """
        Output probability vector of the next selected sentence

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


    def set_embedding(self, word_embedding):
        self._sentence_encoder.set_embedding(word_embedding)


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

