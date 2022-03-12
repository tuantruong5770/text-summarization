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

        self.conv_sent_encoder_vocab_size = conv_sent_encoder_vocab_size
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
        self.hyper_params = parameters
        self._sentence_encoder = SentenceEncoder(vocab_size=parameters.conv_sent_encoder_vocab_size,
                                                 emb_dim=parameters.word_embedding_dim,
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


    def forward(self, _input, summary_length=5, teacher_forcing=False, target=(), predict=False):
        """
        Output LongTensor(LongTensor()) of summary sentences probability vectors

        :param _input: IntTensor(IntTensor()) of torch.Size([num_sent, max_sent_len])
        :param summary_length: number of summary sentences to extract, len(target) if teacher_forcing is True
        :param teacher_forcing: force target input for the LSTMEncoder
        :param target: target label for teacher forcing
        :param predict: if True, _input must be a tensor word vectors
        :return: LongTensor(LongTensor()) of summary sentences probability vectors
        """
        sentence_vec = self._sentence_encoder(_input, predict=predict)
        context_aware_sentence_vec = self._lstm_encoder(sentence_vec)
        # Get the first sentence
        hidden_states = self._lstm_decoder.init_hidden_zero()
        coverage_g = self._lstm_decoder.init_coverage_glimpse(len(context_aware_sentence_vec))
        coverage_p = self._lstm_decoder.init_coverage_pointer(len(context_aware_sentence_vec))
        next_sent, hidden_states, coverage_g, coverage_p = self._lstm_decoder(None, hidden_states,
                                                                              context_aware_sentence_vec, coverage_g,
                                                                              coverage_p, start_token=True)
        if teacher_forcing:
            prob_vector_tensor = torch.empty(len(target), next_sent.size(0))
            prob_vector_tensor[0] = next_sent
            for i, label in enumerate(target[:-1]):
                selected_sent = context_aware_sentence_vec[label]
                next_sent, hidden_states, coverage_g, coverage_p = self._lstm_decoder(selected_sent, hidden_states,
                                                                                      context_aware_sentence_vec,
                                                                                      coverage_g, coverage_p)
                prob_vector_tensor[i + 1] = next_sent
        else:
            prob_vector_tensor = torch.empty(summary_length, next_sent.size(0))
            prob_vector_tensor[0] = next_sent
            for i in range(summary_length - 1):
                selected_sent = context_aware_sentence_vec[torch.argmax(next_sent)]
                next_sent, hidden_states, coverage_g, coverage_p = self._lstm_decoder(selected_sent, hidden_states,
                                                                                      context_aware_sentence_vec,
                                                                                      coverage_g, coverage_p)
                prob_vector_tensor[i + 1] = next_sent
        return prob_vector_tensor


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

