import torch


class Trainer:
    def __init__(self, model):
        self.model = model

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
        context_aware_sentence_vec, output, hidden_states, coverage_g, coverage_p = self.model.initialize(_input,
                                                                                                          predict)
        if teacher_forcing and not predict:
            prob_vector_tensor = torch.empty(len(target), output.size(0))
            prob_vector_tensor[0] = output
            for i, label in enumerate(target[:-1]):
                selected_sent = context_aware_sentence_vec[label]
                output, hidden_states, coverage_g, coverage_p = self.model(selected_sent, hidden_states,
                                                                           context_aware_sentence_vec,
                                                                           coverage_g, coverage_p)
                prob_vector_tensor[i + 1] = output
        else:
            prob_vector_tensor = torch.empty(summary_length, output.size(0))
            prob_vector_tensor[0] = output
            sentence_index = list(range(context_aware_sentence_vec.size(0)))
            for i in range(summary_length - 1):
                if predict:
                    max_index = max(sentence_index, key=lambda s: output[s])
                    selected_sent = context_aware_sentence_vec[max_index]
                    sentence_index.remove(max_index)
                else:
                    selected_sent = context_aware_sentence_vec[torch.argmax(output)]
                output, hidden_states, coverage_g, coverage_p = self.model(selected_sent, hidden_states,
                                                                           context_aware_sentence_vec,
                                                                           coverage_g, coverage_p)
                prob_vector_tensor[i + 1] = output
        return prob_vector_tensor

    def __call__(self, _input, summary_length=5, teacher_forcing=False, target=(), predict=False):
        return self.forward(_input, summary_length, teacher_forcing, target, predict)
