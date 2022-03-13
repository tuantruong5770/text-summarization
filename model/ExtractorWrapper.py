from word2vec_helper import Word2VecHelper
import torch
import torch.nn as nn


class ExtractorWrapper():
    def __init__(self, model, word_to_index):
        """
        Wrapper of SummaryExtractor with some additional functionality

        :param model: SummaryExtractor model
        :param word_to_index: word to index dictionary
        """
        self.model = model
        self.word_to_index = word_to_index
        self.hyper_params = model.hyper_params


    def forward(self, _input, summary_length=5, teacher_forcing=False, target=()):
        """
        Output LongTensor(LongTensor()) of summary sentences probability vectors

        :param _input: IntTensor(IntTensor()) of torch.Size([num_sent, max_sent_len])
        :param summary_length: number of summary sentences to extract, len(target) if teacher_forcing is True
        :param teacher_forcing: force target input for the LSTMEncoder
        :param target: target label for teacher forcing
        :return: LongTensor(LongTensor()) of summary sentences probability vectors
        """
        # Convert text to id tensor
        _input = Word2VecHelper.text_to_id(_input, self.word_to_index)
        context_aware_sentence_vec, output, hidden_states, coverage_g, coverage_p = self.model.initialize(_input)

        if teacher_forcing:
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
            for i in range(summary_length - 1):
                selected_sent = context_aware_sentence_vec[torch.argmax(output)]
                output, hidden_states, coverage_g, coverage_p = self.model(selected_sent, hidden_states,
                                                                           context_aware_sentence_vec,
                                                                           coverage_g, coverage_p)
                prob_vector_tensor[i + 1] = output
        return prob_vector_tensor


    def predict(self, _input, summary_length=5, no_duplicate=False):
        """
        Return extraction probability vectors

        :param _input: document text
        :param summary_length: number of sentences to extract (must be less than number of sentences in text(
        :param no_duplicate: if True then the model pick the second most probable sentence if the most probable sentence
        has been chosen
        :return: prob_vector_tensor, extracted indices
        """
        with torch.no_grad():
            # Convert text to id tensor
            _input = Word2VecHelper.text_to_id(_input, self.word_to_index)
            context_aware_sentence_vec, output, hidden_states, coverage_g, coverage_p = self.model.initialize(_input)

            sentence_index = list(range(context_aware_sentence_vec.size(0)))
            prob_vector_tensor = torch.empty(summary_length, output.size(0))
            prob_vector_tensor[0] = output

            # Get index based on no_duplicate flag
            def get_index(prob_vec):
                if no_duplicate:
                    max_index = max(sentence_index, key=lambda s: prob_vec[s])
                    sentence_index.remove(max_index)
                else:
                    max_index = torch.argmax(prob_vec)
                return max_index

            index = get_index(output)
            extracted_index = [index]
            selected_sent = context_aware_sentence_vec[index]

            for i in range(summary_length - 1):
                output, hidden_states, coverage_g, coverage_p = self.model(selected_sent, hidden_states,
                                                                           context_aware_sentence_vec, coverage_g,
                                                                           coverage_p)
                index = get_index(output)
                selected_sent = context_aware_sentence_vec[index]
                prob_vector_tensor[i + 1] = output
                extracted_index.append(index)

        return prob_vector_tensor, extracted_index


    def comprehensive_test(self, text, summ, label, data_index, print_text=False, outfile=None):
        """
        Comprehensive test of the model with prediction with and without force no dup
        Output either print or to output file if specified

        :param text: Input text
        :param summ: Ground truth summary (abstractive)
        :param label: "true" label
        :param data_index: index of data point for recording purpose
        :param print_text: if true also print original text
        :param outfile: (optional) output txt file
        :return:
        """
        prob_vector, ext_ind = self.predict(text, len(label), no_duplicate=False)
        prob_vector_no_dup, ext_ind_no_dup = self.predict(text, len(label), no_duplicate=True)

        lines = [f'DATA INDEX: {data_index}\n']

        if print_text:
            lines.append('~*~*~*~*~*~*~*~*~*~*~*~*~*~*~ TEXT *~*~~*~*~*~*~*~*~*~*~*~*~*~*~\n')
            for sent in text:
                lines.append(' '.join(sent) + '\n')

        lines.append('~*~*~*~*~*~*~*~*~*~*~*~*~ GROUND TRUTH ~*~*~*~*~*~*~*~*~*~*~*~*~\n')
        for sent in summ:
            lines.append(' '.join(sent) + '\n')

        lines.append('*~*~*~*~*~*~*~*~*~*~*~*~ LABELED SUMMARY ~*~*~*~*~*~*~*~*~*~*~*~\n')
        for label_index in label:
            lines.append(' '.join(text[label_index]) + '\n')

        lines.append('~*~*~*~*~*~*~*~*~*~*~*~ GENERATED SUMMARY ~*~*~*~*~*~*~*~*~*~*~*\n')
        for ind in ext_ind:
            lines.append(' '.join(text[ind]) + '\n')

        lines.append('~*~*~*~*~*~*~*~*~* GENERATED SUMMARY (NO DUP) ~*~*~*~*~*~*~*~*~*\n')
        for ind in ext_ind_no_dup:
            lines.append(' '.join(text[ind]) + '\n')

        if outfile:
            outfile.writelines(lines)
        else:
            for line in lines:
                print(line, end='')


    def __call__(self, _input, summary_length=5, teacher_forcing=False, target=()):
        return self.forward(_input, summary_length, teacher_forcing, target)


if __name__ == '__main__':
    pass
