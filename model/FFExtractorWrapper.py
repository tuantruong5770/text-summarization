from word2vec_helper import Word2VecHelper
import torch
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FFExtractorWrapper:
    def __init__(self, model, word_to_index):
        """
        Wrapper of FFSummaryExtractor with some additional functionality

        :param model: FFSummaryExtractor model
        :param word_to_index: word to index dictionary
        """
        self.model = model
        self.word_to_index = word_to_index
        self.hyper_params = model.hyper_params


    def forward(self, _input):
        """
        Output LongTensor() of summary sentences probability vectors

        :param _input: processed text input
        :return: LongTensor(LongTensor()) of summary sentences probability vectors
        """
        # Convert text to id tensor
        _input = Word2VecHelper.text_to_id(_input, self.word_to_index)
        prob_vector = self.model(_input)
        return prob_vector


    def predict(self, _input, summary_length=5):
        """
        Wrapper Required
        Signature must be [predict(_input, summary_length, **kwargs) -> tensor, extracted_index]

        Return extraction probability vectors with extracted index

        :param _input: document text
        :param summary_length: number of sentences to extract (must be less than number of sentences in text)
        :return: prob_vector_tensor, extracted indices
        """
        with torch.no_grad():
            # Convert text to id tensor
            document_size = len(_input)
            _input = Word2VecHelper.text_to_id(_input, self.word_to_index)
            prob_vector = self.model(_input)
            extracted_index = torch.argsort(prob_vector[:document_size], dim=0)[:summary_length]
        return prob_vector, extracted_index


    def calculate_loss(self, criterion, output, label):
        """
        Wrapper Required
        Signature must be [calculate_loss(criterion, output, label, **kwargs) -> loss]

        Calculate the loss for given criterion, output, label

        :param criterion: loss function
        :param output: output of the model
        :param label: summary label
        :return: loss tensor
        """
        # Creating label tensor
        labels = torch.zeros(output.size()).to(device)
        normalized_prob = 1 / len(label)
        for label_index in label:
            labels[label_index] = normalized_prob
        loss = criterion(output.unsqueeze(0), labels.unsqueeze(0))
        return loss


    def calculate_val_loss(self, criterion, val_set, val_loader):
        """
        Wrapper Required
        Signature must be [calculate_val_loss(criterion, val_set, val_loader, **kwargs) -> loss]

        Calculate the validation loss for given criterion, validation set and loader
        No grad calculation

        :param criterion: loss function
        :param val_set: validation set
        :param val_loader: validation loader (DataLoader)
        :return: loss tensor
        """
        with torch.no_grad():
            num_data = len(val_loader)
            tot_loss = 0
            for index in val_loader:
                text, summ, label, score = val_set[index[0]]
                output = self.forward(text)
                tot_loss += self.calculate_loss(criterion, output, label)
        val_loss = tot_loss / num_data
        return val_loss


    def comprehensive_test(self, text, summ, label, data_index, print_text=False, outfile=None):
        """
        Comprehensive test of the model with prediction
        Output either print or to output file if specified

        :param text: Input text
        :param summ: Ground truth summary (abstractive)
        :param label: "true" label
        :param data_index: index of data point for recording purpose
        :param print_text: if true also print original text
        :param outfile: (optional) output txt file
        :return:
        """
        prob_vector, ext_ind = self.predict(text, len(label))

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

        if outfile:
            outfile.writelines(lines)
        else:
            for line in lines:
                print(line, end='')


    def save_model(self, dataset_name, training_time, train_type, train_params):
        """
        Wrapper Required
        Signature must be [save_model(dataset_name, training_time, train_type, train_params, **kwargs)]

        Save the model with .txt for details

        :param dataset_name: name of dataset used in training
        :param training_time: total time spent training
        :param train_type: type of training used
        :param train_params: training hyper parameters
        :return:
        """
        model = self.model
        num_epochs = train_params.epoch
        teacher_forcing_prob = train_params.teacher_forcing_prob
        num_training = train_params.num_training
        batch_size = train_params.batch_size
        learning_rate = train_params.learning_rate

        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_name = f'{dt_string}_{dataset_name}'
        save_loc = f'./pretrained/{save_name}'
        torch.save(model.state_dict(), save_loc + '.pt')
        with open(save_loc + '.txt', 'w') as f:
            parameters = model.hyper_params
            lines = [
                f'MODEL NAME: {save_name}\n',
                f'MODEL TYPE: FFSummaryExtractor\n',
                f'TRAIN TYPE: {train_type}\n',
                f'\n',
                f'MODEL PARAMETERS:\n',
                f'\n',
                f'word_embedding_dim = {parameters.word_embedding_dim}\n',
                f'\n',
                f'conv_sent_encoder_n_hidden = {parameters.conv_sent_encoder_n_hidden}\n',
                f'conv_sent_encoder_output_dim = {parameters.conv_sent_encoder_output_dim}\n',
                f'conv_sent_encoder_kernel = {parameters.conv_sent_encoder_kernel}\n',
                f'conv_sent_encoder_dropout = {parameters.conv_sent_encoder_dropout}\n',
                f'conv_sent_encoder_training = {parameters.conv_sent_encoder_training}\n',
                f'\n',
                f'lstm_encoder_n_hidden = {parameters.lstm_encoder_n_hidden}\n',
                f'lstm_encoder_n_layer = {parameters.lstm_encoder_n_layer}\n',
                f'lstm_encoder_output_dim = {parameters.lstm_encoder_output_dim}\n',
                f'lstm_encoder_dropout = {parameters.lstm_encoder_dropout}\n',
                f'\n',
                f'ff_network_hidden_dim = {parameters.ff_network_hidden_dim}\n',
                f'\n',
                f'TRAINING PARAMETERS:\n',
                f'\n',
                f'num_training_data = {num_training}\n',
                f'epochs = {num_epochs}\n',
                f'batch_size = {batch_size}\n',
                f'learning_rate = {learning_rate}\n',
                f'teacher_forcing_prob = {teacher_forcing_prob}\n',
                f'training_time: {training_time}\n'
            ]
            f.writelines(lines)


    def __call__(self, _input, summary_length=0):
        return self.forward(_input)


if __name__ == '__main__':
    pass
