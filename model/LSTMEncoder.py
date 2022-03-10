import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout):
        """
        Bidirectional LSTM for context-aware sentence representation

        :param input_dim: dim of sentence rep from convolutional
        :param n_hidden: number of hidden states for lstm
        :param n_layer: number of layer for lstm
        :param dropout: dropout probability
        """
        super().__init__()
        self._n_layer = n_layer
        self._n_hidden = n_hidden
        self._init_h_c = self.init_hidden()
        self.lstm = nn.LSTM(input_dim, n_hidden, n_layer, dropout=dropout, bidirectional=True)


    def forward(self, _input):
        """
        Output context aware sentence representation

        :param _input: tensor of all sentence representation in the document. torch.Size([num_sent, sent_rep_dim])
        :return: torch.Size([num_sent, n_hidden * 2])
        """
        # Pad input with dim (num_sentence, 1, sent_rep_dim)
        # Since this LSTM is bidirectional, discard returned hidden state at t_final
        output, _ = self.lstm(_input.unsqueeze(1), self._init_h_c)
        return output.squeeze()


    def init_hidden(self):
        return (torch.zeros(self._n_layer * 2, 1, self._n_hidden).to(device),
                torch.zeros(self._n_layer * 2, 1, self._n_hidden).to(device))


if __name__ == '__main__':
    sent_rep = torch.rand(15, 30)
    encoder = LSTMEncoder(30, 30, 2, 0.0)
    print(encoder(sent_rep).size())

