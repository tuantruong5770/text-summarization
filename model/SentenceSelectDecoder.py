import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Glimpse(nn.Module):
    def __init__(self, encoder_dim, lstm_dim, output_size):
        super(Glimpse, self).__init__()
        self.output_size = output_size
        self.w1 = nn.Linear(encoder_dim, output_size)
        self.w2 = nn.Linear(lstm_dim, output_size)
        self.v = nn.Linear(output_size, 1)

    def forward(self, encoded_sentences, lstm_output):
        document_size = encoded_sentences.size(0)
        z_tensor = lstm_output.expand(document_size, -1)
        w_h = self.w1(encoded_sentences)
        attention = w_h + self.w2(z_tensor)
        attention = torch.tanh(attention)
        attention = self.v(attention)
        attention = F.softmax(attention, dim=0)
        context_vector = torch.matmul(torch.t(attention), w_h)
        return context_vector.squeeze()


class PointerNetwork(nn.Module):
    def __init__(self, encoder_dim, context_size, hidden_size):
        super(PointerNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(encoder_dim, hidden_size)
        self.w2 = nn.Linear(context_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, encoded_sentences, context_vector):
        document_size = encoded_sentences.size(0)
        e_tensor = context_vector.expand(document_size, -1)
        w_h = self.w1(encoded_sentences)
        u = w_h + self.w2(e_tensor)
        u = torch.tanh(u)
        u = self.v(u)
        conditional_p = F.softmax(u, dim=0)
        return conditional_p.squeeze()


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_dim, lstm_dim, context_size, pointer_size):
        super(LSTMDecoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.SOE_token = nn.Linear(1, encoder_dim)
        self.lstm = nn.LSTM(encoder_dim, lstm_dim)
        self.glimpse = Glimpse(encoder_dim, lstm_dim, context_size)
        self.pointer = PointerNetwork(encoder_dim, context_size, pointer_size)

    def forward(self, _input, hidden, encoded_sentences, start_token=False):
        if start_token:
            _input = self.SOE_token(torch.tensor([1.0]))
        lstm_output, hidden = self.lstm(_input.view(1, 1, -1), hidden)
        lstm_output = lstm_output.squeeze()
        context_vector = self.glimpse(encoded_sentences, lstm_output)
        conditional_p = self.pointer(encoded_sentences, context_vector)
        return conditional_p, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.lstm_dim), torch.zeros(1, 1, self.lstm_dim)