import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Glimpse(nn.Module):
    def __init__(self, encoder_dim, lstm_dim, output_size):
        """
        Model for glimpse operation
        :param encoder_dim: context_aware_sent_rep_dim (h_j)
        :param lstm_dim: LSTMDecoder output dim
        :param output_size: context_size
        """
        super(Glimpse, self).__init__()
        self.output_size = output_size
        self.w1 = nn.Linear(encoder_dim, output_size)
        self.w2 = nn.Linear(lstm_dim, output_size)
        self.v = nn.Linear(output_size, 1)


    def forward(self, encoded_sentences, lstm_output):
        """
        Creating the e_t context vector
        :param encoded_sentences: torch.Size([num_sent, context_aware_sent_rep_dim])
        :param lstm_output: LSTMDecoder output dim
        :return: context_vector size context_size
        """
        document_size = encoded_sentences.size(0)
        # Copy z_t num_sentence times to make input more efficient
        z_tensor = lstm_output.expand(document_size, -1)
        w_h = self.w1(encoded_sentences)
        attention = self.v(torch.tanh(w_h + self.w2(z_tensor)))
        attention = F.softmax(attention, dim=0)
        context_vector = torch.matmul(torch.t(attention), w_h)
        return context_vector.squeeze()


class PointerNetwork(nn.Module):
    def __init__(self, encoder_dim, context_size, hidden_size):
        """
        Pointer network for calculating extract probability
        :param encoder_dim: context_aware_sent_rep_dim (h_j)
        :param context_size: size of context vector e_t
        :param hidden_size: num feature hidden state
        """
        super(PointerNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(encoder_dim, hidden_size)
        self.w2 = nn.Linear(context_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)


    def forward(self, encoded_sentences, context_vector):
        document_size = encoded_sentences.size(0)
        # Copy e_t num_sentence times to make input more efficient
        e_tensor = context_vector.expand(document_size, -1)
        u_t = self.v(torch.tanh(self.w1(encoded_sentences) + self.w2(e_tensor)))
        u_t = F.relu(u_t)
        conditional_p = F.softmax(u_t, dim=0)
        return conditional_p.squeeze()


class LSTMDecoderNoCoverage(nn.Module):
    def __init__(self, encoder_dim, lstm_dim, num_layer, context_size, pointer_size, dropout):
        """
        LSTMDecoder for extracting the next sentence in document
        :param encoder_dim: context_aware_sent_rep_dim
        :param lstm_dim: dim of hidden state
        :param num_layer: number of layers in LSTM
        :param context_size: size of context vector e_t
        :param pointer_size: number of hidden feature of pointer network
        :param dropout: dropout probability
        """
        super(LSTMDecoderNoCoverage, self).__init__()
        self.lstm_dim = lstm_dim
        self.num_layer = num_layer
        self.SOE_token = nn.Linear(1, encoder_dim)
        self.init_hidden_state = nn.Linear(1, lstm_dim * num_layer)
        self.init_cell_state = nn.Linear(1, lstm_dim * num_layer)
        self.lstm = nn.LSTM(encoder_dim, lstm_dim, num_layer, dropout=dropout)
        self.glimpse = Glimpse(encoder_dim, lstm_dim, context_size)
        self.pointer = PointerNetwork(encoder_dim, context_size, pointer_size)


    def forward(self, _input, hidden_states, encoded_sentences, start_token=False):
        """
        Take the input (selected context_aware_sent_rep) and output probability vector of the next sentence to extract
        :param _input: context_aware_sent_rep or a learnable start token if it is a start token
        :param hidden_states: tuple of previous hidden states of the lstm (h_t-1, c_t-1)
        :param encoded_sentences: tensor of size torch.Size([num_sentence, context_aware_sent_rep_dim])
        :param start_token: True if this is the first forward call of the LSTM
        :return: extraction probability vector of size num_sentence
        """
        if start_token:
            _input = self.SOE_token(torch.tensor([1.0]).to(device))
        lstm_output, hidden = self.lstm(_input.view(1, 1, -1), hidden_states)
        lstm_output = lstm_output.squeeze()
        context_vector = self.glimpse(encoded_sentences, lstm_output)
        conditional_p = self.pointer(encoded_sentences, context_vector)
        return conditional_p, hidden


    def init_hidden_zero(self):
        return (torch.zeros(self.num_layer, 1, self.lstm_dim).to(device),
                torch.zeros(self.num_layer, 1, self.lstm_dim).to(device))


    def init_hidden(self):
        return (self.init_hidden_state(torch.ones(1).to(device)).view(self.num_layer, 1, self.lstm_dim),
                self.init_cell_state(torch.ones(1).to(device)).view(self.num_layer, 1, self.lstm_dim))


if __name__ == "__main__":
    torch.manual_seed(0)

    model = LSTMDecoderNoCoverage(20, 25, 3, 30, 10, 0.0).to(device)
    hj = torch.rand(15, 20).to(device)
    hidden_states_0 = model.init_hidden_zero()

    p, hidden_states_t, c_g, c_p = model(hj[0], hidden_states_0, hj)

    print(p.size())
    print(p)

    # t = torch.rand(10)
    # m = nn.LSTM(10, 15)
    # out, hidden = m(t.view(1,1,-1))
    # out, hidden = m(t.view(1,1,-1), hidden)
    # print(out.size())
    # print(hidden[0].size())
    # print(hidden[1].size())
    # [0.0707, 0.0609, 0.0662, 0.0654, 0.0598, 0.0750, 0.0571, 0.0632, 0.0665, 0.0647, 0.0654, 0.0686, 0.0740, 0.0705, 0.0720]
