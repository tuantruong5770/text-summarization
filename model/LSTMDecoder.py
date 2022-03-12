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
        self.wc = nn.Linear(1, output_size)
        self.v = nn.Linear(output_size, 1)


    def forward(self, encoded_sentences, lstm_output, coverage_vector):
        """
        Creating the e_t context vector
        :param encoded_sentences: torch.Size([num_sent, context_aware_sent_rep_dim])
        :param lstm_output: LSTMDecoder output dim
        :param coverage_vector: coverage vector of attention of previous time steps
        :return: context_vector size context_size
        """
        document_size = encoded_sentences.size(0)
        # Copy z_t num_sentence times to make input more efficient
        z_tensor = lstm_output.expand(document_size, -1)
        w_h = self.w1(encoded_sentences)
        attention = self.v(torch.tanh(w_h + self.w2(z_tensor) + self.wc(coverage_vector)))
        attention = F.softmax(attention, dim=0)
        coverage_vector = coverage_vector + attention
        context_vector = torch.matmul(torch.t(attention), w_h)
        return context_vector.squeeze(), coverage_vector


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
        self.wc = nn.Linear(1, hidden_size)
        self.v = nn.Linear(hidden_size, 1)


    def forward(self, encoded_sentences, context_vector, coverage_vector):
        document_size = encoded_sentences.size(0)
        # Copy e_t num_sentence times to make input more efficient
        e_tensor = context_vector.expand(document_size, -1)
        u_t = self.v(torch.tanh(self.w1(encoded_sentences) + self.w2(e_tensor) + self.wc(coverage_vector)))
        u_t = F.relu(u_t)
        conditional_p = F.softmax(u_t, dim=0)
        coverage_vector = coverage_vector + conditional_p
        return conditional_p.squeeze(), coverage_vector


class LSTMDecoder(nn.Module):
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
        super(LSTMDecoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.num_layer = num_layer
        self.SOE_token = nn.Linear(1, encoder_dim)
        self.init_hidden1 = nn.Linear(1, lstm_dim * num_layer)
        self.init_hidden2 = nn.Linear(1, lstm_dim * num_layer)
        self.lstm = nn.LSTM(encoder_dim, lstm_dim, num_layer, dropout=dropout)
        self.glimpse = Glimpse(encoder_dim, lstm_dim, context_size)
        self.pointer = PointerNetwork(encoder_dim, context_size, pointer_size)


    def forward(self, _input, hidden_states, encoded_sentences, coverage_vector_g, coverage_vector_p,
                start_token=False):
        """
        Take the input (context_aware_sent_rep) and output the probability vector of the next sentence to extract
        :param _input: context_aware_sent_rep or a learnable start token if it is a start token
        :param hidden_states: tuple of previous hidden states of the lstm (h_t-1, c_t-1)
        :param encoded_sentences: tensor of size torch.Size([num_sentence, context_aware_sent_rep_dim])
        :param coverage_vector_g: coverage vector for Glimpse layer
        :param coverage_vector_p: coverage vector for Pointer Network
        :param start_token: True if this is the first forward call of the LSTM
        :return: extraction probability vector of size num_sentence
        """
        if start_token:
            _input = self.SOE_token(torch.tensor([1.0]).to(device))
        lstm_output, hidden = self.lstm(_input.view(1, 1, -1), hidden_states)
        lstm_output = lstm_output.squeeze()
        context_vector, coverage_vector_g = self.glimpse(encoded_sentences, lstm_output, coverage_vector_g)
        conditional_p, coverage_vector_p = self.pointer(encoded_sentences, context_vector, coverage_vector_p)
        return conditional_p, hidden, coverage_vector_g, coverage_vector_p


    def init_hidden_zero(self):
        return (torch.zeros(self.num_layer, 1, self.lstm_dim).to(device),
                torch.zeros(self.num_layer, 1, self.lstm_dim).to(device))


    def init_hidden(self):
        return (self.init_hidden1(torch.ones(1).to(device)).view(self.num_layer, 1, self.lstm_dim),
                self.init_hidden2(torch.ones(1).to(device)).view(self.num_layer, 1, self.lstm_dim))


    @staticmethod
    def init_coverage_pointer(document_length):
        return torch.zeros(document_length, 1).to(device)

    @staticmethod
    def init_coverage_glimpse(document_length):
        return torch.zeros(document_length, 1).to(device)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = LSTMDecoder(20, 25, 3, 30, 10, 0.0).to(device)
    hj = torch.rand(15, 20).to(device)
    hidden_states_0 = model.init_hidden_zero()
    c_g = model.init_coverage_glimpse(15)
    c_p = model.init_coverage_pointer(15)

    p, hidden_states_t, c_g, c_p = model(hj[0], hidden_states_0, hj, c_g, c_p)

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
