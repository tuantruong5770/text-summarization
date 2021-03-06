import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FFNetwork(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        """
        Base model for setting base evaluation. Basic feed forward network.
        :param encoder_dim:
        :param hidden_dim:
        """
        super(FFNetwork, self).__init__()
        self.wd = nn.Linear(encoder_dim, hidden_dim)
        self.wc = nn.Linear(encoder_dim, 1)
        self.ws = nn.Linear(hidden_dim, encoder_dim)
        self.bias = nn.Linear(1, 1)

    def forward(self, _input):
        """
        :param _input: all context aware sentence vector representation dimension N_d x encoder_dim
        :return: output: extract probability vector of each sentence
        """
        document_vec = torch.tanh(self.wd(torch.mean(_input, dim=0)))
        doc_len = _input.size(0)
        bias = self.bias(torch.ones(doc_len).to(device).unsqueeze(1))
        ext_p = self.wc(_input) + torch.matmul(_input, self.ws(document_vec)).unsqueeze(1) + bias
        ext_p = F.softmax(ext_p, dim=0)
        return ext_p.squeeze()


if __name__ == "__main__":
    sent_vec = torch.ones(5, 10).to(device)
    m = FFNetwork(10, 100).to(device)
    out = m(sent_vec)
    print(out)