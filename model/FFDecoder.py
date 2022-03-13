import torch
import torch.nn as nn
import torch.nn.functional as F


class FFExtractor(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        super(FFExtractor, self).__init__()
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
        bias = self.bias(torch.ones(doc_len).unsqueeze(1))
        ext_p = self.wc(_input) + torch.matmul(_input, self.ws(document_vec)).unsqueeze(1) + bias
        ext_p = F.softmax(ext_p, dim=0)
        return ext_p


if __name__ == "__main__":
    sent_vec = torch.ones(5, 10)
    m = FFExtractor(10, 100)
    out = m(sent_vec)
    print(out)