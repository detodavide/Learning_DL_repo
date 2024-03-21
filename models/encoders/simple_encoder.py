import torch.nn as nn
import torch


class EncoderV0(nn.Module):
    def __init__(self, n_features, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.simple_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)

    def forward(self, X):
        rnn_out, self.hidden = self.simple_rnn(X)
        return rnn_out
