import torch.nn as nn
import torch


class DecoderV0(nn.Module):
    def __init__(self, n_features, hidden_dim) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.hidden = None

        self.simple_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.regression = nn.Linear(self.hidden_dim, self.n_features)

    def init_hidden(self, hidden_seq):
        hidden_final = hidden_seq[:, -1:]

        # the final hidden state must be sequence first
        self.hidden = hidden_final.permute(1, 0, 2)

    def forward(self, X):
        batch_first_output, self.hidden = self.simple_rnn(X, self.hidden)

        last_output = batch_first_output[:, -1:]
        out = self.regression(last_output)
        return out.view(-1, 1, self.n_features)
