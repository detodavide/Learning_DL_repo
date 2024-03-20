import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs) -> None:
        super(SquareModel, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.hidden = None

        self.simple_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X -> B S F
        # OUT -> B S H
        # final hidden state -> 1 B H
        batch_first_output, self.hidden = self.simple_rnn(X)

        last_output = batch_first_output[:, -1]
        out = self.classifier(last_output)
        return out.view(-1, self.n_outputs)


class SquareModelGRU(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs) -> None:
        super(SquareModelGRU, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.hidden = None

        self.simple_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X -> B S F
        # OUT -> B S H
        # final hidden state -> 1 B H
        batch_first_output, self.hidden = self.simple_rnn(X)

        last_output = batch_first_output[:, -1]
        out = self.classifier(last_output)
        return out.view(-1, self.n_outputs)


class SquareModelLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs) -> None:
        super(SquareModelLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.hidden = None
        # added cell state
        self.cell = None

        self.simple_rnn = nn.LSTM(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X -> B S F
        # OUT -> B S H
        # final hidden state -> 1 B H
        batch_first_output, (self.hidden, self.cell) = self.simple_rnn(X)

        last_output = batch_first_output[:, -1]
        out = self.classifier(last_output)
        return out.view(-1, self.n_outputs)


class SquareModelPacked(nn.Module):
    # This model expect a packed sequence in the forward method
    def __init__(self, n_features, hidden_dim, n_outputs) -> None:
        super(SquareModelPacked, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None

        self.simple_rnn = nn.LSTM(
            self.n_features,
            self.hidden_dim,
            bidirectional=True,
        )
        # Bidirectional model has 2 * hidden_dim
        self.classifier = nn.Linear(2 * self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X -> B S F
        # OUT -> B S H
        # final hidden state -> 1 B H
        rnn_out, (self.hidden, self.cell) = self.simple_rnn(X)

        # Unpacking output for the linear layer
        batch_first_output, seq_sizes = pad_packed_sequence(rnn_out, batch_first=True)

        seq_idx = torch.arange(seq_sizes.size(0))
        last_output = batch_first_output[seq_idx, seq_sizes - 1]
        out = self.classifier(last_output)
        return out.view(-1, self.n_outputs)
