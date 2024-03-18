import torch
import torch.nn as nn


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
