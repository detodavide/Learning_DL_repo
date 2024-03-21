import torch.nn as nn
import torch


class EncoderDecoderV0(nn.Module):
    def __init__(
        self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.outputs = None

    def init_outputs(self, batch_size):

        device = next(self.parameters()).device
        self.outputs = torch.zeros(
            batch_size, self.target_len, self.encoder.n_features
        ).to(device)

    def store_outputs(self, i, out):

        self.outputs[:, i : i + 1, :] = out

    def forward(self, X):
        source_seq = X[:, : self.input_len, :]
        target_seq = X[:, self.input_len :, :]
        self.init_outputs(X.shape[0])

        # Encoder
        hidden_seq = self.encoder(source_seq)

        # Decoder
        # Inputs and hidden_state to pass
        self.decoder.init_hidden(hidden_seq)
        dec_inputs = source_seq[:, -1:, :]

        for i in range(self.target_len):
            out = self.decoder(dec_inputs)
            self.store_outputs(i, out)

            prob = self.teacher_forcing_prob
            if not self.training:
                prob = 0

            if torch.rand(1) <= prob:
                dec_inputs = target_seq[:, i : i + 1, :]
            else:
                dec_inputs = out

        return self.outputs
