import torch.nn as nn
from Pinnoloc.layers.reservoir import LinearReservoirRing
from Pinnoloc.layers.embedding import EmbeddingFixedPad, OneHotEncoding, Encoder
import torch


# TODO: Support transient for padded sequences with lengths
class StackedNetwork(nn.Module):
    def __init__(self, block_cls, n_layers, d_input, d_model, d_output,
                 encoder, decoder, to_vec,
                 min_encoder_scaling=0.0, max_encoder_scaling=1.0,
                 min_decoder_scaling=0.0, max_decoder_scaling=1.0,
                 layer_dropout=0.0,
                 **block_args):
        """
        Stack multiple blocks of the same type to form a deep network.
        """
        encoder_models = ['conv1d', 'reservoir', 'embedding', 'onehot']
        decoder_models = ['conv1d', 'reservoir']

        if encoder not in encoder_models:
            raise ValueError('Encoder must be one of {}'.format(encoder_models))

        if decoder not in decoder_models:
            raise ValueError('Decoder must be one of {}'.format(decoder_models))

        super().__init__()

        if encoder == 'conv1d':
            self.encoder = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=1)
        elif encoder == 'reservoir':
            self.encoder = LinearReservoirRing(d_input=d_input, d_output=d_model,
                                               min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                               field='real')
        elif encoder == 'embedding':
            self.encoder = EmbeddingFixedPad(vocab_size=d_input, d_model=d_model, padding_idx=0)
        elif encoder == 'onehot':
            self.encoder = OneHotEncoding(vocab_size=d_input, d_model=d_model,
                                          min_radius=min_encoder_scaling, max_radius=max_encoder_scaling,
                                          padding_idx=0)

        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_args) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                                       for _ in range(n_layers)])
        self.to_vec = to_vec

        if decoder == 'conv1d':
            self.decoder = nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=1)
        elif decoder == 'reservoir':
            self.decoder = LinearReservoirRing(d_input=d_model, d_output=d_output,
                                               min_radius=min_decoder_scaling, max_radius=max_decoder_scaling,
                                               field='real')

    def forward(self, u, lengths=None):
        """
        args:
            u: torch tensor of shape (B, d_input, L)
            lengths: torch tensor of shape (B)
        return:
            y: torch tensor of shape (B, d_output) or (B, d_output, L))
        """
        y = self.encoder(u)  # (B, d_input, L) -> (B, d_model, L)

        for layer, dropout in zip(self.layers, self.dropouts):
            y, _ = layer(y)
            y = dropout(y)

        if self.to_vec:
            if lengths is not None:
                # Convert lengths to zero-based indices by subtracting 1
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2)  # Shape (B, 1, 1)

                # Expand indices to match the dimensions needed for gathering
                indices = indices.expand(y.shape[0], y.shape[1], 1)  # Shape (B, H, 1)
                y = y.gather(-1, indices)  # (B, d_model, L) -> (B, d_model, 1)
            else:
                y = y[:, :, -1:]  # (B, d_model, L) -> (B, d_model, 1)
            y = self.decoder(y).squeeze(-1)  # (B, d_model, L) -> (B, d_output)
        else:
            y = self.decoder(y)  # (B, d_model, L) -> (B, d_output, L)

        return y
