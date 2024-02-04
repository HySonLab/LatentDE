import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


def get_activation(type: str, dim: int) -> nn.Module:
    if type == "relu":
        return nn.ReLU()
    elif type == "leaky_relu":
        return nn.LeakyReLU()
    elif type == "elu":
        return nn.ELU()
    elif type == "prelu":
        return nn.PReLU(dim)
    else:
        raise ValueError(f"Activation {type} is not supported.")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, padding_idx: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Deconv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: Union[Tuple[int, int], int] = 2,
                 padding: int = 0,
                 dropout: float = 0.2,
                 activation_type: str = "relu"):
        super(Deconv1d, self).__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.deconv = nn.ConvTranspose1d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = get_activation(activation_type, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Defined Deconvolution operation

        Args:
            x (Tensor): tensor of shape `[batch, in_channels, dim1]`

        Returns:
            Tensor: tensor of shape `[batch, out_channels, dim2]`
        """
        x = self.dropout(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Upsampling(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 max_filters: int,
                 low_res_dim: int = 64,
                 min_deconv_dim: int = 32,
                 num_deconv_layers: int = 3,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 0,
                 dropout: float = 0.2,
                 activation_type: str = "relu",
                 device: Union[torch.device, str] = "cpu"):
        super(Upsampling, self).__init__()
        self.low_res_dim = low_res_dim
        low_res_features = min(min_deconv_dim * 2**num_deconv_layers, max_filters)
        self.low_res_features = low_res_features
        self.latent_lin = nn.Linear(latent_dim, low_res_dim * low_res_features)
        self.deconvs = nn.ModuleList()
        for i in range(num_deconv_layers):
            out_channels = min(min_deconv_dim * 2**(num_deconv_layers - i - 1), max_filters)
            deconv = Deconv1d(low_res_features, out_channels, kernel_size,
                              stride, padding, dropout, activation_type)
            self.deconvs.append(deconv)
            low_res_features = out_channels

    def forward(self, latent: Tensor) -> Tensor:
        """Upsampling

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`

        Returns:
            h (Tensor):  tensor of shape `[B, min_deconv_dim, low_res_dim * stride**n_deconv_layer]`
        """
        h = self.latent_lin(latent)
        h = h.view(-1, self.low_res_features, self.low_res_dim)
        for deconv_layer in self.deconvs:
            h = deconv_layer(h)

        h = h.transpose(1, 2)

        return h


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super(LatentEncoder, self).__init__()
        self.linear_mu = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim * 2, latent_dim))
        self.linear_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

    def forward(self, hidden: Tensor):
        mu = self.linear_mu(hidden)
        logsigma = self.linear_logsigma(hidden)
        eps = torch.rand_like(mu)
        z = mu + eps * torch.exp(logsigma * 0.5)
        return z, mu, logsigma


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, keys: Tensor):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class LuongAttention(nn.Module):
    def __init__(self, method: str, hidden_dim: int):
        super(LuongAttention, self).__init__()
        assert method in ["dot", "general", "concat"], f"{method} is not supported."
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(hidden_dim))

    def dot_score(self, hidden: Tensor, encoder_output: Tensor) -> Tensor:
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden: Tensor, encoder_output: Tensor) -> Tensor:
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden: Tensor, encoder_output: Tensor) -> Tensor:
        energy = self.attn(torch.cat((hidden.expand(-1, encoder_output.size(1), -1),
                                      encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden: Tensor, encoder_outputs: Tensor):
        """Forward of attention mechanism

        Args:
            hidden (Tensor): output of rnn of shape `[batch, cur_len, hidden_dim]`
            encoder_outputs (Tensor): upsampled latent of shape `[batch, maxlen, hidden_dim]`

        Returns:
            Tensor: attention weights of shape `[batch, 1, maxlen]`
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.

          from:
          https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch
        """
        super(ConvBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False), nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):

        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = nn.functional.relu(out)

        return out
