import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from random import random
from .components import LuongAttention, TokenEmbedding, ConvBlock


class RNNDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 upsampled_dim: int,
                 hidden_dim: int,
                 num_embeddings: int,
                 num_layers: int = 1,
                 gru_dropout: float = 0.1,
                 inp_dropout: float = 0.45,
                 attn_method: str = "dot",
                 sos_idx: int = 0,
                 padding_idx: int = 1,
                 max_len: int = 500,
                 use_teacher_forcing: float = 0.5,
                 device: Union[str, torch.device] = "cpu"):
        super(RNNDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_layers = num_layers
        self.sos_idx = sos_idx
        self.device = device
        self.num_embeddings = num_embeddings
        self.use_attn = True if attn_method is not None else False
        self.use_teacher_forcing = use_teacher_forcing

        self.hid_lin = nn.Linear(latent_dim, hidden_dim)
        self.embed = TokenEmbedding(num_embeddings, hidden_dim, padding_idx)
        self.dropout = nn.Dropout(inp_dropout)
        self.gru = nn.GRU(hidden_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0 if num_layers == 1 else gru_dropout)
        self.out = nn.Linear(hidden_dim, num_embeddings)
        if self.use_attn:
            self.mem_lin = nn.Linear(upsampled_dim, hidden_dim)
            self.concat_lin = nn.Linear(hidden_dim * 2, hidden_dim)
            self.attn = LuongAttention(attn_method, hidden_dim)

    def forward_step(self, input_step: Tensor, last_hidden: Tensor, upspl_latent: Tensor):
        """Forward step of the decoder

        Args:
            input_step (Tensor): current input of RNN of shape `[batch, 1]`
            last_hidden (Tensor): last hidden state of RNN of shape `[num_layers, B, hidden_dim]`
            upspl_latent (Tensor): upsampled latent vector of shape `[batch, maxlen, hidden_dim]`

        Returns:
            output (Tensor): log probs of shape `[B, num_embeddings]`
            hidden (Tensor): hidden state of gru of shape `[num_layers, B, hidden_dim]`
        """
        embedding = self.dropout(self.embed(input_step))    # (B, 1, hidden_dim)
        # Forward through unidirectional GRU
        # (B, 1, hidden_dim) - (num_layers, B, hidden_dim)
        rnn_output, hidden = self.gru(embedding, last_hidden)
        if self.use_attn:
            # Calculate attention weights from the current GRU output
            attn_weights = self.attn(rnn_output, upspl_latent)  # (B, 1, maxlen)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(upspl_latent)    # (B, 1, hidden_dim)
            # Concatenate weighted context vector and GRU output
            rnn_output = rnn_output.squeeze(1)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), dim=1)
            concat_output = torch.tanh(self.concat_lin(concat_input))   # (B, hidden_dim)
            # Predict next word
            output = self.out(concat_output)    # (B, num_embeddings)
        else:
            output = self.out(rnn_output.squeeze(1))

        return output, hidden

    def forward(self, latent: Tensor, upsampled_latent: Tensor, target: Tensor):
        """Decoder in VAE

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`
            upsampled_latent (Tensor): upsampled latent vector of shape
                                       `[batch, maxlen, min_deconv_dim]`
            target (Tensor): target ids of shape `[batch, maxlen]`

        Returns:
            Tensor: decoder scores of shape `[B, num_embeddings, maxlen]`
        """
        bs = latent.size(0)
        if self.use_attn:
            upsampled_latent = self.mem_lin(upsampled_latent)
        latent = self.hid_lin(latent)

        # Create initial states
        decoder_input = torch.LongTensor([[self.sos_idx] for _ in range(bs)]).to(self.device)
        decoder_hidden = torch.stack(
            [latent for _ in range(self.num_layers)], dim=0
        ).to(self.device)

        outputs = []

        for t in range(self.max_len):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, upsampled_latent
            )
            pred_tokens = decoder_output.argmax(1, keepdim=True).long()
            if self.training and self.use_teacher_forcing > random():
                decoder_input = target[:, t].view(-1, 1)
            else:
                decoder_input = pred_tokens

            outputs.append(decoder_output)

        return torch.stack(outputs, dim=2)

    @torch.inference_mode()
    def generate_from_latent(self, latent: Tensor, upsampled_latent: Tensor = None):
        logits = self.forward(latent, upsampled_latent, None)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        pred_tokens = log_probs.argmax(1)
        return pred_tokens


class CNNDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        input_dim: int,
        seq_len: int,
    ):
        super(CNNDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            ConvBlock(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim,
                      self.input_dim,
                      kernel_size=1,
                      padding=0),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

    def forward(self, z_rep: Tensor):
        h_rep = z_rep  # B x 1 X L
        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)
            h_rep = layer(h_rep)
        return h_rep

    @torch.inference_mode()
    def generate_from_latent(self, latent: Tensor) -> Tensor:
        logits = self.forward(latent.unsqueeze(1))
        log_probs = nn.functional.log_softmax(logits, dim=1)
        pred_tokens = log_probs.argmax(1)
        return pred_tokens
