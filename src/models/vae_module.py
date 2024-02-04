import torch
from torch import Tensor
from typing import List, Union
from .base import BaseVAE
from .modules import CNNDecoder, RNNDecoder, Upsampling
from ..common.constants import convert_ids2seqs, convert_seqs2ids, get_token2id, VOCAB


class GruVAE(BaseVAE):

    def __init__(
        self,
        expected_kl: float,
        pretrained_encoder_path: str = "facebook/esm2_t12_35M_UR50D",
        latent_dim: int = 64,
        upsampler_max_filter: int = 300,
        upsampler_kernel_size: int = 2,
        upsampler_low_res_dim: int = 64,
        upsampler_min_deconv_dim: int = 32,
        upsampler_num_deconv_layers: int = 3,
        upsampler_stride: int = 2,
        upsampler_padding: int = 0,
        upsampler_dropout: float = 0.2,
        upsampler_act_type: str = "relu",
        dec_hidden_dim: int = 512,
        dec_num_layers: int = 3,
        dec_attn_method: str = "dot",
        use_teacher_forcing: bool = True,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        inp_dropout: float = 0.45,
        gru_dropout: float = 0.2,
        max_len: int = 500,
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        Kp: float = 0.01,
        Ki: float = 0.0001,
        lr: float = 0.001,
        device: Union[torch.device, str] = "cuda",
        reduction: str = "sum",
    ):
        super(GruVAE, self).__init__(
            expected_kl, pretrained_encoder_path, latent_dim,
            pred_hidden_dim, pred_dropout, nll_weight, mse_weight,
            kl_weight, beta_min, beta_max, Kp, Ki, lr, reduction,
        )

        self.save_hyperparameters(ignore=["device"])

        # Decoder
        vocab_size = len(VOCAB)
        token2id = get_token2id()
        sos_idx = token2id["<sos>"]
        padding_idx = token2id["<pad>"]

        self.decoder = RNNDecoder(
            latent_dim, upsampler_min_deconv_dim, dec_hidden_dim, vocab_size,
            dec_num_layers, gru_dropout, inp_dropout, dec_attn_method,
            sos_idx, padding_idx, max_len, use_teacher_forcing, device
        )
        self.use_attn = True if dec_attn_method is not None else False

        if self.use_attn:
            # Upsampler
            self.upsampler = Upsampling(latent_dim,
                                        upsampler_max_filter,
                                        upsampler_low_res_dim,
                                        upsampler_min_deconv_dim,
                                        upsampler_num_deconv_layers,
                                        upsampler_kernel_size,
                                        upsampler_stride,
                                        upsampler_padding,
                                        upsampler_dropout,
                                        upsampler_act_type,
                                        device)

    def decode(self, latent: Tensor, gt_seq: Tensor) -> Tensor:
        """ Decoder workflow in VAE model

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`
            gt_seq (Tensor): target sequence ids of shape `[batch, seq_len]`

        Returns:
            logits (Tensor): logits of vocabs of shape `[batch, vocab, seq_len]`
        """
        if self.use_attn:
            decoder_input = self.upsampler(latent)
        else:
            decoder_input = None
        logits = self.decoder(latent, decoder_input, gt_seq)     # [B, vocab_size, seq_len]
        return logits

    def forward(self, seqs: List[str]):
        # Encoder
        latent, mu, logvar = self.encode(seqs)

        # Decoder
        input_ids = torch.tensor(
            convert_seqs2ids(seqs, add_sos=False, add_eos=True, max_length=self.hparams.max_len),
            dtype=torch.long,
            device=self.device,
        )  # [B, maxlen]

        logits = self.decode(latent, input_ids)

        pred_property = self.predict(latent)

        return logits, pred_property, mu, logvar, input_ids, latent

    def generate_from_latent(self, latents: List[Tensor] | Tensor) -> List[str]:
        def main_process(latent: Tensor):
            upsl_latent = self.upsampler(latent) if self.use_attn else None
            pred_tokens = self.decoder.generate_from_latent(latent, upsl_latent).tolist()
            seq = convert_ids2seqs(pred_tokens)
            return seq

        with torch.inference_mode():
            if isinstance(latents, Tensor):
                return main_process(latents)
            else:
                seqs = []
                for latent in latents:
                    seqs.extend(main_process(latent))
                return seqs


class CNNVAE(BaseVAE):

    def __init__(
        self,
        expected_kl: float,
        pretrained_encoder_path: str = "facebook/esm2_t12_35M_UR50D",
        latent_dim: int = 64,
        dec_hidden_dim: int = 512,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        max_len: int = 500,
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        Kp: float = 0.01,
        Ki: float = 0.0001,
        lr: float = 0.001,
        device: Union[torch.device, str] = "cuda",
        reduction: str = "sum",
    ):
        super(CNNVAE, self).__init__(
            expected_kl, pretrained_encoder_path, latent_dim,
            pred_hidden_dim, pred_dropout, nll_weight, mse_weight,
            kl_weight, beta_min, beta_max, Kp, Ki, lr, reduction,
        )

        # Decoder
        vocab_size = len(VOCAB)
        self.max_len = max_len

        self.decoder = CNNDecoder(dec_hidden_dim, latent_dim, vocab_size, max_len).to(device)

    def decode(self, latent: Tensor, gt_seq: Tensor = None) -> Tensor:
        """ Decoder workflow in VAE model

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`
            gt_seq (Tensor): target sequence ids of shape `[batch, seq_len]`

        Returns:
            logits (Tensor): logits of vocabs of shape `[batch, vocab, seq_len]`
        """
        logits = self.decoder(latent)
        return logits

    def forward(self, seqs: List[str]):
        # Encoder
        latent, mu, logvar = self.encode(seqs)

        # Decoder
        input_ids = torch.tensor(
            convert_seqs2ids(seqs, add_sos=False, add_eos=False, max_length=self.max_len),
            dtype=torch.long,
            device=self.device,
        )   # [B, max_len]

        logits = self.decode(latent)

        pred_property = self.predict(latent)

        return logits, pred_property, mu, logvar, input_ids, latent

    def generate_from_latent(self, latents: List[Tensor] | Tensor) -> List[str]:

        def main_process(latent: Tensor):
            pred_tokens = self.decoder.generate_from_latent(latent).tolist()
            seq = convert_ids2seqs(pred_tokens)
            return seq

        with torch.inference_mode():
            if isinstance(latents, Tensor):
                return main_process(latents)
            else:
                seqs = []
                for latent in latents:
                    seqs.extend(main_process(latent))
                return seqs
