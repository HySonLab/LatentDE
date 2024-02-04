import random
import torch
import torch.nn as nn
from torch import Tensor
from types import SimpleNamespace
from typing import List
from .modules import (
    CNNDecoder,
    RNNDecoder,
    Upsampling,
    ESM2Encoder,
    LatentEncoder,
    DropoutPredictor,
    PositionalPIController
)
from .train_helper import KLDivergence
from ..common.constants import convert_ids2seqs, convert_seqs2ids, get_token2id, VOCAB


class BaseVAEModel(nn.Module):

    def __init__(self, cfg: SimpleNamespace, device: torch.device):
        super(BaseVAEModel, self).__init__()
        self.cfg = cfg
        self.device = device

        # Interpolation
        self.interp_ids = None
        self.kl_weight = self.cfg.kl_weight

        # Encoder
        self.encoder = ESM2Encoder(cfg.pretrained_encoder_path)
        enc_dim = self.encoder.hidden_dim

        # Latent encoder
        self.latent_encoder = LatentEncoder(cfg.latent_dim, enc_dim)
        self.glob_attn_module = nn.Sequential(nn.Linear(enc_dim, 1), nn.Softmax(1))

        # Predictor
        self.predictor = DropoutPredictor(cfg.latent_dim, cfg.pred_hidden_dim, cfg.pred_dropout)

        self.pi_controller = PositionalPIController(cfg.expected_kl, cfg.kl_weight,
                                                    cfg.beta_min, cfg.beta_max, cfg.Kp, cfg.Ki)

        # Loss functions
        token2id = get_token2id()
        reduction = 'none' if cfg.k_val is not None else cfg.reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.recon_loss = nn.CrossEntropyLoss(ignore_index=token2id["<pad>"],
                                              reduction=cfg.reduction)
        self.kl_div = KLDivergence(reduction)
        self.neg_loss = nn.MSELoss(reduction=reduction)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, x: List[str]):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        enc_inp = self.encoder.tokenize(x).to(self.device)
        enc_out = self.encoder(enc_inp)  # [B, L, D]

        global_enc_out = self.glob_attn_module(enc_out)
        z_rep = torch.bmm(global_enc_out.transpose(1, 2), enc_out).squeeze(1)
        z, mu, logsigma = self.latent_encoder(z_rep)
        return z, mu, logsigma

    def predict(self, mu: Tensor) -> Tensor:
        """ Property prediction in VAE model

        Args:
            mu (Tensor): mu vector of shape `[batch, latent_dim]`

        Returns:
            prop (Tensor): Property value of shape `[batch, 1]`
        """
        prop = self.predictor(mu)
        return prop

    def forward(self, seqs: List[str]):
        """Forward workflow of VAE

        Args:
            seqs: List[str]: list of protein sequences

        Returns:
            dec_probs (Tensor): log probability of vocabs of shape [batch, vocab, seq_len]
            pred_property (Tensor): Property value of shape [batch, 1]
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
            seqs_ids (Tensor): decoder input ids of shape `[batch, maxlen]`
        """
        raise NotImplementedError

    def loss_function(self, pred_seq: Tensor, target_seq: Tensor,
                      pred_prop: Tensor, target_prob: Tensor,
                      mu: Tensor, logvar: Tensor, latent: Tensor):
        """Measure loss and control kl weight

        Args:
            pred_seq (Tensor): prob, output of decoder of shape `[batch, vocab, seq_len]`
            target_seq (Tensor): target sequence to reconstruct of shape `[batch, seq_len]`
            pred_prop (Tensor): predicted property values of shape `[batch, 1]`
            target_prob (Tensor): target property values of shape `[batch, 1]`
            mu (Tensor): mean of latent z of shape `[batch, latent_dim]`
            logvar (Tensor): log-variance of latent z of shape `[batch, latent_dim]`

        Returns:
            total_loss (Tensor): total loss, scalar
            kl_loss (Tensor): Kullback-Leibler divergence, scalar
            recon_loss (Tensor): reconstruction loss, scalar
            mse_loss (Tensor): MSE loss, scalar
        """
        # KL divergence
        kl_loss = self.kl_div(mu, logvar)

        # Reconstruction loss
        if self.training and self.cfg.use_interp_sampling:
            hyp_seq = pred_seq[:-self.cfg.interp_size]
        else:
            hyp_seq = pred_seq
        recon_loss = self.recon_loss(hyp_seq, target_seq)

        # MSE (predictor) loss
        if self.training and self.cfg.use_neg_sampling:
            hyp_prop = pred_prop[:-self.cfg.neg_size]
            extend_prop = pred_prop[-self.cfg.neg_size:]
        else:
            hyp_prop = pred_prop
        mse_loss = self.mse_loss(hyp_prop, target_prob)

        # Interpolation loss
        bs = mu.size(0)
        if self.training and self.cfg.use_interp_sampling:
            seq_preds = nn.functional.gumbel_softmax(pred_seq, tau=1, dim=1, hard=True)
            seq_preds = seq_preds.transpose(1, 2).flatten(1, 2)
            seq_dist_mat = torch.cdist(seq_preds, seq_preds, p=1)

            ext_ids = torch.arange(bs, bs + self.cfg.interp_size)
            tr_dists = seq_dist_mat[self.interp_ids[:, 0], self.interp_ids[:, 1]]
            inter_dist1 = seq_dist_mat[ext_ids, self.interp_ids[:, 0]]
            inter_dist2 = seq_dist_mat[ext_ids, self.interp_ids[:, 1]]

            interp_loss = 0.5 * (inter_dist1 + inter_dist2) - 0.5 * tr_dists
            interp_loss = interp_loss.mean() \
                if self.reduction == "mean" else interp_loss.sum()
            interp_loss = max(0, interp_loss) * self.cfg.interp_weight
        else:
            interp_loss = 0.0

        # Negative sampling loss
        if self.training and self.cfg.use_neg_sampling:
            neg_targets = torch.ones(
                (self.cfg.neg_size), device=self.device) * self.cfg.neg_floor
            neg_loss = self.neg_loss(extend_prop.flatten(), neg_targets.flatten())
        else:
            neg_loss = 0.0

        # Latent regularization
        if self.cfg.regularize_latent:
            latent_loss = 0.5 * torch.linalg.vector_norm(latent, 2, dim=1)**2
            latent_loss = latent_loss.mean() \
                if self.reduction == "mean" else latent_loss.sum()
        else:
            latent_loss = 0.0

        total_loss = self.kl_weight * kl_loss \
            + self.cfg.nll_weight * recon_loss \
            + self.cfg.mse_weight * mse_loss \
            + self.cfg.interp_weight * interp_loss \
            + self.cfg.neg_weight * neg_loss \
            + self.cfg.latent_weight * latent_loss

        if self.training:
            # Control kl weight
            self.kl_weight = self.pi_controller(kl_loss.detach())

        return total_loss, kl_loss, recon_loss, mse_loss, interp_loss, neg_loss, latent_loss

    def predict_property_from_latent(self, mu: Tensor) -> float:
        return self.predictor(mu)

    def model_step(self, batch):
        x, y = batch["sequences"], batch["fitness"]
        y = y.unsqueeze(1).to(self.device)
        dec_probs, pred_property, mu, logvar, seqs_ids, latent = self.forward(x)

        loss, kl_loss, recon_loss, mse_loss, interp_loss, neg_loss, latent_loss = \
            self.loss_function(dec_probs, seqs_ids, pred_property, y, mu, logvar, latent)

        return loss, kl_loss, recon_loss, mse_loss, interp_loss, neg_loss, latent_loss

    def generate_from_latent(latent: Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def predict_fitness_with_scale(self,
                                   seqs: List[str],
                                   scale: float,
                                   factor: float,
                                   i: int):
        latent, *_ = self.encode_with_scale(seqs, scale, factor, i)
        scores = self.predict_property_from_latent(latent)
        return scores.squeeze().cpu().tolist()

    @torch.no_grad()
    def reconstruct_from_wt(self,
                            wt_seq: List[str],
                            scale: float,
                            factor: float,
                            i: int) -> List[str]:
        """Reconstruct from wild-type sequence(s)"""
        latent, *_ = self.encode_with_scale(wt_seq, scale, factor, i)
        seqs = self.generate_from_latent(latent)
        return seqs

    def reconstruct_from_wt_glob(self,
                                 wt_seq: List[str],
                                 scale: float,
                                 factor: float,
                                 i: int,
                                 batch_size: int) -> List[str]:
        wt_seq_chunk = [wt_seq[i:i + batch_size] for i in range(0, len(wt_seq), batch_size)]
        new_seqs = []
        for wt_seqs in wt_seq_chunk:
            seqs = self.reconstruct_from_wt(wt_seqs, scale, factor, i)
            new_seqs.extend(seqs)
        return new_seqs

    @torch.no_grad()
    def encode_with_scale(self, x: List[str], scale: float, factor: float, i: int):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        enc_inp = self.encoder.tokenize(x).to(self.device)
        enc_out = self.encoder(enc_inp)  # [B, L, D]

        global_enc_out = self.glob_attn_module(enc_out)
        z_rep = torch.bmm(global_enc_out.transpose(1, 2), enc_out).squeeze(1)
        z, mu, logsigma = self.latent_encoder(z_rep)

        if random.random() > 0.5:
            num_random = int(0.2 * z.shape[0])
            eps = torch.randn_like(mu[-num_random:])
            z[-num_random:] = z[-num_random:] + (scale - factor * i) * eps
            # z[:num_random] = (1 - scale) * z[:num_random] + scale * eps

        return z, mu, logsigma

    def interpolation_sampling(self, z_rep: Tensor):
        """Get interpolations between z_reps in batch"""
        z_dist_mat = self.pairwise_l2(z_rep)
        k_val = min(len(z_rep), 2)
        _, z_nn_ids = z_dist_mat.topk(k_val, largest=False)
        z_nn_ids = z_nn_ids[:, 1]

        z_nn = z_rep[:, 1].unsqueeze(1)
        z_interp = (z_rep + z_nn) / 2

        subset_ids = torch.randperm(len(z_rep), device=self.device)[:self.cfg.interp_size]
        sub_z_interp = z_interp[subset_ids]
        sub_nn_ids = z_nn_ids[subset_ids]

        self.interp_ids = torch.cat((subset_ids.unsqueeze(1), sub_nn_ids.unsqueeze(1)), dim=1)
        return sub_z_interp

    def add_negative_samples(self, z_rep):
        max2norm = torch.norm(z_rep, p=2, dim=1).max()
        rand_ids = torch.randperm(len(z_rep))
        if self.cfg.neg_focus:
            neg_z = 0.5 * torch.randn_like(z_rep)[:self.cfg.neg_size] \
                + z_rep[rand_ids][:self.cfg.neg_size]
            neg_z = neg_z / torch.linalg.vector_norm(neg_z, 2, dim=1).reshape(-1, 1)
            neg_z = neg_z * (max2norm * self.cfg.neg_norm)
        else:
            center = z_rep.mean(0, keepdims=True)
            dist_set = z_rep - center

            # gets maximally distant rep from center
            dist_sort = torch.norm(dist_set, 2, dim=1).reshape(-1, 1).sort().indices[-1]
            max_dist = dist_set[dist_sort]
            adj_dist = self.cfg.neg_norm * max_dist.repeat(len(z_rep), 1) - dist_set
            neg_z = z_rep + adj_dist
            neg_z = neg_z[rand_ids][:self.cfg.neg_size]

        return neg_z

    def pairwise_l2(self, x):
        bs = x.size(0)
        z1 = x.unsqueeze(0).expand(bs, -1, -1)
        z2 = x.unsqueeze(1).expand(-1, bs, -1)
        dist = torch.pow(z2 - z1, 2).sum(2)
        return dist


class GruVAEModel(BaseVAEModel):

    def __init__(self, cfg: SimpleNamespace, device: torch.device):
        super(GruVAEModel, self).__init__(cfg, device)
        # Decoder
        vocab_size = len(VOCAB)
        token2id = get_token2id()
        sos_idx = token2id["<sos>"]
        padding_idx = token2id["<pad>"]

        self.decoder = RNNDecoder(
            cfg.latent_dim, cfg.upsampler_min_deconv_dim, cfg.dec_hidden_dim, vocab_size,
            cfg.dec_num_layers, cfg.gru_dropout, cfg.inp_dropout, cfg.dec_attn_method,
            sos_idx, padding_idx, cfg.max_len, cfg.use_teacher_forcing, device
        )
        self.use_attn = True if cfg.dec_attn_method is not None else False

        if self.use_attn:
            # Upsampler
            self.upsampler = Upsampling(cfg.latent_dim,
                                        cfg.upsampler_max_filter,
                                        cfg.upsampler_low_res_dim,
                                        cfg.upsampler_min_deconv_dim,
                                        cfg.upsampler_num_deconv_layers,
                                        cfg.upsampler_kernel_size,
                                        cfg.upsampler_stride,
                                        cfg.upsampler_padding,
                                        cfg.upsampler_dropout,
                                        cfg.upsampler_act_type,
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
            convert_seqs2ids(seqs, add_sos=False, add_eos=True, max_length=self.cfg.max_len),
            dtype=torch.long,
            device=self.device,
        )  # [B, maxlen]

        # Interpolative sampling
        if self.training and self.cfg.use_interp_sampling:
            z_i_rep = self.interpolation_sampling(latent)
            interp_z_rep = torch.cat((latent, z_i_rep), 0)
            padded_input_ids = torch.cat((input_ids, input_ids[:z_i_rep.size(0)]), 0)
            logits = self.decode(interp_z_rep, padded_input_ids)
        else:
            logits = self.decode(latent, input_ids)

        # Negative sampling and predictor
        if self.training and self.cfg.use_neg_sampling:
            z_n_rep = self.add_negative_samples(latent)
            neg_z_rep = torch.cat((latent, z_n_rep), 0)
            pred_property = self.predict(neg_z_rep)
        else:
            pred_property = self.predict(latent)

        return logits, pred_property, mu, logvar, input_ids, latent

    def generate_from_latent(self, latents: List[Tensor] | Tensor) -> List[str]:
        def main_process(latent: Tensor):
            upsl_latent = self.upsampler(latent) if self.use_attn else None
            pred_tokens = self.decoder.generate_from_latent(latent, upsl_latent).tolist()
            seq = convert_ids2seqs(pred_tokens)
            return seq

        with torch.no_grad():
            if isinstance(latents, Tensor):
                return main_process(latents)
            else:
                seqs = []
                for latent in latents:
                    seqs.extend(main_process(latent))
                return seqs


class CNNVAEModel(BaseVAEModel):

    def __init__(self, cfg: SimpleNamespace, device: torch.device):
        super(CNNVAEModel, self).__init__(cfg, device)

        # Decoder
        vocab_size = len(VOCAB)
        self.decoder = CNNDecoder(cfg.dec_hidden_dim,
                                  cfg.latent_dim,
                                  vocab_size,
                                  cfg.max_len).to(device)

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
            convert_seqs2ids(seqs, add_sos=False, add_eos=False, max_length=self.cfg.max_len),
            dtype=torch.long,
            device=self.device,
        )   # [B, max_len]

        # Interpolative sampling
        if self.training and self.cfg.use_interp_sampling:
            z_i_rep = self.interpolation_sampling(latent)
            interp_z_rep = torch.cat((latent, z_i_rep), 0)
            logits = self.decode(interp_z_rep)
        else:
            logits = self.decode(latent)

        # Negative sampling
        if self.training and self.cfg.use_neg_sampling:
            z_n_rep = self.add_negative_samples(latent)
            neg_z_rep = torch.cat((latent, z_n_rep), 0)
            pred_property = self.predict(neg_z_rep)
        else:
            # Predictor
            pred_property = self.predict(latent)

        return logits, pred_property, mu, logvar, input_ids, latent

    def generate_from_latent(self, latent: Tensor) -> List[str]:
        with torch.no_grad():
            pred_tokens = self.decoder.generate_from_latent(latent).tolist()
            seq = convert_ids2seqs(pred_tokens)
            return seq


def get_vae_model(cfg, dec_type, device):
    if dec_type == "cnn":
        return CNNVAEModel(cfg, device)
    elif dec_type == "rnn":
        return GruVAEModel(cfg, device)
    else:
        raise ValueError(f"dec_type {dec_type} is not supproted.")
