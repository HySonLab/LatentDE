import random
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import MeanMetric, MinMetric
from typing import List
from .modules import (
    ESM2Encoder,
    LatentEncoder,
    DropoutPredictor,
    PositionalPIController,
)
from .train_helper import KLDivergence
from ..common.constants import get_token2id


class BaseVAE(LightningModule):

    def __init__(
        self,
        expected_kl: float,
        pretrained_encoder_path: str = "facebook/esm2_t12_35M_UR50D",
        latent_dim: int = 64,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        Kp: float = 0.01,
        Ki: float = 0.0001,
        lr: float = 0.001,
        reduction: str = "sum",
    ):
        super(BaseVAE, self).__init__()

        self.save_hyperparameters(ignore=["device"])

        # Interpolation
        self.interp_ids = None

        # Encoder
        self.encoder = ESM2Encoder(pretrained_encoder_path)
        enc_dim = self.encoder.hidden_dim

        # Latent encoder
        self.latent_encoder = LatentEncoder(latent_dim, enc_dim)
        self.glob_attn_module = nn.Sequential(nn.Linear(enc_dim, 1), nn.Softmax(1))

        # Predictor
        self.predictor = DropoutPredictor(latent_dim, pred_hidden_dim, pred_dropout)

        self.pi_controller = PositionalPIController(expected_kl, kl_weight,
                                                    beta_min, beta_max, Kp, Ki)

        # Loss functions
        token2id = get_token2id()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.recon_loss = nn.CrossEntropyLoss(ignore_index=token2id["<pad>"],
                                              reduction=reduction)
        self.kl_div = KLDivergence(reduction)
        self.neg_loss = nn.MSELoss(reduction=reduction)

        # Metrics
        self.train_total_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_mse_loss = MeanMetric()

        self.valid_total_loss = MeanMetric()
        self.valid_kl_loss = MeanMetric()
        self.valid_recon_loss = MeanMetric()
        self.valid_mse_loss = MeanMetric()

        self.valid_best_loss = MinMetric()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, x: List[str]):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (torch.Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        enc_inp = self.encoder.tokenize(x).to(self.device)
        enc_out = self.encoder(enc_inp)  # [B, L, D]

        global_enc_out = self.glob_attn_module(enc_out)
        z_rep = torch.bmm(global_enc_out.transpose(1, 2), enc_out).squeeze(1)
        z, mu, logsigma = self.latent_encoder(z_rep)
        return z, mu, logsigma

    def predict(self, mu: torch.Tensor) -> torch.Tensor:
        """ Property prediction in VAE model

        Args:
            mu (torch.Tensor): mu vector of shape `[batch, latent_dim]`

        Returns:
            prop (torch.Tensor): Property value of shape `[batch, 1]`
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}

    def loss_function(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                      pred_prop: torch.Tensor, target_prob: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor):
        """Measure loss and control kl weight

        Args:
            pred_seq (torch.Tensor): prob, output of decoder of shape `[batch, vocab, seq_len]`
            target_seq (torch.Tensor): target sequence to reconstruct of shape `[batch, seq_len]`
            pred_prop (torch.Tensor): predicted property values of shape `[batch, 1]`
            target_prob (torch.Tensor): target property values of shape `[batch, 1]`
            mu (torch.Tensor): mean of latent z of shape `[batch, latent_dim]`
            logvar (torch.Tensor): log-variance of latent z of shape `[batch, latent_dim]`

        Returns:
            total_loss (torch.Tensor): total loss, scalar
            kl_loss (Tensor): Kullback-Leibler divergence, scalar
            recon_loss (Tensor): reconstruction loss, scalar
            mse_loss (Tensor): MSE loss, scalar
        """
        # KL divergence
        kl_loss = self.kl_div(mu, logvar)

        # Reconstruction loss
        recon_loss = self.recon_loss(pred_seq, target_seq)

        # MSE (predictor) loss
        mse_loss = self.mse_loss(pred_prop, target_prob)

        total_loss = self.hparams.kl_weight * kl_loss \
            + self.hparams.nll_weight * recon_loss \
            + self.hparams.mse_weight * mse_loss

        if self.training:
            # Control kl weight
            self.hparams.kl_weight = self.pi_controller(kl_loss.detach())

        return total_loss, kl_loss, recon_loss, mse_loss

    def predict_property_from_latent(self, mu: torch.Tensor) -> float:
        return self.predictor(mu)

    def model_step(self, batch):
        x, y = batch["sequences"], batch["fitness"]
        y = y.unsqueeze(1)
        dec_probs, pred_property, mu, logvar, seqs_ids = self.forward(x)

        loss, kl_loss, recon_loss, mse_loss = \
            self.loss_function(dec_probs, seqs_ids, pred_property, y, mu, logvar)

        return loss, kl_loss, recon_loss, mse_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, kl_loss, recon_loss, mse_loss = self.model_step(batch)

        # update and log metrics
        self.train_total_loss(loss)
        self.train_kl_loss(kl_loss)
        self.train_recon_loss(recon_loss)
        self.train_mse_loss(mse_loss)

        self.log("train_loss", self.train_total_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_kldiv", self.train_kl_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", self.train_recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_mse", self.train_mse_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_kl_weight", self.hparams.kl_weight, on_step=True,
                 on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, kl_loss, recon_loss, mse_loss = self.model_step(batch)

        # update and log metrics
        self.valid_total_loss(loss)
        self.valid_kl_loss(kl_loss)
        self.valid_recon_loss(recon_loss)
        self.valid_mse_loss(mse_loss)

        self.log("valid_loss", self.valid_total_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_kldiv", self.valid_kl_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_recon_loss", self.valid_recon_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_mse", self.valid_mse_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_start(self) -> None:
        self.valid_total_loss.reset()
        self.valid_kl_loss.reset()
        self.valid_mse_loss.reset()
        self.valid_recon_loss.reset()
        self.valid_best_loss.reset()

    def on_validation_epoch_end(self) -> None:
        cur_valid_loss = self.valid_total_loss.compute()
        self.valid_best_loss(cur_valid_loss)
        self.log("valid_best_loss", self.valid_best_loss.compute(), sync_dist=True, prog_bar=True)

    def generate_from_latent(latent: torch.Tensor):
        raise NotImplementedError

    def predict_fitness_with_scale(self,
                                   seqs: List[str],
                                   scale: float,
                                   factor: float,
                                   i: int):
        with torch.inference_mode():
            latent, *_ = self.encode_with_scale(seqs, scale, factor, i)
            scores = self.predict_property_from_latent(latent)
            return scores.squeeze().cpu().tolist()

    def reconstruct_from_wt(self,
                            wt_seq: List[str],
                            scale: float,
                            factor: float,
                            i: int) -> List[str]:
        """Reconstruct from wild-type sequence(s)"""
        with torch.inference_mode():
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

    def encode_with_scale(self, x: List[str], scale: float, factor: float, i: int):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (torch.Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        with torch.inference_mode():
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
