#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Tuple

# Reuse encoder/decoder building blocks for consistency with existing AE
from scripts.model.motion_ae import PoseEncoderConv, PoseDecoderConv

class SkeletonAE(nn.Module):
    """Skeleton-only sequence AutoEncoder (no audio).
    Uses the same conv encoder/decoder topology as MotionAE to stay consistent.
    """
    def __init__(self, seq_len: int, pose_dim: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim

        self.encoder = PoseEncoderConv(seq_len, pose_dim, latent_dim)
        self.decoder = PoseDecoderConv(seq_len, pose_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, pose_dim]
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class SkeletonVAE(nn.Module):
    """Skeleton-only sequence Variational AutoEncoder (no audio)."""
    def __init__(self, seq_len: int, pose_dim: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim

        # Reuse PoseEncoderConv up to flattened features by replicating its body
        # For simplicity, wrap it and then replace the final head with mu/logvar heads
        self._conv_encoder = PoseEncoderConv(seq_len, pose_dim, latent_dim)
        # Replace the out_net with a feature head that outputs a fixed-size feature
        # PoseEncoderConv produces flatten length = 384 for seq_len=34
        # We mirror the same MLP dims to extract a shared feature then heads
        self.feature_net = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = PoseDecoderConv(seq_len, pose_dim, latent_dim)

        # Hijack conv trunk from PoseEncoderConv, but we will not use its out_net
        self._conv_trunk = self._conv_encoder.net  # Conv trunk ending with Conv1d(64->32,3)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, pose_dim]
        x = x.transpose(1, 2)  # [B, pose_dim, T]
        h = self._conv_trunk(x)        # [B, 32, L]
        h = h.flatten(1)               # [B, 384] when T=34
        h = self.feature_net(h)        # [B, 128]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma) || N(0,1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() 