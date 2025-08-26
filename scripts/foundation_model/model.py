import os
import torch
import torch.nn as nn
from typing import Dict

from scripts.model.AEs.temporal_autoencoders import SkeletonAE, SkeletonVAE
from scripts.model.pose_diffusion import WavEncoder
from scripts.model.diffusion_util import TransformerModel, VarianceSchedule
from scripts.model.diffusion_net import DiffusionNet


DATASET_CFG = {
    "trinity": {
        "pose_dim": 129,
        "ae_ckpt": "scripts/model/AEs/ckpt/ae_trinity/best.pth",
        "vae_ckpt": "scripts/model/AEs/ckpt/vae_trinity/best.pth",
    },
    "beat": {
        "pose_dim": 177,
        "ae_ckpt": "scripts/model/AEs/ckpt/ae_beat/best.pth",
        "vae_ckpt": "scripts/model/AEs/ckpt/vae_beat/best.pth",
    },
    "ted_expressive": {
        "pose_dim": 114,
        "ae_ckpt": "scripts/model/AEs/ckpt/ae_ted_expressive/best.pth",
        "vae_ckpt": "scripts/model/AEs/ckpt/vae_ted_expressive/best.pth",
    },
}


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


class FoundationModel(nn.Module):
    """
    Dataset-aware fusion using AE/VAE latent embeddings in place of raw pose channels.
    - For each dataset name, use its own encoder/decoder from saved AE/VAE.
    - Build context as in original PoseDiffusion: concat [pre_seq, audio_feat_seq] but
      with pre_seq replaced by a tiled latent projection.
    - Keep original TransformerModel and DiffusionNet and their loss.
    """
    def __init__(self, seq_len: int, latent_dim: int, diff_hidden_dim: int, block_depth: int,
                 embed_backbone: str = 'ae'):
        super().__init__()

        assert embed_backbone in ('ae', 'vae')
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.embed_backbone = embed_backbone

        # Load dataset-specific full models and expose decoders
        self.models: Dict[str, nn.Module] = {}
        self.decoders: Dict[str, nn.Module] = {}
        self.pose_dims: Dict[str, int] = {}
        for name, cfg in DATASET_CFG.items():
            pose_dim = cfg['pose_dim']
            self.pose_dims[name] = pose_dim
            if embed_backbone == 'ae':
                model = SkeletonAE(seq_len=seq_len, pose_dim=pose_dim, latent_dim=latent_dim)
                ckpt_path = cfg['ae_ckpt']
            elif embed_backbone == 'vae':
                model = SkeletonVAE(seq_len=seq_len, pose_dim=pose_dim, latent_dim=latent_dim)
                ckpt_path = cfg['vae_ckpt']
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location='cpu')
                msd = state.get('model_state', state)
                model.load_state_dict(msd, strict=False)
            else:
                print(f"[foundation] Warning: checkpoint not found for {name}: {ckpt_path}. Using random init.")

            self.models[name] = model
            self.decoders[name] = model.decoder

        self.models = nn.ModuleDict(self.models)
        self.decoders = nn.ModuleDict(self.decoders)

        # Shared projections from latent to per-dataset pose_dim to allow tiling into T frames
        self.latent_to_pose: Dict[str, nn.Module] = {}
        for name, pose_dim in self.pose_dims.items():
            self.latent_to_pose[name] = nn.Sequential(
                nn.Linear(latent_dim, pose_dim),
                nn.Tanh(),
            )
        self.latent_to_pose = nn.ModuleDict(self.latent_to_pose)

        # Audio encoder (original)
        self.audio_encoder = WavEncoder()

        # Original diffusion with unchanged transformer and variance schedule.
        self.transformer_cfg = {
            "hidden_dim": diff_hidden_dim,
            "depth": block_depth // 2,
            "decoder_depth": block_depth // 2,
        }
        self.diffusion = _DynamicDiffusion(self.transformer_cfg)

    def _encode(self, dataset_name: str, poses: torch.Tensor) -> torch.Tensor:
        model = self.models[dataset_name]
        if isinstance(model, SkeletonAE):
            return model.encoder(poses)
        elif isinstance(model, SkeletonVAE):
            mu, logvar = model.encode(poses)
            return mu
        else:
            return model.encoder(poses)

    def forward_get_loss(self, dataset_name: str, batch: Dict[str, torch.Tensor]):
        poses = batch['pose_seqs'].float()  # [B,T,D] or [B,T,*,*]
        audios = batch['audios'].float()

        # Flatten poses to [B, T, D]
        if poses.dim() > 3:
            poses = poses.reshape(poses.shape[0], poses.shape[1], -1)

        z = self._encode(dataset_name, poses)
        pose_dim = self.pose_dims[dataset_name]
        pose_token = self.latent_to_pose[dataset_name](z).unsqueeze(1).repeat(1, self.seq_len, 1)
        audio_feat_seq = self.audio_encoder(audios)  # [B,T,32]
        in_data = torch.cat([pose_token, audio_feat_seq], dim=2)

        loss = self.diffusion.get_loss(num_pose=self.seq_len, pose_dim=pose_dim, in_data=in_data, target=poses)
        return loss

    @torch.no_grad()
    def forward_sample(self, dataset_name: str, batch: Dict[str, torch.Tensor]):
        poses = batch['pose_seqs'].float()
        audios = batch['audios'].float()

        if poses.dim() > 3:
            poses = poses.reshape(poses.shape[0], poses.shape[1], -1)

        z = self._encode(dataset_name, poses)
        pose_dim = self.pose_dims[dataset_name]
        pose_token = self.latent_to_pose[dataset_name](z).unsqueeze(1).repeat(1, self.seq_len, 1)
        audio_feat_seq = self.audio_encoder(audios)
        in_data = torch.cat([pose_token, audio_feat_seq], dim=2)
        samples = self.diffusion.sample(num_pose=self.seq_len, pose_dim=pose_dim, in_data=in_data)
        return samples

    # Freeze helpers used by schedules
    def freeze_encoders(self, flag: bool):
        for name, mdl in self.models.items():
            # freeze everything except decoder
            for p in mdl.parameters():
                p.requires_grad = not flag
            for p in self.decoders[name].parameters():
                p.requires_grad = True  # decoder controlled separately

    def freeze_decoders(self, flag: bool):
        for dec in self.decoders.values():
            set_requires_grad(dec, not flag)


class _DynamicDiffusion(nn.Module):
    """
    A helper to create and apply DiffusionNet with runtime pose_dim and context_dim
    per batch, while keeping weights persistent across calls.
    """
    def __init__(self, tf_cfg: Dict):
        super().__init__()
        self.tf_cfg = tf_cfg
        self.net = None
        self.var_sched = VarianceSchedule(num_steps=500, beta_1=1e-4, beta_T=0.02, mode='linear')

    def _ensure(self, num_pose: int, pose_dim: int, context_dim: int, device: torch.device):
        embed_dim = pose_dim + 3 + context_dim
        if self.net is None:
            self.net = TransformerModel(num_pose=num_pose,
                pose_dim=pose_dim,
                embed_dim=embed_dim,
                hidden_dim=self.tf_cfg['hidden_dim'],
                depth=self.tf_cfg['depth'],
                decoder_depth=self.tf_cfg['decoder_depth'])
        else:
            curr_pose_dim = self.net.out[-1].out_features if hasattr(self.net, 'out') else None
            if curr_pose_dim != pose_dim or self.net.linear.in_features != embed_dim:
                self.net = TransformerModel(num_pose=num_pose,
                    pose_dim=pose_dim,
                    embed_dim=embed_dim,
                    hidden_dim=self.tf_cfg['hidden_dim'],
                    depth=self.tf_cfg['depth'],
                    decoder_depth=self.tf_cfg['decoder_depth'])

        # Ensure modules/buffers are on the same device as inputs
        self.net = self.net.to(device)
        self.var_sched = self.var_sched.to(device)

        return DiffusionNet(net=self.net, var_sched=self.var_sched)

    def get_loss(self, num_pose: int, pose_dim: int, in_data: torch.Tensor, target: torch.Tensor):
        context_dim = in_data.shape[-1]
        diff = self._ensure(num_pose=num_pose, pose_dim=pose_dim, context_dim=context_dim, device=in_data.device)
        return diff.get_loss(target, in_data)

    @torch.no_grad()
    def sample(self, num_pose: int, pose_dim: int, in_data: torch.Tensor):
        context_dim = in_data.shape[-1]
        diff = self._ensure(num_pose=num_pose, pose_dim=pose_dim, context_dim=context_dim, device=in_data.device)
        return diff.sample(num_pose, in_data, pose_dim) 