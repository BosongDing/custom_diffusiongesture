#!/usr/bin/env python3
import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add scripts to path
sys.path.append('scripts')

from scripts.model.AEs.temporal_autoencoders import SkeletonVAE
from scripts.data_loader.unified_data_loader import create_multi_dataloader


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_ckpt(state, ckpt_dir, filename):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, filename))


def train_one_epoch(model, optimizer, loader, device, use_vec=True, beta=1.0):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0

    for _ in range( math.ceil( max(1, min(loader.dataset_sizes.values())) / max(1, min(loader.dataset_batch_sizes.values())) ) ):
        batch = loader.get_batch()
        for name, data in batch.items():
            if data['batch_size'] == 0:
                continue
            x = data['vec_seqs'] if use_vec else data['pose_seqs']
            x = x.to(device, non_blocking=True).float()
            recon, mu, logvar, _ = model(x)
            recon_loss = mse(recon, x)
            kl = model.kl_divergence(mu, logvar)
            loss = recon_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate(model, loader, device, use_vec=True, steps=20, beta=1.0):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        it = iter(loader)
        for _ in range(steps):
            batch = next(it)
            for name, data in batch.items():
                if data['batch_size'] == 0:
                    continue
                x = data['vec_seqs'] if use_vec else data['pose_seqs']
                x = x.to(device, non_blocking=True).float()
                recon, mu, logvar, _ = model(x)
                recon_loss = mse(recon, x)
                kl = model.kl_divergence(mu, logvar)
                loss = recon_loss + beta * kl
                total_loss += loss.item()
                total_batches += 1
    return total_loss / max(1, total_batches)


def main():
    parser = argparse.ArgumentParser(description='Train Skeleton-only VAE')
    parser.add_argument('--trinity_path', type=str, default='./data/trinity_all_cache')
    parser.add_argument('--beat_path', type=str, default='./data/beat_english_v0.2.1/beat_all_cache')
    parser.add_argument('--ted_expressive_path', type=str, default='./data/ted_expressive_dataset/train')
    parser.add_argument('--batch_per_dataset', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=34)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--use_vec', action='store_true', help='Use direction vectors as inputs (default)')
    parser.add_argument('--use_pose', dest='use_vec', action='store_false', help='Use absolute pose instead of vectors')
    parser.set_defaults(use_vec=True)
    # NEW: restrict training to a single dataset
    parser.add_argument('--dataset', type=str, choices=['trinity', 'beat', 'ted_expressive'], default='ted_expressive',
                        help='Select which dataset to train on exclusively')
    # NEW: dataloader performance
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers for the selected dataset')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Batches prefetched per worker')

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1e-4, help='KL weight')
    parser.add_argument('--ckpt_dir', type=str, default='scripts/model/AEs/ckpt/vae')
    parser.add_argument('--log_dir', type=str, default='scripts/model/AEs/ckpt/vae_logs')
    args = parser.parse_args()

    # Enforce single-dataset training by nulling other paths and validating selection
    if args.dataset == 'trinity':
        if not os.path.exists(args.trinity_path):
            raise ValueError(f"Trinity path does not exist: {args.trinity_path}")
        args.beat_path = None
        args.ted_expressive_path = None
        print(f"Training on Trinity only: {args.trinity_path}")
    elif args.dataset == 'beat':
        if not os.path.exists(args.beat_path):
            raise ValueError(f"BEAT path does not exist: {args.beat_path}")
        args.trinity_path = None
        args.ted_expressive_path = None
        print(f"Training on BEAT only: {args.beat_path}")
    elif args.dataset == 'ted_expressive':
        if not os.path.exists(args.ted_expressive_path):
            raise ValueError(f"TED Expressive path does not exist: {args.ted_expressive_path}")
        args.trinity_path = None
        args.beat_path = None
        print(f"Training on TED Expressive only: {args.ted_expressive_path}")

    device = get_device()

    overrides = {
        args.dataset: {
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": True,
            "persistent_workers": True,
        }
    }

    loader = create_multi_dataloader(
        trinity_path=args.trinity_path if args.trinity_path and os.path.exists(args.trinity_path) else None,
        beat_path=args.beat_path if args.beat_path and os.path.exists(args.beat_path) else None,
        ted_expressive_path=args.ted_expressive_path if args.ted_expressive_path and os.path.exists(args.ted_expressive_path) else None,
        samples_per_dataset_per_step=args.batch_per_dataset,
        target_fps=15,
        n_poses=args.seq_len,
        dataloader_overrides=overrides,
    )

    pose_dims = [meta['pose_dim'] for meta in loader.info()['datasets'].values()]
    model_pose_dim = max(pose_dims)

    class PadToDim(nn.Module):
        def __init__(self, target_dim):
            super().__init__()
            self.target_dim = target_dim
        def forward(self, x):
            b,t,d = x.shape
            if d == self.target_dim:
                return x
            pad = self.target_dim - d
            return torch.nn.functional.pad(x, (0,pad), 'constant', 0.0)

    pad = PadToDim(model_pose_dim).to(device)

    model = SkeletonVAE(seq_len=args.seq_len, pose_dim=model_pose_dim, latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=args.log_dir)

    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        loss_tr = train_one_epoch(model, optimizer, loader, device, use_vec=args.use_vec, beta=args.beta)
        writer.add_scalar('train/loss', loss_tr, epoch)

        loss_val = evaluate(model, loader, device, use_vec=args.use_vec, steps=20, beta=args.beta)
        writer.add_scalar('val/loss', loss_val, epoch)

        print(f"Epoch {epoch:03d} | train {loss_tr:.4f} | val {loss_val:.4f}")

        if loss_val < best_val:
            best_val = loss_val
            save_ckpt({'epoch': epoch,
                       'model_state': model.state_dict(),
                       'optimizer_state': optimizer.state_dict(),
                       'pose_dim': model_pose_dim,
                       'seq_len': args.seq_len,
                       'latent_dim': args.latent_dim,
                       'beta': args.beta,
                       'use_vec': args.use_vec},
                      args.ckpt_dir,
                      f'best.pth')
        if epoch % 5 == 0:
            save_ckpt({'epoch': epoch,
                       'model_state': model.state_dict(),
                       'optimizer_state': optimizer.state_dict(),
                       'pose_dim': model_pose_dim,
                       'seq_len': args.seq_len,
                       'latent_dim': args.latent_dim,
                       'beta': args.beta,
                       'use_vec': args.use_vec},
                      args.ckpt_dir,
                      f'epoch_{epoch:03d}.pth')

    writer.close()

if __name__ == '__main__':
    main() 