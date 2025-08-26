import os
import sys
import math
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# make sure scripts is in path when launched from project root
sys.path.insert(0, 'scripts')
try:
    import importlib
    sys.modules['model'] = importlib.import_module('scripts.model')
except Exception:
    pass

from scripts.foundation_model.args import build_arg_parser
from scripts.foundation_model.model import FoundationModel
from scripts.foundation_model.schedules import build_param_groups, apply_mode_freeze, step_based_unfreeze, isolated_window_control
from scripts.data_loader.unified_data_loader import create_multi_dataloader


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not (1 <= len(args.datasets) <= 3):
        raise ValueError("--datasets must include 1 to 3 choices among ['trinity','beat','ted_expressive']")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    overrides = {
        "trinity": {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers},
        "beat": {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers},
        "ted_expressive": {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers},
    }

    # filter dataset paths by selection
    tr_path = args.trinity_path if 'trinity' in args.datasets and args.trinity_path and os.path.exists(args.trinity_path) else None
    be_path = args.beat_path if 'beat' in args.datasets and args.beat_path and os.path.exists(args.beat_path) else None
    te_path = args.ted_expressive_path if 'ted_expressive' in args.datasets and args.ted_expressive_path and os.path.exists(args.ted_expressive_path) else None

    loader = create_multi_dataloader(
        trinity_path=tr_path,
        beat_path=be_path,
        ted_expressive_path=te_path,
        samples_per_dataset_per_step=args.batch_per_dataset,
        target_fps=args.target_fps,
        n_poses=args.seq_len,
        dataloader_overrides=overrides,
    )

    model = FoundationModel(seq_len=args.seq_len,
                            latent_dim=args.latent_dim,
                            diff_hidden_dim=args.diff_hidden_dim,
                            block_depth=args.block_depth,
                            embed_backbone=args.embed_backbone).to(device)

    apply_mode_freeze(model, args.mode)

    # warm-up to instantiate diffusion transformer parameters
    with torch.no_grad():
        warm = loader.get_batch()
        for name, data in warm.items():
            if data['batch_size'] == 0:
                continue
            for k in ['pose_seqs', 'audios']:
                if isinstance(data.get(k), torch.Tensor):
                    data[k] = data[k].to(device, non_blocking=True).float()
            _ = model.forward_get_loss(name, data)
            break

    param_groups = build_param_groups(model, args.lr_audio, args.lr_diff, args.lr_dec, args.weight_decay)
    optimizer = optim.AdamW(param_groups)

    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    best_val = float('inf')

    steps_per_epoch = loader.get_steps_per_epoch(mode='max')
    print(f"Steps per epoch: {steps_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = loader.get_batch()
            loss = 0.0
            for name, data in batch.items():
                if data['batch_size'] == 0:
                    continue
                # move to device
                for k in ['pose_seqs', 'vec_seqs', 'audios', 'spectrograms']:
                    if isinstance(data.get(k), torch.Tensor):
                        data[k] = data[k].to(device, non_blocking=True).float()
                # compute loss per dataset
                loss += model.forward_get_loss(name, data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # mode-specific controls
            if args.mode == 'step_unfreeze':
                s1, s2, s3 = args.unfreeze_steps
                step_based_unfreeze(global_step, s1, s2, s3, model)
            elif args.mode == 'isolated':
                isolated_window_control(global_step, tuple(args.windows), model, optimizer)

            if global_step % 50 == 0:
                writer.add_scalar('train/step_loss', loss.item(), global_step)

            epoch_avg = epoch_loss / max(1, (step + 1))
            print(f"Epoch {epoch:03d} Step {step+1:04d}/{steps_per_epoch} Loss {loss.item():.4f}")

        # simple val-on-train few steps
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            it = iter(loader)
            for _ in range(min(20, steps_per_epoch)):
                batch = next(it)
                l = 0.0
                for name, data in batch.items():
                    if data['batch_size'] == 0:
                        continue
                    for k in ['pose_seqs', 'vec_seqs', 'audios', 'spectrograms']:
                        if isinstance(data.get(k), torch.Tensor):
                            data[k] = data[k].to(device, non_blocking=True).float()
                    l += model.forward_get_loss(name, data)
                val_loss += l.item()
            val_loss /= max(1, min(20, steps_per_epoch))

        writer.add_scalar('train/epoch_loss', epoch_avg, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        print(f"Epoch {epoch:03d} | train {epoch_avg:.4f} | val {val_loss:.4f}")

        # save
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config': vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, f'epoch_{epoch:03d}.pth'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.ckpt_dir, f'best.pth'))

    writer.close()


if __name__ == '__main__':
    main() 