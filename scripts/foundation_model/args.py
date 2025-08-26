import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Foundation model training with AE/VAE embeddings + original diffusion')

    # dataset paths
    parser.add_argument('--trinity_path', type=str, default='./data/trinity_all_cache')
    parser.add_argument('--beat_path', type=str, default='./data/beat_english_v0.2.1/beat_all_cache')
    parser.add_argument('--ted_expressive_path', type=str, default='./data/ted_expressive_dataset/train')
    parser.add_argument('--datasets', type=str, nargs='+', choices=['trinity', 'beat', 'ted_expressive'],
                        default=['trinity', 'beat', 'ted_expressive'], help='Select 1 to 3 datasets to include')

    # dataloader
    parser.add_argument('--batch_per_dataset', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=34)
    parser.add_argument('--target_fps', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--persistent_workers', action='store_true', default=True)

    # AE/VAE swap
    parser.add_argument('--embed_backbone', type=str, choices=['ae', 'vae'], default='ae', help='Use AE or VAE latent as pose embedding')
    parser.add_argument('--latent_dim', type=int, default=128)

    # diffusion config: we reuse original structure; pose_dim is dataset-specific but head is adapted inside
    parser.add_argument('--diff_hidden_dim', type=int, default=256)
    parser.add_argument('--block_depth', type=int, default=8)
    parser.add_argument('--classifier_free', action='store_true', default=False)
    parser.add_argument('--null_cond_prob', type=float, default=0.1)

    # training modes
    parser.add_argument('--mode', type=str, choices=['frozen', 'step_unfreeze', 'isolated'], default='frozen')
    parser.add_argument('--unfreeze_steps', type=int, nargs=3, default=[1000, 3000, 6000], help='S1 S2 S3 for step_unfreeze mode')
    parser.add_argument('--windows', type=int, nargs=3, default=[2000, 2000, 2000], help='W1 W2 W3 lengths for isolated mode')

    # optim
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr_audio', type=float, default=1e-3)
    parser.add_argument('--lr_diff', type=float, default=1e-3)
    parser.add_argument('--lr_dec', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # io
    parser.add_argument('--log_dir', type=str, default='scripts/model/AEs/ckpt/foundation_logs')
    parser.add_argument('--ckpt_dir', type=str, default='scripts/model/AEs/ckpt/foundation_model')
    parser.add_argument('--resume', type=str, default='')

    return parser 