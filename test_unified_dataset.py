#!/usr/bin/env python3

import os
import sys
import argparse
import time
import torch

# Add the scripts directory to Python path
sys.path.append('scripts')

# Import the new three-loader wrapper
from scripts.data_loader.unified_data_loader import (
    MultiDataLoaderWrapper,
    create_multi_dataloader,
)


def create_loader(trinity_path=None, beat_path=None, ted_expressive_path=None, batch_per_dataset=8):
    print("=" * 60)
    print("CREATING MULTI-DATALOADER (THREE-LOADER WRAPPER)")
    print("=" * 60)

    loader = create_multi_dataloader(
        trinity_path=trinity_path if trinity_path and os.path.exists(trinity_path) else None,
        beat_path=beat_path if beat_path and os.path.exists(beat_path) else None,
        ted_expressive_path=ted_expressive_path if ted_expressive_path and os.path.exists(ted_expressive_path) else None,
        samples_per_dataset_per_step=batch_per_dataset,
        target_fps=15,
        n_poses=34,
    )

    info = loader.info()
    print("Datasets loaded:")
    for name, meta in info["datasets"].items():
        print(f"  - {name}: size={meta['size']}, pose_dim={meta['pose_dim']}, batch_size={meta['batch_size']}, type={meta['type']}")

    print(f"Dataset order: {info['names']}")
    return loader


def test_balanced_batches(loader, steps=3, device=None):
    print("\n" + "=" * 60)
    print("TESTING BALANCED BATCHES (get_batch)")
    print("=" * 60)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    for step in range(steps):
        batch_dict = loader.get_batch()
        print(f"\nStep {step + 1}:")
        for dataset_name, batch in batch_dict.items():
            bsz = batch.get('batch_size', 0)
            pose = batch.get('pose_seqs')
            vec = batch.get('vec_seqs')
            audio = batch.get('audios')
            spec = batch.get('spectrograms')

            pose_shape = tuple(pose.shape) if hasattr(pose, 'shape') else 'N/A'
            vec_shape = tuple(vec.shape) if hasattr(vec, 'shape') else 'N/A'
            audio_shape = tuple(audio.shape) if hasattr(audio, 'shape') else 'N/A'
            spec_shape = tuple(spec.shape) if hasattr(spec, 'shape') else 'N/A'

            print(f"  {dataset_name}: batch_size={bsz}")
            print(f"    pose_seqs: {pose_shape}")
            print(f"    vec_seqs:  {vec_shape}")
            print(f"    audios:    {audio_shape}")
            print(f"    spectros:  {spec_shape}")


def test_iteration(loader, steps=2):
    print("\n" + "=" * 60)
    print("TESTING ITERATION (for balanced batches)")
    print("=" * 60)

    it = iter(loader)
    for i in range(steps):
        combined = next(it)
        print(f"\nIter step {i + 1}:")
        for name, batch in combined.items():
            bsz = batch.get('batch_size', 0)
            pose = batch.get('pose_seqs')
            pose_shape = tuple(pose.shape) if hasattr(pose, 'shape') else 'N/A'
            print(f"  {name}: batch_size={bsz}, pose_seqs={pose_shape}")


def test_steps_per_epoch(loader):
    print("\n" + "=" * 60)
    print("TESTING STEPS PER EPOCH")
    print("=" * 60)

    try:
        steps_max = loader.get_steps_per_epoch(mode='max')
        steps_min = loader.get_steps_per_epoch(mode='min')
        # Prefer 'beat' if present, else fall back to the first dataset
        ref_name = 'beat' if 'beat' in loader.dataset_names else loader.dataset_names[0]
        steps_ref = loader.get_steps_per_epoch(mode='name', reference_name=ref_name)

        print(f"  max steps/epoch (largest dataset): {steps_max}")
        print(f"  min steps/epoch (smallest dataset): {steps_min}")
        print(f"  steps/epoch ({ref_name}): {steps_ref}")
    except Exception as e:
        print(f"  ✗ Error computing steps per epoch: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test MultiDataLoaderWrapper (three-loader)')
    parser.add_argument('--trinity_path', type=str,
                        default="./data/trinity_all_cache",
                        help='Path to Trinity dataset cache')
    parser.add_argument('--beat_path', type=str,
                        default="./data/beat_english_v0.2.1/beat_all_cache",
                        help='Path to BEAT dataset cache')
    parser.add_argument('--ted_expressive_path', type=str,
                        default="./data/ted_expressive_dataset/train",
                        help='Path to TED Expressive dataset')
    parser.add_argument('--batch_per_dataset', type=int, default=8,
                        help='Samples per dataset per training step')
    args = parser.parse_args()

    available = []
    if args.trinity_path and os.path.exists(args.trinity_path):
        available.append('trinity')
    if args.beat_path and os.path.exists(args.beat_path):
        available.append('beat')
    if args.ted_expressive_path and os.path.exists(args.ted_expressive_path):
        available.append('ted_expressive')

    print("AVAILABLE DATASETS:", available if available else 'None')
    if not available:
        print("ERROR: No datasets found! Please check your paths.")
        return

    # Create loader
    loader = create_loader(
        trinity_path=args.trinity_path,
        beat_path=args.beat_path,
        ted_expressive_path=args.ted_expressive_path,
        batch_per_dataset=args.batch_per_dataset,
    )

    # Run tests
    test_balanced_batches(loader, steps=3)
    test_iteration(loader, steps=2)
    test_steps_per_epoch(loader)

    print("\nAll tests complete.")


if __name__ == "__main__":
    main() 