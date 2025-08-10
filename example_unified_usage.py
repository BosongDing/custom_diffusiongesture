#!/usr/bin/env python3

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the scripts directory to Python path
sys.path.append('scripts')

# Import the unified dataloader
from scripts.data_loader.unified_data_loader import (
    UnifiedMotionDataset, 
    unified_collate_fn, 
    create_unified_dataset
)

def main():
    """Minimal example of using the unified data loader"""
    
    print("=== Unified Data Loader Example ===")
    
    # Method 1: Using the convenience function (recommended)
    print("\n1. Using create_unified_dataset() convenience function:")
    
    try:
        # Create unified dataset with available datasets
        unified_dataset = create_unified_dataset(
            trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
            beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
            ted_expressive_path="./data/ted_expressive_dataset/train" if os.path.exists("./data/ted_expressive_dataset/train") else None,
            target_fps=15,
            n_poses=34
        )
        
        print(f"✓ Successfully created unified dataset with {len(unified_dataset)} samples")
        
        # Print dataset composition
        dataset_info = unified_dataset.get_dataset_info()
        print(f"Total datasets: {dataset_info['num_datasets']}")
        for name, info in dataset_info['datasets'].items():
            print(f"  {name}: {info['count']} samples, dim: {info['pose_dim']}")
        
    except Exception as e:
        print(f"✗ Failed to create unified dataset: {e}")
        return
    
    # Method 2: Manual configuration (for more control)
    print("\n2. Manual configuration example:")
    
    dataset_configs = []
    
    # Add Trinity if available
    if os.path.exists("./data/trinity_all_cache"):
        dataset_configs.append({
            'name': 'trinity',
            'path': './data/trinity_all_cache',
            'type': 'lmdb_new',
            'pose_dim': 129,
            'weight': 1.0
        })
    
    # Add BEAT if available
    if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache"):
        dataset_configs.append({
            'name': 'beat',
            'path': './data/beat_english_v0.2.1/beat_all_cache',
            'type': 'lmdb_new',
            'pose_dim': 177,
            'weight': 1.0
        })
    
    # Add TED Expressive if available
    if os.path.exists("./data/ted_expressive_dataset/train"):
        dataset_configs.append({
            'name': 'ted_expressive',
            'path': './data/ted_expressive_dataset/train',
            'type': 'lmdb_expressive',
            'pose_dim': 126,
            'weight': 1.0
        })
    
    if dataset_configs:
        try:
            manual_dataset = UnifiedMotionDataset(
                dataset_configs=dataset_configs,
                target_fps=15,
                n_poses=34
            )
            print(f"✓ Manual dataset created with {len(manual_dataset)} samples")
        except Exception as e:
            print(f"✗ Manual dataset creation failed: {e}")
    
    # 3. Creating a DataLoader
    print("\n3. Creating DataLoader:")
    
    try:
        dataloader = DataLoader(
            unified_dataset,
            batch_size=4,  # Small batch for demo
            shuffle=True,
            collate_fn=unified_collate_fn,
            num_workers=0  # Use 0 for debugging, increase for performance
        )
        
        print(f"✓ DataLoader created with batch_size=4")
        
        # Test loading a batch
        print("\n4. Loading a sample batch:")
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            
            # The batch is a dictionary with dataset names as keys
            for dataset_name, dataset_batch in batch.items():
                batch_size = dataset_batch['batch_size']
                pose_shape = dataset_batch['pose_seqs'].shape
                vec_shape = dataset_batch['vec_seqs'].shape
                
                print(f"  {dataset_name}: {batch_size} samples")
                print(f"    pose_seqs shape: {pose_shape}")
                print(f"    vec_seqs shape: {vec_shape}")
                print(f"    audio shape: {dataset_batch['audios'].shape}")
                print(f"    spectrogram shape: {dataset_batch['spectrograms'].shape}")
            
            # Only process first batch for demo
            break
            
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
    
    # 5. Filtering by dataset
    print("\n5. Filtering by specific dataset:")
    
    try:
        # Get only BEAT samples
        beat_only = unified_dataset.filter_by_dataset('beat')
        if len(beat_only) > 0:
            print(f"✓ Filtered to BEAT only: {len(beat_only)} samples")
            
            # Create dataloader for BEAT only
            beat_dataloader = DataLoader(
                beat_only,
                batch_size=2,
                shuffle=False,
                collate_fn=unified_collate_fn,
                num_workers=0
            )
            
            # Test one batch
            for batch in beat_dataloader:
                print(f"  BEAT batch contains {len(batch)} dataset(s)")
                break
        else:
            print("No BEAT samples available for filtering demo")
            
    except Exception as e:
        print(f"✗ Filtering test failed: {e}")
    
    # 6. Accessing individual samples
    print("\n6. Accessing individual samples:")
    
    try:
        if len(unified_dataset) > 0:
            sample = unified_dataset[0]
            word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info, dataset_info = sample
            
            print(f"✓ Sample 0:")
            print(f"  Dataset: {dataset_info['name']}")
            print(f"  Pose dim: {dataset_info['pose_dim']}")
            print(f"  pose_seq shape: {pose_seq.shape}")
            print(f"  vec_seq shape: {vec_seq.shape}")
            print(f"  audio shape: {audio.shape}")
            
    except Exception as e:
        print(f"✗ Individual sample access failed: {e}")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main() 