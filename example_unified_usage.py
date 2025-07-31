#!/usr/bin/env python3
"""
Example script showing how to use the unified dataset for training
This demonstrates the basic usage patterns without actual training
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the scripts directory to Python path
sys.path.append('scripts')

from scripts.data_loader.unified_data_loader import (
    create_unified_dataset, 
    unified_collate_fn
)

def basic_unified_dataset_example():
    """Basic example of creating and using a unified dataset"""
    print("="*50)
    print("BASIC UNIFIED DATASET EXAMPLE")
    print("="*50)
    
    # Create unified dataset with available datasets
    # Update these paths to match your actual dataset locations
    unified_dataset = create_unified_dataset(
        trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
        beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
        # ted_expressive_path="./data/ted_expressive_dataset/train" if os.path.exists("./data/ted_expressive_dataset/train") else None
    )
    
    print(f"Created unified dataset with {len(unified_dataset)} samples")
    
    # Get dataset information
    info = unified_dataset.get_dataset_info()
    print("\nDataset composition:")
    for name, stats in info['datasets'].items():
        print(f"  {name}: {stats['count']} samples (dim: {stats['pose_dim']})")
    
    # Create dataloader
    dataloader = DataLoader(
        unified_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=unified_collate_fn,
        num_workers=0
    )
    
    print(f"\nCreated DataLoader with batch_size=16")
    
    # Iterate through a few batches
    for batch_idx, batch_dict in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Contains datasets: {list(batch_dict.keys())}")
        
        # Process each dataset in the batch separately
        for dataset_name, dataset_batch in batch_dict.items():
            pose_seqs = dataset_batch['pose_seqs']
            vec_seqs = dataset_batch['vec_seqs']
            audios = dataset_batch['audios']
            batch_size = dataset_batch['batch_size']
            
            print(f"  {dataset_name}:")
            print(f"    Batch size: {batch_size}")
            print(f"    Pose shape: {pose_seqs.shape}")
            print(f"    Vec shape: {vec_seqs.shape}")
            print(f"    Audio shape: {audios.shape}")
            
            # Here you would route each dataset to its appropriate encoder
            # Example: trinity_output = trinity_encoder(pose_seqs, audios)
            
        if batch_idx >= 2:  # Just show first 3 batches
            break

def filtered_dataset_example():
    """Example of using dataset filtering"""
    print("\n" + "="*50)
    print("FILTERED DATASET EXAMPLE")
    print("="*50)
    
    # Create unified dataset
    unified_dataset = create_unified_dataset(
        trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
        beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
    )
    
    info = unified_dataset.get_dataset_info()
    available_datasets = list(info['datasets'].keys())
    
    print(f"Available datasets: {available_datasets}")
    
    # Example 1: Train on only Trinity data
    if 'trinity' in available_datasets:
        trinity_only = unified_dataset.filter_by_dataset('trinity')
        print(f"\nTrinity-only dataset: {len(trinity_only)} samples")
        
        # Create dataloader for Trinity only
        trinity_loader = DataLoader(
            trinity_only,
            batch_size=8,
            shuffle=True,
            collate_fn=unified_collate_fn
        )
        
        # Test one batch
        batch = next(iter(trinity_loader))
        print(f"Trinity batch contains: {list(batch.keys())}")
    
    # Example 2: Train on multiple specific datasets
    if len(available_datasets) > 1:
        subset_datasets = available_datasets[:2]  # First two available
        multi_dataset = unified_dataset.filter_by_dataset(subset_datasets)
        print(f"\nMulti-dataset ({subset_datasets}): {len(multi_dataset)} samples")

def training_simulation_example():
    """Example showing how you might use this in a training loop"""
    print("\n" + "="*50)
    print("TRAINING SIMULATION EXAMPLE")
    print("="*50)
    
    # Create unified dataset
    unified_dataset = create_unified_dataset(
        trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
        beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        unified_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=unified_collate_fn,
        num_workers=0
    )
    
    print("Simulating training loop...")
    
    # Simulate a few training steps
    for epoch in range(2):  # Just 2 epochs for demo
        print(f"\nEpoch {epoch + 1}:")
        
        epoch_loss = 0
        for batch_idx, batch_dict in enumerate(dataloader):
            
            # Process each dataset type separately
            batch_losses = {}
            
            for dataset_name, dataset_batch in batch_dict.items():
                pose_seqs = dataset_batch['pose_seqs']
                vec_seqs = dataset_batch['vec_seqs']
                audios = dataset_batch['audios']
                
                # Simulate forward pass (replace with actual model)
                # Different datasets might use different encoders/models
                if dataset_name == 'trinity':
                    # Use Trinity-specific processing
                    simulated_loss = torch.randn(1).abs()  # Fake loss
                elif dataset_name == 'beat':
                    # Use BEAT-specific processing  
                    simulated_loss = torch.randn(1).abs()  # Fake loss
                elif dataset_name == 'ted_expressive':
                    # Use TED Expressive-specific processing
                    simulated_loss = torch.randn(1).abs()  # Fake loss
                else:
                    simulated_loss = torch.randn(1).abs()  # Fake loss
                
                batch_losses[dataset_name] = simulated_loss.item()
            
            # Combine losses (you might weight them differently)
            total_loss = sum(batch_losses.values())
            epoch_loss += total_loss
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"  Batch {batch_idx}: {batch_losses}, Total: {total_loss:.3f}")
            
            if batch_idx >= 20:  # Limit for demo
                break
        
        avg_loss = epoch_loss / min(21, len(dataloader))
        print(f"  Average loss: {avg_loss:.3f}")

def dataset_statistics_example():
    """Example showing how to analyze dataset statistics"""
    print("\n" + "="*50)
    print("DATASET STATISTICS EXAMPLE")
    print("="*50)
    
    # Create unified dataset
    unified_dataset = create_unified_dataset(
        trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
        beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
    )
    
    info = unified_dataset.get_dataset_info()
    
    print("Detailed dataset statistics:")
    total_samples = 0
    
    for dataset_name, stats in info['datasets'].items():
        count = stats['count']
        pose_dim = stats['pose_dim']
        total_samples += count
        percentage = (count / info['total_samples']) * 100
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Samples: {count:,} ({percentage:.1f}%)")
        print(f"  Pose dimension: {pose_dim}")
        print(f"  Data path: {stats['path']}")
        
        # Get a sample to show actual data shapes
        dataset_indices = unified_dataset.get_samples_by_dataset(dataset_name)
        if dataset_indices:
            sample = unified_dataset[dataset_indices[0]]
            pose_seq = sample[2]
            audio = sample[4]
            spectrogram = sample[5]
            
            print(f"  Sample shapes:")
            print(f"    Pose sequence: {pose_seq.shape}")
            print(f"    Audio: {audio.shape}")
            print(f"    Spectrogram: {spectrogram.shape}")
    
    print(f"\nTOTAL: {total_samples:,} samples across {len(info['datasets'])} datasets")

def main():
    """Run all examples"""
    print("UNIFIED DATASET USAGE EXAMPLES")
    print("="*50)
    
    # Check if any datasets are available
    available_paths = []
    test_paths = [
        "./data/trinity_all_cache",
        "./data/beat_english_v0.2.1/beat_all_cache",
        "./data/ted_expressive_dataset/train"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            available_paths.append(path)
    
    if not available_paths:
        print("ERROR: No dataset caches found!")
        print("Please make sure you have created dataset caches using the preprocessing scripts.")
        print("Expected paths:")
        for path in test_paths:
            print(f"  {path}")
        return
    
    print(f"Found {len(available_paths)} dataset(s)")
    
    try:
        # Run examples
        basic_unified_dataset_example()
        filtered_dataset_example()
        training_simulation_example()
        dataset_statistics_example()
        
        print("\n" + "="*50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 