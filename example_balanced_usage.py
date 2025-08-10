#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add the scripts directory to Python path
sys.path.append('scripts')

# Import the balanced unified dataloader
from scripts.data_loader.unified_data_loader import (
    BalancedUnifiedDataset,
    create_balanced_unified_dataset
)

# Mock encoders for demonstration
class TrinityEncoder(nn.Module):
    """Mock encoder for Trinity dataset (pose_dim=129)"""
    def __init__(self, pose_dim=129, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Output embedding
        )
    
    def forward(self, pose_seq, audio, spectrogram):
        # pose_seq shape: [batch, time, pose_dim]
        batch_size, time_steps, pose_dim = pose_seq.shape
        
        # Flatten time and pose dimensions
        pose_flat = pose_seq.view(batch_size, -1)
        
        # Process through encoder
        embedding = self.encoder(pose_flat)
        return embedding

class BeatEncoder(nn.Module):
    """Mock encoder for BEAT dataset (pose_dim=177)"""
    def __init__(self, pose_dim=177, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Output embedding
        )
    
    def forward(self, pose_seq, audio, spectrogram):
        # pose_seq shape: [batch, time, pose_dim]
        batch_size, time_steps, pose_dim = pose_seq.shape
        
        # Flatten time and pose dimensions
        pose_flat = pose_seq.view(batch_size, -1)
        
        # Process through encoder
        embedding = self.encoder(pose_flat)
        return embedding

class TedExpressiveEncoder(nn.Module):
    """Mock encoder for TED Expressive dataset (pose_dim=126)"""
    def __init__(self, pose_dim=126, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Output embedding
        )
    
    def forward(self, pose_seq, audio, spectrogram):
        # pose_seq shape: [batch, time, pose_dim]
        batch_size, time_steps, pose_dim = pose_seq.shape
        
        # Flatten time and pose dimensions
        pose_flat = pose_seq.view(batch_size, -1)
        
        # Process through encoder
        embedding = self.encoder(pose_flat)
        return embedding

def main():
    """Example of using balanced dataset with 3 different encoders"""
    
    print("=== Balanced Multi-Encoder Example ===")
    
    # 1. Create balanced dataset
    print("\n1. Creating balanced dataset...")
    
    try:
        balanced_dataset = create_balanced_unified_dataset(
            trinity_path="./data/trinity_all_cache" if os.path.exists("./data/trinity_all_cache") else None,
            beat_path="./data/beat_english_v0.2.1/beat_all_cache" if os.path.exists("./data/beat_english_v0.2.1/beat_all_cache") else None,
            ted_expressive_path="./data/ted_expressive_dataset/train" if os.path.exists("./data/ted_expressive_dataset/train") else None,
            batch_size=8,  # Each dataset will provide 8 samples per batch
            target_fps=15,
            n_poses=34
        )
        
        print(f"✓ Balanced dataset created with {len(balanced_dataset)} batches per epoch")
        
    except Exception as e:
        print(f"✗ Failed to create balanced dataset: {e}")
        return
    
    # 2. Create encoders for each dataset
    print("\n2. Creating encoders for each dataset...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoders = {}
    
    # Create encoders based on available datasets
    if 'trinity' in balanced_dataset.dataset_names:
        encoders['trinity'] = TrinityEncoder(pose_dim=129).to(device)
        print("✓ Trinity encoder created")
    
    if 'beat' in balanced_dataset.dataset_names:
        encoders['beat'] = BeatEncoder(pose_dim=177).to(device)
        print("✓ BEAT encoder created")
    
    if 'ted_expressive' in balanced_dataset.dataset_names:
        encoders['ted_expressive'] = TedExpressiveEncoder(pose_dim=126).to(device)
        print("✓ TED Expressive encoder created")
    
    # 3. Create DataLoader (no custom collate_fn needed for balanced dataset)
    print("\n3. Creating DataLoader...")
    
    dataloader = DataLoader(
        balanced_dataset,
        batch_size=1,  # This is meta-batch size (1 means 1 set of balanced batches)
        shuffle=True,
        num_workers=0
    )
    
    print(f"✓ DataLoader created")
    
    # 4. Training simulation
    print("\n4. Training simulation...")
    
    # Put encoders in training mode
    for encoder in encoders.values():
        encoder.train()
    
    # Mock optimizers
    optimizers = {}
    for name, encoder in encoders.items():
        optimizers[name] = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    for epoch in range(2):  # Just 2 epochs for demo
        print(f"\nEpoch {epoch + 1}:")
        
        epoch_losses = {name: 0.0 for name in encoders.keys()}
        num_batches = 0
        
        for batch_idx, batch_dict in enumerate(dataloader):
            
            # Process each dataset batch with its corresponding encoder
            batch_losses = {}
            
            for dataset_name, dataset_batch in batch_dict.items():
                if dataset_name not in encoders:
                    continue
                
                batch_size = dataset_batch['batch_size']
                if batch_size == 0:  # Skip empty batches
                    continue
                
                # Get data
                pose_seqs = dataset_batch['pose_seqs'].to(device)
                audios = dataset_batch['audios'].to(device)
                spectrograms = dataset_batch['spectrograms'].to(device)
                
                # Forward pass through corresponding encoder
                encoder = encoders[dataset_name]
                optimizer = optimizers[dataset_name]
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = encoder(pose_seqs, audios, spectrograms)
                
                # Mock loss (replace with your actual loss function)
                # For demo purposes, we'll use a simple MSE loss with random targets
                target = torch.randn_like(embeddings)
                loss = nn.MSELoss()(embeddings, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_losses[dataset_name] = loss.item()
                epoch_losses[dataset_name] += loss.item()
                
                print(f"  Batch {batch_idx}, {dataset_name}: "
                      f"batch_size={batch_size}, "
                      f"pose_shape={pose_seqs.shape}, "
                      f"loss={loss.item():.4f}")
            
            num_batches += 1
            
            # Stop after a few batches for demo
            if batch_idx >= 5:
                break
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for dataset_name, total_loss in epoch_losses.items():
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"  {dataset_name}: avg_loss = {avg_loss:.4f}")
    
    # 5. Inference example
    print("\n5. Inference example...")
    
    # Put encoders in evaluation mode
    for encoder in encoders.values():
        encoder.eval()
    
    with torch.no_grad():
        # Get one batch
        batch_dict = next(iter(dataloader))
        
        print("Inference results:")
        for dataset_name, dataset_batch in batch_dict.items():
            if dataset_name not in encoders or dataset_batch['batch_size'] == 0:
                continue
            
            pose_seqs = dataset_batch['pose_seqs'].to(device)
            audios = dataset_batch['audios'].to(device)
            spectrograms = dataset_batch['spectrograms'].to(device)
            
            # Forward pass
            embeddings = encoders[dataset_name](pose_seqs, audios, spectrograms)
            
            print(f"  {dataset_name}:")
            print(f"    Input shape: {pose_seqs.shape}")
            print(f"    Output embeddings shape: {embeddings.shape}")
            print(f"    Embedding mean: {embeddings.mean().item():.4f}")
            print(f"    Embedding std: {embeddings.std().item():.4f}")
    
    print("\n=== Example Complete ===")
    print("\nKey Features of Balanced Dataset:")
    print("• Each iteration provides one batch from EACH dataset")
    print("• Each dataset gets its own encoder with appropriate dimensions")
    print("• Training is balanced across all datasets")
    print("• Easy to extend to more datasets/encoders")

if __name__ == "__main__":
    main() 