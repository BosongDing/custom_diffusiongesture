import os
import numpy as np
from scripts.data_loader.lmdb_data_loader_new import *

def calculate_mean_dir_vec(npz_files, original_fps=60, target_fps=15, n_samples=1000):
    """
    Calculate the mean direction vector from a subset of the data
    
    Args:
        data_dir: Directory containing NPZ files
        original_fps: Original frame rate
        target_fps: Target frame rate
        n_samples: Number of samples to use
        
    Returns:
        numpy.ndarray: Mean direction vector
    """
    print(f"Found {len(npz_files)} NPZ files")
    # Randomly sample files if there are many
    import random
    if len(npz_files) > n_samples:
        npz_files = random.sample(npz_files, n_samples)
    
    all_vectors = []
    
    for npz_file in npz_files: 
        data = np.load(npz_file)
        dir_vectors = data['direction_vectors']
        
        # Resample to target fps
        resampled_vectors = resample_pose_seq(dir_vectors, original_fps, target_fps)
        
        # Flatten the vectors
        flat_vectors = resampled_vectors.reshape(resampled_vectors.shape[0], -1)
        
        all_vectors.append(flat_vectors)
    
    # Concatenate all vectors
    all_vectors = np.concatenate(all_vectors, axis=0)
    
    # Calculate mean
    mean_vec = np.mean(all_vectors, axis=0)
    
    return mean_vec

if __name__ == "__main__":
    data_dir = '/home/bsd/cospeech/DiffGesture/data/trinity/allRec'
    data_dir2 = '/home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion'
    mean_dir_vec = calculate_mean_dir_vec(data_dir,data_dir2)
    print("Mean direction vector shape:", mean_dir_vec.shape)
    
    # Save the mean vector
    np.save('/home/bsd/cospeech/DiffGesture/data/trinity/mean_dir_vec.npy', mean_dir_vec)
    print("Mean direction vector saved")
    
    # Test dataset with mean vector
    dataset = TrinityDataset(
        data_dir=data_dir,
        audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRecAudio',
        n_poses=34,
        mean_dir_vec=mean_dir_vec
    )
    
    # Get a sample
    sample = dataset[10]
    _, _, pose_seq, _, _, _, _ = sample
    print("Normalized pose sequence shape:", pose_seq.shape)
    print("Min value:", pose_seq.min().item())
    print("Max value:", pose_seq.max().item())
    print("Mean value:", pose_seq.mean().item())