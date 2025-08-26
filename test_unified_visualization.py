#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import librosa
import librosa.display

# Add the scripts directory to Python path
sys.path.append('scripts')

# Import the unified dataloader
from scripts.data_loader.unified_data_loader import (
    MultiDataLoaderWrapper,
    create_multi_dataloader,
)

# Import video saver and expressive connections
from scripts.beat_visualizer import save_mp4
from scripts.utils.data_utils_expressive import dir_vec_pairs as expressive_dir_vec_pairs

def visualize_audio_data(audio_data, spectrogram_data, sample_rate=16000, target_fps=15, n_poses=34):
    """
    Create audio visualizations including waveform and spectrogram
    
    Args:
        audio_data: Raw audio tensor [audio_length]
        spectrogram_data: Mel spectrogram tensor [mel_bins, time_frames]
        sample_rate: Audio sample rate
        target_fps: Target FPS for pose data
        n_poses: Number of poses in sequence
    
    Returns:
        fig: matplotlib figure with audio visualizations
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Convert to numpy if tensor
    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = np.array(audio_data)
    
    if isinstance(spectrogram_data, torch.Tensor):
        spec_np = spectrogram_data.cpu().numpy()
    else:
        spec_np = np.array(spectrogram_data)
    
    # Calculate time duration
    duration = n_poses / target_fps
    time_audio = np.linspace(0, duration, len(audio_np))
    
    # Plot waveform
    axes[0].plot(time_audio, audio_np, color='blue', alpha=0.7)
    axes[0].set_title('Audio Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot spectrogram
    if len(spec_np.shape) == 2:
        time_spec = np.linspace(0, duration, spec_np.shape[1])
        
        # Display mel spectrogram
        im = axes[1].imshow(
            spec_np, 
            aspect='auto', 
            origin='lower',
            extent=[0, duration, 0, spec_np.shape[0]],
            cmap='viridis'
        )
        axes[1].set_title('Mel Spectrogram')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mel Bins')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], label='Magnitude')
    
    plt.tight_layout()
    return fig

def visualize_pose_sequence(pose_data, vec_data, dataset_name, sample_idx=0):
    """
    Create pose sequence visualization
    
    Args:
        pose_data: Pose tensor [batch_size, n_poses, pose_dim] or [batch_size, n_poses, n_joints, 3]
        vec_data: Direction vector tensor [batch_size, n_poses, pose_dim] 
        dataset_name: Name of the dataset
        sample_idx: Index of sample to visualize
    
    Returns:
        fig: matplotlib figure with pose visualization
    """
    # Convert to numpy if tensor
    if isinstance(pose_data, torch.Tensor):
        poses = pose_data[sample_idx].cpu().numpy()
    else:
        poses = np.array(pose_data[sample_idx])
    
    if isinstance(vec_data, torch.Tensor):
        vecs = vec_data[sample_idx].cpu().numpy()
    else:
        vecs = np.array(vec_data[sample_idx])
    
    # Handle different pose data formats
    if len(poses.shape) == 3:
        # Format: [n_poses, n_joints, 3] - Trinity dataset
        n_poses, n_joints, _ = poses.shape
        pose_dim = n_joints * 3
        # Flatten to [n_poses, pose_dim] for consistency
        poses_flat = poses.reshape(n_poses, -1)
    elif len(poses.shape) == 2:
        # Format: [n_poses, pose_dim] - BEAT/TED Expressive
        n_poses, pose_dim = poses.shape
        poses_flat = poses
        n_joints = pose_dim // 3
    else:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")
    
    # Create subplot grid showing motion over time
    n_frames_to_show = min(9, n_poses)
    frame_indices = np.linspace(0, n_poses-1, n_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{dataset_name} - Pose Sequence Visualization (Sample {sample_idx})', fontsize=16)
    
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        
        # Get pose for this frame
        if len(poses.shape) == 3:
            # Trinity format: [n_poses, n_joints, 3]
            frame_pose_3d = poses[frame_idx]  # [n_joints, 3]
            joints_3d = frame_pose_3d
        else:
            # BEAT/TED format: [n_poses, pose_dim]
            frame_pose = poses_flat[frame_idx]
            # Reshape to [n_joints, 3] for 3D coordinates
            joints_3d = frame_pose.reshape(n_joints, 3)
        
        # Color by dataset
        color_map = {'trinity': 'blue', 'beat': 'red', 'ted_expressive': 'green'}
        color = color_map.get(dataset_name, 'black')
        
        # Create simple stick figure (this is simplified, real skeleton needs proper connections)
        ax.scatter(joints_3d[:, 0], joints_3d[:, 1], c=color, s=20, alpha=0.7)
        
        # Connect some joints (simplified)
        if n_joints >= 10:
            connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9)]
            for conn in connections:
                if conn[0] < n_joints and conn[1] < n_joints:
                    ax.plot([joints_3d[conn[0], 0], joints_3d[conn[1], 0]], 
                           [joints_3d[conn[0], 1], joints_3d[conn[1], 1]], 
                           color=color, alpha=0.6)
        
        # Set equal aspect and clean up
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame_idx}/{n_poses-1}')
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(len(frame_indices), 9):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def visualize_unified_batch(loader, output_dir="./unified_visualizations"):
    """
    Visualize one batch from the unified dataset loader
    
    Args:
        loader: MultiDataLoaderWrapper instance
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("VISUALIZING UNIFIED DATASET BATCH")
    print("=" * 60)
    
    # Get one batch
    batch_dict = loader.get_batch()
    
    # Process each dataset in the batch
    for dataset_name, batch_data in batch_dict.items():
        print(f"\nProcessing {dataset_name} dataset:")
        print(f"  Batch size: {batch_data['batch_size']}")
        
        # Extract data
        pose_seqs = batch_data['pose_seqs']
        vec_seqs = batch_data['vec_seqs'] 
        audios = batch_data['audios']
        spectrograms = batch_data['spectrograms']
        aux_infos = batch_data['aux_infos']
        
        print(f"  Pose shape: {pose_seqs.shape}")
        print(f"  Vec shape: {vec_seqs.shape}")
        print(f"  Audio shape: {audios.shape}")
        print(f"  Spectrogram shape: {spectrograms.shape}")
        
        # Visualize first sample from this dataset
        sample_idx = 0
        if batch_data['batch_size'] > 0:
            # Create pose visualization
            print(f"  Creating pose visualization for sample {sample_idx}...")
            pose_fig = visualize_pose_sequence(
                pose_seqs, vec_seqs, dataset_name, sample_idx
            )
            pose_output = os.path.join(output_dir, f"{dataset_name}_poses_sample_{sample_idx}.png")
            pose_fig.savefig(pose_output, dpi=300, bbox_inches='tight')
            plt.close(pose_fig)
            print(f"  Saved pose visualization: {pose_output}")
            
            # Create audio visualization  
            print(f"  Creating audio visualization for sample {sample_idx}...")
            audio_fig = visualize_audio_data(
                audios[sample_idx], 
                spectrograms[sample_idx],
                sample_rate=16000,
                target_fps=loader.target_fps,
                n_poses=loader.n_poses
            )
            audio_output = os.path.join(output_dir, f"{dataset_name}_audio_sample_{sample_idx}.png")
            audio_fig.savefig(audio_output, dpi=300, bbox_inches='tight')
            plt.close(audio_fig)
            print(f"  Saved audio visualization: {audio_output}")
            
            # Create combined visualization
            print(f"  Creating combined visualization for sample {sample_idx}...")
            fig_combined = create_combined_visualization(
                pose_seqs[sample_idx], 
                vec_seqs[sample_idx],
                audios[sample_idx], 
                spectrograms[sample_idx],
                dataset_name,
                sample_idx,
                loader.target_fps,
                loader.n_poses
            )
            combined_output = os.path.join(output_dir, f"{dataset_name}_combined_sample_{sample_idx}.png")
            fig_combined.savefig(combined_output, dpi=300, bbox_inches='tight')
            plt.close(fig_combined)
            print(f"  Saved combined visualization: {combined_output}")

def create_combined_visualization(pose_data, vec_data, audio_data, spectrogram_data, 
                                dataset_name, sample_idx, target_fps, n_poses):
    """
    Create a combined visualization showing poses and audio for one sample
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{dataset_name} - Combined Visualization (Sample {sample_idx})', fontsize=16)
    
    # Create a 2x3 grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Convert tensors to numpy
    if isinstance(pose_data, torch.Tensor):
        poses = pose_data.cpu().numpy()
    else:
        poses = np.array(pose_data)
        
    if isinstance(audio_data, torch.Tensor):
        audio = audio_data.cpu().numpy()
    else:
        audio = np.array(audio_data)
        
    if isinstance(spectrogram_data, torch.Tensor):
        spec = spectrogram_data.cpu().numpy()
    else:
        spec = np.array(spectrogram_data)
    
    # Handle different pose data formats
    if len(poses.shape) == 3:
        # Trinity format: [n_poses, n_joints, 3]
        n_pose_frames, n_joints, _ = poses.shape
        pose_dim = n_joints * 3
        # Flatten for trajectory analysis
        poses_flat = poses.reshape(n_pose_frames, -1)
    elif len(poses.shape) == 2:
        # BEAT/TED format: [n_poses, pose_dim]
        n_pose_frames, pose_dim = poses.shape
        poses_flat = poses
        n_joints = pose_dim // 3
    else:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")
    
    # Audio waveform (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    duration = n_poses / target_fps
    time_audio = np.linspace(0, duration, len(audio))
    ax1.plot(time_audio, audio, color='blue', alpha=0.7)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(spec.shape) == 2:
        im = ax2.imshow(spec, aspect='auto', origin='lower', 
                       extent=[0, duration, 0, spec.shape[0]], cmap='viridis')
        ax2.set_title('Mel Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mel Bins')
        plt.colorbar(im, ax=ax2, label='Magnitude')
    
    # Pose trajectory over time (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Show movement of a few key joints over time
    if pose_dim >= 9:  # At least 3 joints in 3D
        if len(poses.shape) == 3:
            # Trinity format: already in [n_poses, n_joints, 3]
            joints_3d = poses
        else:
            # BEAT/TED format: reshape from flattened
            joints_3d = poses_flat.reshape(n_pose_frames, n_joints, 3)
        
        # Plot trajectory of first few joints
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for j in range(min(5, n_joints)):
            ax3.plot(joints_3d[:, j, 0], joints_3d[:, j, 1], 
                    color=colors[j % len(colors)], alpha=0.7, 
                    label=f'Joint {j}', linewidth=2)
        ax3.set_title('Joint Trajectories')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Show 4 key poses from the sequence (bottom row)
    n_frames_to_show = 4
    frame_indices = np.linspace(0, n_pose_frames-1, n_frames_to_show, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[1, i % 3] if i < 3 else gs[1, 2])
        
        if i >= 3:  # Overlay on the last subplot
            ax.clear()
        
        # Get joints for this frame
        if len(poses.shape) == 3:
            # Trinity format: [n_poses, n_joints, 3]
            joints_3d = poses[frame_idx]  # [n_joints, 3]
        else:
            # BEAT/TED format: [n_poses, pose_dim]
            frame_pose = poses_flat[frame_idx]
            joints_3d = frame_pose.reshape(n_joints, 3)
        
        # Color by dataset
        color_map = {'trinity': 'blue', 'beat': 'red', 'ted_expressive': 'green'}
        color = color_map.get(dataset_name, 'black')
        
        ax.scatter(joints_3d[:, 0], joints_3d[:, 1], c=color, s=30, alpha=0.8)
        
        # Add simple connections for visualization
        if n_joints >= 6:
            connections = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)]
            for conn in connections:
                if conn[0] < n_joints and conn[1] < n_joints:
                    ax.plot([joints_3d[conn[0], 0], joints_3d[conn[1], 0]], 
                           [joints_3d[conn[0], 1], joints_3d[conn[1], 1]], 
                           color=color, alpha=0.6)
        
        ax.set_title(f'Pose Frame {frame_idx}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    return fig

def _load_connections_for_dataset(dataset_name, trinity_npz=None, beat_npz=None):
    """
    Build connections list [(parent_idx, child_idx, length), ...] for each dataset.
    - ted_expressive: use expressive_dir_vec_pairs
    - trinity/beat: load from provided NPZ file if given; else return None
    """
    if dataset_name == 'ted_expressive':
        # expressive_dir_vec_pairs is list of (parent, child, length)
        # Drop (38,39), (38,40), (39,41), (40,42)
        drop = {(38, 39), (38, 40), (39, 41), (40, 42)}
        kept = [(int(a), int(b), float(l)) for (a, b, l) in expressive_dir_vec_pairs if (a, b) not in drop]
        return kept
    
    if dataset_name == 'trinity' and trinity_npz and os.path.exists(trinity_npz):
        data = np.load(trinity_npz, allow_pickle=True)
        conns = data['connections'].tolist()
        conns = [(int(p), int(c), float(l)) for (p, c, l) in conns]
        return conns
    
    if dataset_name == 'beat' and beat_npz and os.path.exists(beat_npz):
        data = np.load(beat_npz, allow_pickle=True)
        conns = data['connections'].tolist()
        conns = [(int(p), int(c), float(l)) for (p, c, l) in conns]
        return conns
    
    return None


def _save_video_for_sample(dataset_name, pose_seqs, vec_seqs, audios, fps, output_dir, sample_idx,
                           connections, pose_dim=None, mean_dir_vec=None):
    """
    Save MP4 for one sample using direction vectors and provided connections.
    vec_seqs: [B, T, pose_dim]
    connections: list of (parent, child, length)
    mean_dir_vec: optional mean direction vector to add back (unnormalize)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select sample
    vec = vec_seqs[sample_idx]
    audio = audios[sample_idx]
    
    # Ensure numpy
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().numpy()
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = np.array(audio)
    
    # vec shape [T, pose_dim]. We need [T, n_connections, 3]
    if vec.ndim != 2:
        raise ValueError(f"Expected vec shape [T, pose_dim], got {vec.shape}")
    
    if connections is None:
        print(f"  ! No connections for {dataset_name}. Skipping video.")
        return None
    
    n_connections = len(connections)
    # pose_dim might equal n_connections*3; if not provided, infer
    if pose_dim is None:
        pose_dim = vec.shape[1]
    
    if pose_dim != n_connections * 3:
        # Try to reshape anyway by inferring connections count
        if vec.shape[1] % 3 == 0:
            n_connections_in_vec = vec.shape[1] // 3
            if n_connections_in_vec != n_connections:
                print(f"  ! Mismatch: vec has {n_connections_in_vec} connections, provided {n_connections}. Using vec count.")
                n_connections = n_connections_in_vec
                connections = connections[:n_connections]
        else:
            raise ValueError(f"pose_dim {pose_dim} not divisible by 3")
    
    # Add back mean if provided (unnormalize)
    if mean_dir_vec is not None:
        if mean_dir_vec.shape[0] != pose_dim:
            print(f"  ! Mean vec shape {mean_dir_vec.shape} doesn't match pose_dim {pose_dim}")
        else:
            print(f"  + Adding back mean direction vector")
            vec = vec + mean_dir_vec  # broadcast over time dimension
    
        dir_vectors = vec.reshape(vec.shape[0], n_connections, 3)
 
     # Output path
    output_path = os.path.join(output_dir, f"{dataset_name}_sample_{sample_idx}.mp4")
    
    # Save video with audio
    save_mp4(dir_vectors=dir_vectors, connections=connections, output_path=output_path, fps=fps, audio_tensor=audio_np)
    print(f"  ✓ Saved video: {output_path}")
    return output_path

def _load_expressive_mean_from_config(config_path: str):
    """Load mean_dir_vec (list of floats) from a YAML config without requiring PyYAML.
    Assumes mean_dir_vec is on a single line with brackets, as in the provided config.
    Returns a numpy array or None if not found.
    """
    try:
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip().startswith('mean_dir_vec:'):
                    # Extract bracket content
                    start = line.find('[')
                    end = line.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        content = line[start+1:end]
                        # Split by commas and parse floats
                        parts = [p.strip() for p in content.split(',') if p.strip()]
                        import numpy as np
                        vec = np.array([float(x) for x in parts], dtype=np.float32)
                        return vec
        return None
    except Exception:
        return None

# New helper: adjust TED expressive mean to target pose_dim (e.g., 114) by pruning connections once
def _adjust_ted_expressive_mean(mean_vec: np.ndarray, target_pose_dim: int):
    """Return a mean direction vector adjusted to TED expressive target pose_dim.
    If the input is 126 and target is 114, drop the pruned connections.
    Otherwise, return unchanged.
    """
    if mean_vec is None:
        return None
    if mean_vec.ndim != 1:
        mean_vec = mean_vec.reshape(-1)
    if target_pose_dim is None:
        return mean_vec
    if mean_vec.size == target_pose_dim:
        return mean_vec
    if mean_vec.size in (114, 126) and target_pose_dim in (114, 126):
        # Handle pruning 126 -> 114 using expressive_dir_vec_pairs and drop set
        if mean_vec.size == 126 and target_pose_dim == 114:
            drop = {(38, 39), (38, 40), (39, 41), (40, 42)}
            keep_indices = [i for i, (a, b, _) in enumerate(expressive_dir_vec_pairs) if (a, b) not in drop]
            try:
                pruned = mean_vec.reshape(-1, 3)[keep_indices].reshape(-1)
                return pruned
            except Exception:
                return mean_vec
    return mean_vec

def main():
    parser = argparse.ArgumentParser(description='Visualize unified dataset batch with audio')
    parser.add_argument('--trinity_path', type=str,
                        default="./data/trinity_all_cache",
                        help='Path to Trinity dataset cache')
    parser.add_argument('--beat_path', type=str,
                        default="./data/beat_english_v0.2.1/beat_all_cache",
                        help='Path to BEAT dataset cache')
    parser.add_argument('--ted_expressive_path', type=str,
                        default="./data/ted_expressive_dataset/train",
                        help='Path to TED Expressive dataset')
    parser.add_argument('--batch_per_dataset', type=int, default=4,
                        help='Samples per dataset per training step')
    parser.add_argument('--output_dir', type=str, default="./unified_visualizations",
                        help='Output directory for visualizations')
    parser.add_argument('--trinity_npz', type=str, default="./data/Recording_001_direction_vectors.npz",
                        help='NPZ file containing connections for Trinity (direction_vectors.npz)')
    parser.add_argument('--beat_npz', type=str, default="./data/1_wayne_0_100_100_direction_vectors.npz",
                        help='NPZ file containing connections for BEAT (direction_vectors.npz)')
    parser.add_argument('--trinity_mean', type=str, default="./data/trinity_all_mean_dir_vec.npy",
                        help='Mean direction vector file for Trinity')
    parser.add_argument('--beat_mean', type=str, default="./data/beat_all_mean_dir_vec.npy", 
                        help='Mean direction vector file for BEAT')
    parser.add_argument('--expressive_mean', type=str, default=None,
                        help='Mean direction vector file for TED Expressive (optional)')
    parser.add_argument('--expressive_config', type=str, default="./config/pose_diffusion_expressive.yml",
                        help='Config file containing TED Expressive mean_dir_vec')
    parser.add_argument('--expressive_mean_out', type=str, default="./data/ted_expressive_mean_dir_vec.npy",
                        help='Output path to save the 114-dim TED Expressive mean vector')
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFIED DATASET VISUALIZATION TEST")
    print("=" * 60)
    
    # Check available datasets
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
    
    # Create unified loader
    try:
        loader = create_multi_dataloader(
            trinity_path=args.trinity_path,
            beat_path=args.beat_path,
            ted_expressive_path=args.ted_expressive_path,
            samples_per_dataset_per_step=args.batch_per_dataset,
            target_fps=15,
            n_poses=34,
        )
        
        info = loader.info()
        print("\nDatasets loaded:")
        for name, meta in info["datasets"].items():
            print(f"  - {name}: size={meta['size']}, pose_dim={meta['pose_dim']}, batch_size={meta['batch_size']}")
        
        # Preload mean vectors once, validated/sliced against target pose_dim
        mean_vectors = {}
        for name, meta in info["datasets"].items():
            pose_dim_target = meta.get('pose_dim')
            mean_vec = None
            if name == 'trinity' and args.trinity_mean and os.path.exists(args.trinity_mean):
                mean_vec = np.load(args.trinity_mean)
                print(f"  + Loaded Trinity mean vector: {mean_vec.shape}")
            elif name == 'beat' and args.beat_mean and os.path.exists(args.beat_mean):
                mean_vec = np.load(args.beat_mean)
                print(f"  + Loaded BEAT mean vector: {mean_vec.shape}")
            elif name == 'ted_expressive':
                # Prefer explicit file; else fallback to config
                if args.expressive_mean and os.path.exists(args.expressive_mean):
                    mean_vec = np.load(args.expressive_mean)
                    print(f"  + Loaded TED Expressive mean vector: {mean_vec.shape}")
                elif args.expressive_config:
                    mean_vec = _load_expressive_mean_from_config(args.expressive_config)
                    if mean_vec is not None:
                        print(f"  + Loaded TED Expressive mean vector from config: {mean_vec.shape}")
                    else:
                        print("  ! Could not load TED Expressive mean from config; proceeding without it")
                # Adjust to match target pose_dim (e.g., 114)
                if mean_vec is not None:
                    adjusted = _adjust_ted_expressive_mean(mean_vec, pose_dim_target)
                    if adjusted is not None and adjusted.size != mean_vec.size:
                        print(f"  + Sliced TED Expressive mean to match pose_dim {pose_dim_target}: {adjusted.shape}")
                    mean_vec = adjusted
                    # Persist 114-dim mean to file for future training parity with trinity/beat
                    if args.expressive_mean_out and isinstance(args.expressive_mean_out, str):
                        out_dir = os.path.dirname(args.expressive_mean_out)
                        if out_dir:
                            os.makedirs(out_dir, exist_ok=True)
                        try:
                            np.save(args.expressive_mean_out, mean_vec)
                            print(f"  ✓ Saved TED Expressive mean to: {args.expressive_mean_out}")
                        except Exception as e:
                            print(f"  ! Failed to save TED Expressive mean: {e}")
            mean_vectors[name] = mean_vec
        
        # Create one MP4 per dataset for sample 0 using correct connections
        batch = loader.get_batch()
        for name, data in batch.items():
            print(f"\nCreating video for {name}...")
            conns = _load_connections_for_dataset(
                name,
                trinity_npz=args.trinity_npz,
                beat_npz=args.beat_npz,
            )
            if conns is None:
                print(f"  ! No connections provided for {name}. Provide --{name}_npz to enable video.")
                continue
            
            _save_video_for_sample(
                dataset_name=name,
                pose_seqs=data['pose_seqs'],
                vec_seqs=data['vec_seqs'],
                audios=data['audios'],
                fps=loader.target_fps,
                output_dir=args.output_dir,
                sample_idx=2,
                connections=conns,
                pose_dim=data['vec_seqs'].shape[-1] if hasattr(data['vec_seqs'], 'shape') else None,
                mean_dir_vec=mean_vectors.get(name),
            )
        
        print(f"\n✓ All videos saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"✗ Error creating unified loader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 