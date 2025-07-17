import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
print(os.getcwd())
import tempfile
import subprocess
import librosa
import soundfile as sf
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from scripts.data_loader.lmdb_data_loader_new import LMDBDataset
from scripts.data_loader.calculate_mean_vector import calculate_mean_dir_vec
import argparse
def visualize_direction_vectors(joint_positions, dir_vectors, connections, frame_idx=3):
    """
    Visualize direction vectors for a given frame
    
    Args:
        joint_positions (numpy.ndarray): Joint positions, shape (frames, joints, 3)
        dir_vectors (numpy.ndarray): Direction vectors, shape (frames, connections, 3)
        connections (list): List of tuples (parent_idx, child_idx, length)
        frame_idx (int): Frame index to visualize
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 8))
    
    # Plot joint positions
    ax1 = fig.add_subplot(121, projection='3d')
    positions = joint_positions[frame_idx]
    
    # Plot joints
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=30)
    
    # Plot connections - maintain coordinate system consistency
    for i, (parent_idx, child_idx, _) in enumerate(connections):
        parent_pos = positions[parent_idx]
        child_pos = positions[child_idx]
        ax1.plot([parent_pos[0], child_pos[0]], 
                [parent_pos[1], child_pos[1]], 
                [parent_pos[2], child_pos[2]], 'r-', linewidth=2)
    
    ax1.set_title('Joint Positions (Original)')
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Plot direction vectors
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Reconstruct skeleton from direction vectors
    reconstructed_positions = np.zeros_like(positions)
    # Root joint (typically joint 0) stays at origin or at its original position
    reconstructed_positions[0] = positions[0]  # Keep the root at its original position
    
    # Reconstruct positions using direction vectors and connections
    for i, (parent_idx, child_idx, length) in enumerate(connections):
        direction = dir_vectors[frame_idx, i] * length
        reconstructed_positions[child_idx] = reconstructed_positions[parent_idx] + direction
    
    # Plot reconstructed joints
    ax2.scatter(reconstructed_positions[:, 0], reconstructed_positions[:, 1], 
                reconstructed_positions[:, 2], c='blue', s=30)
    
    # Plot reconstructed connections
    for i, (parent_idx, child_idx, _) in enumerate(connections):
        parent_pos = reconstructed_positions[parent_idx]
        child_pos = reconstructed_positions[child_idx]
        ax2.plot([parent_pos[0], child_pos[0]], 
                [parent_pos[1], child_pos[1]], 
                [parent_pos[2], child_pos[2]], 'g-', linewidth=2)
    
    ax2.set_title('Reconstructed from Direction Vectors')
    ax2.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Set common properties
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Make axes equal
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(f'direction_vectors_{frame_idx}.png')
    
    
def save_mp4(dir_vectors, connections, output_path='animation.mp4', fps=15, audio_tensor=None):
    """
    Create and save an animation of the skeleton reconstructed from direction vectors
    
    Args:
        dir_vectors (numpy.ndarray): Direction vectors, shape (frames, connections, 3)
        connections (list): List of tuples (parent_idx, child_idx, length)
        output_path (str): Path to save the MP4 file
        fps (int): Frames per second for the animation
        audio_path (str, optional): Path to audio file to add to the animation
        audio_tensor (torch.Tensor, optional): Audio tensor from dataloader
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get number of frames and joints
    n_frames = dir_vectors.shape[0]
    n_joints = max(max(parent, child) for parent, child, _ in connections) + 1
    
    # Initialize skeleton data
    skeleton_data = np.zeros((n_frames, n_joints, 3))
    
    # Reconstruct positions for each frame
    for frame in range(n_frames):
        # Root joint stays at origin
        skeleton_data[frame, 0] = np.zeros(3)
        
        # Reconstruct positions using direction vectors
        for i, (parent_idx, child_idx, length) in enumerate(connections):
            direction = dir_vectors[frame, i] * length
            skeleton_data[frame, child_idx] = skeleton_data[frame, parent_idx] + direction
    
    # Find global min/max for consistent axes
    all_x = skeleton_data[:, :, 0].flatten()
    all_y = skeleton_data[:, :, 1].flatten()
    all_z = skeleton_data[:, :, 2].flatten()
    
    max_range = max(all_x.max() - all_x.min(),
                    all_y.max() - all_y.min(),
                    all_z.max() - all_z.min()) / 2.0
    
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    
    # Plot function for animation
    joint_dots = None
    lines = []
    
    def init():
        nonlocal joint_dots, lines
        # Initialize empty plots
        joint_dots = ax.scatter([], [], [], c='blue', s=30)
        lines = [ax.plot([], [], [], 'g-', linewidth=2)[0] for _ in connections]
        
        # Set axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Skeleton Animation')
        
        return [joint_dots] + lines
    
    def update(frame):
        nonlocal joint_dots, lines
        
        # Update joint positions
        positions = skeleton_data[frame]
        joint_dots._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Update connections
        for i, ((parent_idx, child_idx, _), line) in enumerate(zip(connections, lines)):
            parent_pos = positions[parent_idx]
            child_pos = positions[child_idx]
            line.set_data([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]])
            line.set_3d_properties([parent_pos[2], child_pos[2]])
        
        return [joint_dots] + lines
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, 
                                 init_func=init, blit=True, interval=1000/fps)
    
    # Save animation
    temp_video = output_path
    has_audio = audio_tensor is not None
    
    if has_audio:
        # If audio is provided, first save without audio
        temp_video = tempfile.mktemp('.mp4')
    
    # Save the animation
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='DiffGesture'), bitrate=1800)
    ani.save(temp_video, writer=writer)
    
    # Add audio if provided
    if has_audio:
        # Save temporary audio file
        temp_audio = tempfile.mktemp('.wav')
        
        if audio_tensor is not None:
            # Convert PyTorch tensor to numpy if needed
            import torch
            if isinstance(audio_tensor, torch.Tensor):
                audio_data = audio_tensor.numpy()
            else:
                audio_data = audio_tensor
                
            # Assume sample rate of 16000 Hz (common for speech)
            sr = 16000
            
            # Trim or pad audio to match video duration
            video_duration = n_frames / fps
            audio_duration = len(audio_data) / sr
            
            if audio_duration > video_duration:
                # Trim audio
                audio_data = audio_data[:int(video_duration * sr)]
            elif audio_duration < video_duration:
                # Pad audio with silence
                padding = np.zeros(int((video_duration - audio_duration) * sr))
                audio_data = np.concatenate([audio_data, padding])
                
            # Save audio to temporary file
            sf.write(temp_audio, audio_data, sr)
            
        
        # Combine video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', temp_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        subprocess.call(cmd)
        
        # Clean up temporary files
        os.remove(temp_video)
        os.remove(temp_audio)
    
    plt.close(fig)
    print(f"Animation saved to {output_path}")
def check_files(data_path_list, audio_path_list):
    for data_path in data_path_list:
        if not os.path.exists(data_path):
            print(f"File {data_path} does not exist")
            raise ValueError("File does not exist")
    for audio_path in audio_path_list:
        if not os.path.exists(audio_path):
            print(f"File {audio_path} does not exist")
            raise ValueError("File does not exist")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Trinity/Beat/all")
    args = parser.parse_args()
    #print current directory
    
    if args.dataset == "trinity" or args.dataset == "all":
        train_data_dir = "./data/trinity/allRec"
        train_audio_dir = "./data/trinity/allRecAudio"
        train_npz_files = [f for f in os.listdir(train_data_dir) if f.endswith('_direction_vectors.npz')]
        train_npz_list = [os.path.join(train_data_dir, f) for f in train_npz_files]
        train_audio_list = [os.path.join(train_audio_dir, f.replace('_direction_vectors.npz', '.wav')) for f in train_npz_files]
        test_data_dir = "./data/trinity/allTestMotion"
        test_audio_dir = "./data/trinity/allTestAudio"
        test_npz_files = [f for f in os.listdir(test_data_dir) if f.endswith('_direction_vectors.npz')]
        test_npz_list = [os.path.join(test_data_dir, f) for f in test_npz_files]
        test_audio_list = [os.path.join(test_audio_dir, f.replace('_direction_vectors.npz', '.wav')) for f in test_npz_files]
        check_files(train_npz_list, train_audio_list)
        check_files(test_npz_list, test_audio_list)
        all_npz_list = train_npz_list + test_npz_list
        mean_dir_vec = calculate_mean_dir_vec(all_npz_list, original_fps=60, target_fps=15, n_samples=1000)
        np.save("./data/trinity_all_mean_dir_vec.npy", mean_dir_vec)
        dataset = LMDBDataset(
            dataset_name="trinity_all",
            data_path_list=train_npz_list,
            audio_path_list=train_audio_list,
            n_poses=34,
            original_fps=60,
            target_fps=15,
            mean_dir_vec=mean_dir_vec
        )
        test_dataset = LMDBDataset(
            dataset_name="trinity_all_test",
            data_path_list=test_npz_list,
            audio_path_list=test_audio_list,
            n_poses=34,
            original_fps=60,
            target_fps=15,
            mean_dir_vec=mean_dir_vec
        )
        sample_npz = train_npz_list[0]
    if args.dataset == "beat_all" or args.dataset == "all":
        original_data_dir = "./data/beat_english_v0.2.1"
        train_data_dir = []
        for i in range(1,27):
            train_data_dir.append(original_data_dir + "/"+str(i))
        train_npz_list = []
        train_audio_list = []
        for data_dir in train_data_dir:
            for f in os.listdir(data_dir):
                if f.endswith('_direction_vectors.npz'):
                    train_npz_list.append(os.path.join(data_dir, f))
                    train_audio_list.append(os.path.join(data_dir, f.replace('_direction_vectors.npz', '.wav')))
        test_data_dir = []
        for i in [27,28,29,30]:
            test_data_dir.append(original_data_dir + "/"+str(i))
        test_npz_list = []
        test_audio_list = []
        for data_dir in test_data_dir:
            for f in os.listdir(data_dir):
                if f.endswith('_direction_vectors.npz'):
                    test_npz_list.append(os.path.join(data_dir, f))
                    test_audio_list.append(os.path.join(data_dir, f.replace('_direction_vectors.npz', '.wav')))
        check_files(train_npz_list, train_audio_list)
        check_files(test_npz_list, test_audio_list)
        all_npz_list = train_npz_list + test_npz_list
        mean_dir_vec = calculate_mean_dir_vec(all_npz_list, original_fps=120, target_fps=15, n_samples=1500)
        np.save("./data/beat_all_mean_dir_vec.npy", mean_dir_vec)
        dataset = LMDBDataset(
            dataset_name="beat_all",
            data_path_list=train_npz_list,
            audio_path_list=train_audio_list,
            n_poses=34,
            original_fps=120,
            target_fps=15,
            mean_dir_vec=mean_dir_vec
        )
        test_dataset = LMDBDataset(
            dataset_name="beat_all_test",
            data_path_list=test_npz_list,
            audio_path_list=test_audio_list,
            n_poses=34,
            original_fps=120,
            target_fps=15,
            mean_dir_vec=mean_dir_vec
        )
        sample_npz = train_npz_list[0]
    if args.dataset == "beat_warmup" or args.dataset == "all":
        original_data_dir = "./data/beat_english_v0.2.1"
        train_data_dir = []
        for i in range(12,20):
            train_data_dir.append(original_data_dir + "/"+str(i))
        train_npz_list = []
        train_audio_list = []
        for data_dir in train_data_dir:
            for f in os.listdir(data_dir):
                if f.endswith('_direction_vectors.npz'):
                    train_npz_list.append(os.path.join(data_dir, f))
                    train_audio_list.append(os.path.join(data_dir, f.replace('_direction_vectors.npz', '.wav')))
        
        check_files(train_npz_list, train_audio_list)
        mean_dir_vec = np.load("./data/beat_all_mean_dir_vec.npy")
        np.save("./data/beat_all_mean_dir_vec.npy", mean_dir_vec)
        dataset = LMDBDataset(
            dataset_name="beat_warmup",
            data_path_list=train_npz_list,
            audio_path_list=train_audio_list,
            n_poses=34,
            original_fps=120,
            target_fps=15,
            mean_dir_vec=mean_dir_vec
        )
        sample_npz = train_npz_list[0]
    
    if args.dataset == "beat_separated" or args.dataset == "all":
        for train_data_dir in os.listdir("./data/beat_english_v0.2.1"):
            if not train_data_dir.isdigit():
                continue
            dataset_name = str(int(train_data_dir))
            train_data_dir = os.path.join("./data/beat_english_v0.2.1", train_data_dir)
            train_npz_files = [f for f in os.listdir(train_data_dir) if f.endswith('_direction_vectors.npz')]
            train_npz_list = [os.path.join(train_data_dir, f) for f in train_npz_files]
            train_audio_list = [f.replace('_direction_vectors.npz', '.wav') for f in train_npz_list]
            check_files(train_npz_list, train_audio_list)
            mean_dir_vec = np.load("./data/beat_all_mean_dir_vec.npy")
            dataset = LMDBDataset(
                dataset_name=dataset_name,
                data_path_list=train_npz_list,
                audio_path_list=train_audio_list,
                n_poses=34,
                original_fps=120,
                target_fps=15,
                mean_dir_vec=mean_dir_vec
            )
        sample_npz = train_npz_list[30]
    # elif args.dataset == "quest" or args.dataset == "all":
    #     dataset = LMDBDataset(
    #         data_dir="./data/quest",
    #         audio_dir="./data/quest",
    #         n_poses=34,
    #         original_fps=70,
    #         target_fps=15
    #     )
    # Load connections from a sample NPZ file
    data = np.load(sample_npz, allow_pickle=True)
    connections = data['connections'].tolist()  # Convert to Python list if it's a NumPy array
    #connection covert  to int
    connections = [(int(parent_idx), int(child_idx), length) for parent_idx, child_idx, length in connections]
    
    joint_positions = data['joint_positions']
    original_dir_vectors = data['direction_vectors']
    print(f"original_dir_vectors shape: {original_dir_vectors.shape}")
    # Print information about the connections
    print(f"Loaded {len(connections)} connections")
    if len(connections) > 0:
        print(f"First connection: {connections[0]}")
    
    # Get sample from dataset
    sample_idx = 4000
    sample = dataset[sample_idx]
    
    # Print info about the sample
    word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
    print(f"Sample pose shape: {pose_seq.shape}")
    print(f"Sample vec_seq shape: {vec_seq.shape}")
    print(f"Sample audio shape: {audio.shape}")
    print(f"Aux info: {aux_info}")
    vec_seq = vec_seq.numpy() + mean_dir_vec
    print(f"vec_seq shape: {vec_seq.shape}")
    vec_seq = vec_seq.reshape(vec_seq.shape[0], -1, 3)
    print(f"vec_seq shape: {vec_seq.shape}")
    # Visualize the reconstructed skeleton
    visualize_direction_vectors(joint_positions, vec_seq, connections, frame_idx=4)
    save_mp4(vec_seq, connections, output_path='animationlmdb.mp4', fps=15, audio_tensor=audio)
    
if __name__ == "__main__":
    main()