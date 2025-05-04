import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
from scipy.spatial.transform import Rotation as R
import argparse
from BVH_convert import parse_bvh

def create_skeleton_animation(bvh_data, output_video_path, audio_path=None, fps=30, dpi=100, duration=None):
    """
    Create an animation of the skeleton from BVH data and combine with audio
    
    Args:
        bvh_data (dict): BVH data from parse_bvh function
        output_video_path (str): Path to save the output video
        audio_path (str): Path to the audio file to combine with the video
        fps (int): Frames per second for the animation
        dpi (int): DPI for the output video
        duration (float): Duration of the animation in seconds (None for full animation)
    """
    # Extract required data
    joint_data = bvh_data['joint_data']
    parent_dict = bvh_data['parent_dict']
    joint_order = bvh_data['joint_order']
    motion_data = bvh_data['motion_data']
    frame_time = bvh_data['frame_time']
    
    # Calculate actual fps from BVH if not specified
    bvh_fps = 1.0 / frame_time
    if fps is None:
        fps = bvh_fps
    
    # Determine number of frames to animate
    if duration is None:
        num_frames = len(motion_data)
    else:
        num_frames = min(int(duration * fps), len(motion_data))
    
    # Set up channel indices for each joint
    channel_indices = {}
    index = 0
    for joint_name in joint_order:
        if joint_name not in joint_data:
            continue
        num_channels = len(joint_data[joint_name]['channels'])
        channel_indices[joint_name] = (index, index + num_channels)
        index += num_channels
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to calculate joint positions for a given frame
    def calculate_frame_positions(frame_idx):
        frame_data = motion_data[frame_idx]
        joint_positions = {}
        orientations = {}  # Store world orientation matrices
        
        # Helper function to calculate global position and rotation
        def calculate_global_transform(joint_name):
            if joint_name in joint_positions:
                return joint_positions[joint_name], orientations[joint_name]
            
            # Get joint properties
            parent_name = parent_dict[joint_name]
            offset = np.array(joint_data[joint_name]['offset'])
            
            # Get this joint's channels and values
            start_idx, end_idx = channel_indices.get(joint_name, (0, 0))
            channel_values = frame_data[start_idx:end_idx] if start_idx < len(frame_data) else []
            channels = joint_data[joint_name]['channels']
            
            # Initialize local transform
            translation = np.zeros(3)
            rotation_angles_deg = []
            rotation_order_channels = []
            
            # Extract translation and rotation values
            for i, channel in enumerate(channels):
                if i >= len(channel_values):
                    continue
                
                value = channel_values[i]
                channel_upper = channel.upper()
                
                if 'POSITION' in channel_upper:
                    if 'X' in channel_upper: translation[0] = value
                    if 'Y' in channel_upper: translation[1] = value
                    if 'Z' in channel_upper: translation[2] = value
                elif 'ROTATION' in channel_upper:
                    rotation_angles_deg.append(value)
                    rotation_order_channels.append(channel)
            
            # Convert rotation values from degrees to radians
            rotation_angles_rad = np.radians(rotation_angles_deg)
            
            # Create rotation matrix using correct order
            local_rot_mat = np.identity(3)
            if rotation_angles_deg:
                order_str = "".join(ch[0] for ch in rotation_order_channels).upper()
                try:
                    local_rot_mat = R.from_euler(order_str, rotation_angles_rad, degrees=False).as_matrix()
                except ValueError as e:
                    print(f"Error creating rotation for joint '{joint_name}': {e}")
            
            # Calculate global transform
            if parent_name is None:  # Root joint
                global_pos = translation
                global_rot_mat = local_rot_mat
            else:
                # Get parent transformation (recursive call)
                parent_pos, parent_rot_mat = calculate_global_transform(parent_name)
                
                # Apply parent rotation to joint offset
                rotated_offset = parent_rot_mat @ offset
                
                # Global position is parent position plus rotated offset
                global_pos = parent_pos + rotated_offset
                
                # Global rotation is parent rotation times local rotation
                global_rot_mat = parent_rot_mat @ local_rot_mat
            
            # Store and return global transformation
            joint_positions[joint_name] = global_pos
            orientations[joint_name] = global_rot_mat
            return global_pos, global_rot_mat
        
        # Calculate positions for all joints
        positions = []
        labels = []
        
        for joint_name in joint_order:
            if joint_name not in joint_data:
                continue
            global_pos, _ = calculate_global_transform(joint_name)
            positions.append(global_pos)
            labels.append(joint_name)
        
        return np.array(positions), labels
    
    # Calculate positions for first frame to set up plot limits
    positions, labels = calculate_frame_positions(0)
    
    # Ensure skeleton is in standing position by aligning with Y-axis
    # Find the height range of the skeleton
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    height = y_max - y_min
    
    # Function to update the animation for each frame
    def update(frame_idx):
        ax.clear()
        
        # Calculate positions for the current frame
        positions, labels = calculate_frame_positions(frame_idx)
        
        # Ensure skeleton is in standing position
        # Normalize positions to make skeleton stand upright along Y-axis
        positions[:, 1] = positions[:, 1] - y_min  # Shift to start at 0
        
        # Plot joints
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=30)
        
        # Plot connections between joints
        for joint_name in joint_order:
            if joint_name not in joint_data:
                continue
            parent_name = parent_dict[joint_name]
            if parent_name is not None and parent_name in joint_order:
                try:
                    parent_idx = joint_order.index(parent_name)
                    joint_idx = joint_order.index(joint_name)
                    
                    parent_pos = positions[parent_idx]
                    child_pos = positions[joint_idx]
                    
                    ax.plot([parent_pos[0], child_pos[0]], 
                            [parent_pos[1], child_pos[1]], 
                            [parent_pos[2], child_pos[2]], 'r-', linewidth=2)
                except ValueError:
                    # Skip if parent or joint not in the list
                    pass
        
        # Set axis properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set consistent view limits
        max_range = max(
            positions[:, 0].max() - positions[:, 0].min(),
            height,
            positions[:, 2].max() - positions[:, 2].min()
        ) * 0.6
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = height * 0.5  # Center in the middle of the height
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(0, height)  # Start from ground (0)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set title with frame information
        ax.set_title(f'Frame {frame_idx} / {num_frames-1}')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return ax
    
    # Create animation
    print(f"Creating animation with {num_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)
    
    # Save animation as video without audio
    temp_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
    anim.save(temp_video_path, writer=writer, dpi=dpi)
    
    # If audio is provided, combine video with audio
    if audio_path and os.path.exists(audio_path):
        print(f"Combining video with audio from {audio_path}...")
        # Calculate video duration
        video_duration = num_frames / fps
        
        # Use FFmpeg to combine video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',  # End when the shortest input stream ends
            output_video_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Remove temporary video
        os.remove(temp_video_path)
        print(f"Video with audio saved to {output_video_path}")
    else:
        # If no audio, just rename the temp video
        os.rename(temp_video_path, output_video_path)
        print(f"Video saved to {output_video_path}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Create a skeleton animation from BVH file with audio')
    parser.add_argument('--bvh_file', help='Path to the BVH file')
    parser.add_argument('--audio', help='Path to the audio file to combine with the video')
    parser.add_argument('--output', help='Path to save the output video', default='skeleton_animation.mp4')
    parser.add_argument('--fps', type=int, help='Frames per second for the animation', default=None)
    parser.add_argument('--dpi', type=int, help='DPI for the output video', default=100)
    parser.add_argument('--duration', type=float, help='Duration of the animation in seconds (default: full animation)')
    
    args = parser.parse_args()
    args.bvh_file = "/home/bsd/cospeech/DiffGesture/data/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh"
    args.audio = "/home/bsd/cospeech/DiffGesture/data/beat_english_v0.2.1/1/1_wayne_0_1_1.wav"
    # Parse BVH file
    print(f"Parsing BVH file: {args.bvh_file}")
    bvh_data = parse_bvh(args.bvh_file)
    # get the file name from the bvh file
    args.output  = args.output.replace('.mp4', '_' + args.bvh_file.split('/')[-1].replace('.bvh', '.mp4'))
    # Create animation
    create_skeleton_animation(
        bvh_data,
        args.output,
        audio_path=args.audio,
        fps=args.fps,
        dpi=args.dpi,
        duration=args.duration
    )

if __name__ == "__main__":
    main()