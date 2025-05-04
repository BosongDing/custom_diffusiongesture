import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_npz_data(npz_file_path):
    """Load pre-computed joint data from NPZ file."""
    print(f"Loading data from {npz_file_path}")
    data = np.load(npz_file_path, allow_pickle=True)
    
    joint_positions = data['joint_positions']
    direction_vectors = data['direction_vectors']
    joint_names = data['joint_names'].tolist() if isinstance(data['joint_names'], np.ndarray) else data['joint_names']
    connections = data['connections'].tolist() if isinstance(data['connections'], np.ndarray) else data['connections']
    
    # Convert connection indices to integers if needed
    connections = [(int(parent_idx), int(child_idx), float(length)) for parent_idx, child_idx, length in connections]
    
    print(f"Loaded data with {joint_positions.shape[0]} frames, {len(joint_names)} joints")
    return joint_positions, direction_vectors, joint_names, connections

def is_hand_joint(joint_name):
    """Check if a joint is part of a hand."""
    hand_keywords = ['hand', 'thumb', 'index', 'middle', 'ring', 'pinky', 'finger']
    return any(keyword in joint_name.lower() for keyword in hand_keywords)

def is_spine_joint(joint_name):
    """Check if a joint is part of the spine."""
    return 'spine' in joint_name.lower() or 'neck' in joint_name.lower() or 'head' in joint_name.lower()

def is_shoulder_joint(joint_name):
    """Check if joint is a shoulder or clavicle."""
    return ('shoulder' in joint_name.lower() or 'clavicle' in joint_name.lower()) and \
           ('left' in joint_name.lower() or 'right' in joint_name.lower())

def visualize_motion_heatmap(joint_positions, joint_names, connections, output_path=None, 
                            downsample_factor=2, trail_alpha=0.05, plane='XY', fps=10):
    """
    Visualize all frames of motion data stacked together to show motion trajectory.
    
    Args:
        joint_positions: Array of shape (frames, joints, 3) containing joint positions
        joint_names: List of joint names for each position
        connections: List of tuples (parent_idx, child_idx, length)
        output_path: Path to save the output image
        downsample_factor: Factor to downsample frames (1 = use all frames)
        trail_alpha: Alpha value for motion trail
        plane: Viewing plane ('XY', 'XZ', or 'YZ')
        fps: Frames per second for showing skeleton connections
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set which coordinates to use based on the selected plane
    if plane == 'XY':
        x_idx, y_idx = 0, 1
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    elif plane == 'XZ':
        x_idx, y_idx = 0, 2
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    else:  # Default to YZ
        x_idx, y_idx = 1, 2
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
    
    # Fix the root position (first joint)
    num_frames = joint_positions.shape[0]
    root_fixed_positions = np.copy(joint_positions)
    
    # Get the root position from the first frame
    root_pos = root_fixed_positions[0, 0].copy()
    
    # Center all frames around this fixed root position
    for frame_idx in range(num_frames):
        offset = root_pos - root_fixed_positions[frame_idx, 0]
        for joint_idx in range(root_fixed_positions.shape[1]):
            root_fixed_positions[frame_idx, joint_idx] += offset
    
    # Downsample frames if needed
    positions = root_fixed_positions[::downsample_factor]
    
    # Calculate frame skip for the desired FPS
    # Assuming original data is at 30fps (or a similar rate)
    # We'll approximate this based on the number of frames and downsample factor
    original_fps = 15  # This is an assumption, adjust if known
    frame_skip = max(1, int((original_fps / downsample_factor) / fps))
    
    # Store hand joint positions for point cloud visualization
    right_hand_points_x = []
    right_hand_points_y = []
    right_hand_alphas = []
    right_hand_colors = []
    
    left_hand_points_x = []
    left_hand_points_y = []
    left_hand_alphas = []
    left_hand_colors = []
    
    # Draw the first frame skeleton with higher opacity for reference
    first_frame_positions = positions[0]
    
    # Draw connections between joints in the first frame (reference skeleton)
    for parent_idx, child_idx, _ in connections:
        if parent_idx >= first_frame_positions.shape[0] or child_idx >= first_frame_positions.shape[0]:
            print(f"Warning: Connection ({parent_idx}, {child_idx}) exceeds joint count {first_frame_positions.shape[0]}")
            continue
            
        parent_pos = first_frame_positions[parent_idx]
        child_pos = first_frame_positions[child_idx]
        
        # Determine color based on joint types
        color = 'black'  # Default color
        
        # Use joint names to determine type
        if parent_idx < len(joint_names) and child_idx < len(joint_names):
            parent_name = joint_names[parent_idx]
            child_name = joint_names[child_idx]
            
            # Color for spine and neck joints
            if is_spine_joint(child_name):
                color = 'green'
            # Color for shoulder connections
            elif is_shoulder_joint(child_name) or is_shoulder_joint(parent_name):
                color = 'purple'
            # Check if these are hand joints
            elif is_hand_joint(child_name) and is_hand_joint(parent_name):
                color = 'red' if 'right' in child_name.lower() else 'blue'
        
        # Draw the connection with moderate opacity for clarity (reference skeleton)
        ax.plot([parent_pos[x_idx], child_pos[x_idx]], 
                [parent_pos[y_idx], child_pos[y_idx]], 
                color=color, alpha=0.5, linewidth=1.5)
    
    # Draw regular intervals of skeleton based on fps
    for frame_idx in range(0, len(positions), frame_skip):
        frame_positions = positions[frame_idx]
        
        # Lower alpha for non-reference frames
        connection_alpha = 0.15
        
        # Process each connection
        for parent_idx, child_idx, _ in connections:
            if parent_idx >= frame_positions.shape[0] or child_idx >= frame_positions.shape[0]:
                continue
                
            parent_pos = frame_positions[parent_idx]
            child_pos = frame_positions[child_idx]
            
            # Determine color based on joint types
            color = 'black'  # Default color
            
            # Use joint names to determine type
            if parent_idx < len(joint_names) and child_idx < len(joint_names):
                parent_name = joint_names[parent_idx]
                child_name = joint_names[child_idx]
                
                # Color for spine and neck joints
                if is_spine_joint(child_name):
                    color = 'green'
                # Color for shoulder connections
                elif is_shoulder_joint(child_name) or is_shoulder_joint(parent_name):
                    color = 'purple'
                # Check if these are hand joints
                elif is_hand_joint(child_name) and is_hand_joint(parent_name):
                    color = 'red' if 'right' in child_name.lower() else 'blue'
            
            # Skip the first frame as it's already drawn with higher opacity
            if frame_idx != 0:
                ax.plot([parent_pos[x_idx], child_pos[x_idx]], 
                        [parent_pos[y_idx], child_pos[y_idx]], 
                        color=color, alpha=connection_alpha, linewidth=0.5)
    
    # Process each frame to gather hand joint positions (hand point cloud)
    for frame_idx, frame_positions in enumerate(positions):
        # Calculate alpha - more recent frames are more opaque
        alpha = trail_alpha
        
        # Process each joint in this frame
        for joint_idx, joint_pos in enumerate(frame_positions):
            if joint_idx >= len(joint_names):
                continue
                
            joint_name = joint_names[joint_idx]
            
            # If this is a hand joint, collect its position for the point cloud
            if is_hand_joint(joint_name):
                # Determine if it's right or left hand
                if 'right' in joint_name.lower():
                    right_hand_points_x.append(joint_pos[x_idx])
                    right_hand_points_y.append(joint_pos[y_idx])
                    right_hand_alphas.append(alpha)
                    right_hand_colors.append('red')
                elif 'left' in joint_name.lower():
                    left_hand_points_x.append(joint_pos[x_idx])
                    left_hand_points_y.append(joint_pos[y_idx])
                    left_hand_alphas.append(alpha)
                    left_hand_colors.append('blue')
    
    # Plot hand joints as point clouds
    # Right hand points
    if right_hand_points_x:
        ax.scatter(right_hand_points_x, right_hand_points_y, 
                  c=right_hand_colors, alpha=right_hand_alphas, s=20, marker='.')
    
    # Left hand points
    if left_hand_points_x:
        ax.scatter(left_hand_points_x, left_hand_points_y,
                  c=left_hand_colors, alpha=left_hand_alphas, s=20, marker='.')
    
    # Draw vertical center line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set title
    filename = os.path.basename(output_path) if output_path else "Motion Heatmap"
    ax.set_title(f'Motion Trajectory - {filename} ({fps} fps)')
    
    # Auto-adjust limits around the hand motion
    all_points_x = right_hand_points_x + left_hand_points_x
    all_points_y = right_hand_points_y + left_hand_points_y
    
    if all_points_x and all_points_y:
        min_x, max_x = min(all_points_x), max(all_points_x)
        min_y, max_y = min(all_points_y), max(all_points_y)
        
        # Add padding
        padding_x = (max_x - min_x) * 0.2
        padding_y = (max_y - min_y) * 0.2
        
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
    else:
        # Default limits if no hand points
        ax.set_xlim(-100, 100)
        ax.set_ylim(50, 190)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize motion data from NPZ files as heatmaps')
    parser.add_argument('--input', type=str, required=True, help='Path to the NPZ file or directory containing NPZ files')
    parser.add_argument('--plane', type=str, default='XY', choices=['XY', 'XZ', 'YZ'], help='Viewing plane')
    parser.add_argument('--downsample', type=int, default=1, help='Frame downsample factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha value for motion trails')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for skeleton connections')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input) and args.input.endswith('.npz'):
        # Process a single file
        joint_positions, direction_vectors, joint_names, connections = load_npz_data(args.input)
        output_path = args.input.replace('.npz', f'_heatmap_{args.plane}.png')
        visualize_motion_heatmap(joint_positions, joint_names, connections, 
                                output_path, args.downsample, args.alpha, args.plane, args.fps)
    elif os.path.isdir(args.input):
        # Process all NPZ files in the directory
        for filename in os.listdir(args.input):
            if filename.endswith('_direction_vectors.npz'):
                file_path = os.path.join(args.input, filename)
                try:
                    joint_positions, direction_vectors, joint_names, connections = load_npz_data(file_path)
                    output_path = file_path.replace('.npz', f'_heatmap_{args.plane}.png')
                    visualize_motion_heatmap(joint_positions, joint_names, connections, 
                                           output_path, args.downsample, args.alpha, args.plane, args.fps)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    else:
        print(f"Invalid input: {args.input}. Please provide a valid NPZ file or directory.")

if __name__ == "__main__":
    main()
