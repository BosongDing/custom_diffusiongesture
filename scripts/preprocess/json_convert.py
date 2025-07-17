import json
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_json_motion(json_file_path):
    """
    Parse JSON motion file and extract joint data
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        dict: Contains joint data
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")
    
    if 'frames' not in data:
        raise ValueError("Error: 'frames' not found in JSON data.")
    
    return data

def filter_joints(json_data, exclude_joints):
    """
    Filter out excluded joints from JSON data
    
    Args:
        json_data (dict): Parsed JSON data
        exclude_joints (list): List of joints to exclude
        
    Returns:
        dict: Filtered JSON data with excluded joints removed
    """
    filtered_data = {"frames": []}
    
    for frame in json_data["frames"]:
        filtered_frame = {
            "time": frame["time"],
            "joints": []
        }
        
        # Filter joints in this frame
        for joint in frame["joints"]:
            if joint["name"] not in exclude_joints:
                filtered_frame["joints"].append(joint)
        
        filtered_data["frames"].append(filtered_frame)
    
    return filtered_data

def build_joint_hierarchy(json_data):
    """
    Build joint hierarchy from filtered JSON data
    
    Args:
        json_data (dict): Filtered JSON data
        
    Returns:
        tuple: (joint_data, parent_dict, joint_order, root_joint)
    """
    # Extract the first frame to build hierarchy
    if not json_data['frames']:
        raise ValueError("Error: No frames found in JSON data.")
    
    first_frame = json_data['frames'][0]
    joints = first_frame['joints']
    
    # Create a set of all available joint names
    available_joints = {joint['name'] for joint in joints}
    
    # Build joint data structure
    joint_data = {}
    parent_dict = {}
    joint_order = []
    roots = []
    
    for joint in joints:
        joint_name = joint['name']
        parent_name = joint['parentName']
        
        # Check if parent exists in our filtered joint set
        if parent_name == 'none' or parent_name not in available_joints:
            parent_name = None
            roots.append(joint_name)
        
        joint_data[joint_name] = {
            'offset': [0, 0, 0],  # Will be calculated from positions
            'channels': [],  # Not used in JSON format
            'children': []
        }
        
        parent_dict[joint_name] = parent_name
        joint_order.append(joint_name)
    
    # Check if we have at least one root
    if not roots:
        raise ValueError("Error: No root joint found after filtering.")
    
    # If multiple roots, use the first one (can be modified based on requirements)
    root_joint = roots[0]
    
    # Add children to joint_data
    for joint_name, parent_name in parent_dict.items():
        if parent_name is not None and parent_name in joint_data:
            joint_data[parent_name]['children'].append(joint_name)
    
    # Calculate offsets based on positions in the first frame
    for joint in joints:
        joint_name = joint['name']
        parent_name = parent_dict[joint_name]
        
        if parent_name is not None:
            # Find parent and child positions
            parent_pos = None
            for j in joints:
                if j['name'] == parent_name:
                    parent_pos = np.array([
                        j['position']['x'],
                        j['position']['y'],
                        j['position']['z']
                    ])
                    break
            
            if parent_pos is not None:
                child_pos = np.array([
                    joint['position']['x'],
                    joint['position']['y'],
                    joint['position']['z']
                ])
                
                # Calculate offset
                offset = child_pos - parent_pos
                joint_data[joint_name]['offset'] = offset
    
    return joint_data, parent_dict, joint_order, root_joint

def extract_joint_positions(json_data, joint_order):
    """
    Extract joint positions from all frames
    
    Args:
        json_data (dict): Filtered JSON data
        joint_order (list): Order of joints
        
    Returns:
        numpy.ndarray: Joint positions for each frame, shape (frames, joints, 3)
    """
    frames = json_data['frames']
    num_frames = len(frames)
    num_joints = len(joint_order)
    
    # Initialize joint positions array
    positions = np.zeros((num_frames, num_joints, 3))
    
    # Extract positions for each frame
    for frame_idx, frame in enumerate(frames):
        # Create a dictionary for fast lookup
        joint_dict = {joint['name']: joint for joint in frame['joints']}
        
        for joint_idx, joint_name in enumerate(joint_order):
            if joint_name in joint_dict:
                joint = joint_dict[joint_name]
                position = np.array([
                    joint['position']['x'],
                    joint['position']['y'],
                    joint['position']['z']
                ])
                positions[frame_idx, joint_idx] = position
    
    return positions

def build_connections(joint_order, parent_dict, joint_positions, frame_idx=0):
    """
    Build connections between joints
    
    Args:
        joint_order (list): Order of joints
        parent_dict (dict): Parent-child relationships
        joint_positions (numpy.ndarray): Joint positions
        frame_idx (int): Frame to use for distance calculation
        
    Returns:
        list: List of tuples (parent_idx, child_idx, length)
    """
    connections = []
    
    for i, joint_name in enumerate(joint_order):
        parent_name = parent_dict[joint_name]
        if parent_name is not None and parent_name in joint_order:
            parent_idx = joint_order.index(parent_name)
            child_idx = i
            
            # Calculate length using joint positions
            parent_pos = joint_positions[frame_idx, parent_idx]
            child_pos = joint_positions[frame_idx, child_idx]
            length = np.linalg.norm(child_pos - parent_pos)
            
            connections.append((parent_idx, child_idx, length))
    
    return connections

def calculate_direction_vectors(joint_positions, joint_connections):
    """
    Calculate direction vectors between connected joints
    
    Args:
        joint_positions (numpy.ndarray): Joint positions, shape (frames, joints, 3)
        joint_connections (list): List of tuples (parent_idx, child_idx, length)
        
    Returns:
        numpy.ndarray: Direction vectors, shape (frames, connections, 3)
    """
    num_frames, num_joints, _ = joint_positions.shape
    num_connections = len(joint_connections)
    
    # Initialize direction vectors array
    dir_vectors = np.zeros((num_frames, num_connections, 3))
    
    # Calculate direction vectors
    for frame_idx in range(num_frames):
        for conn_idx, (parent_idx, child_idx, _) in enumerate(joint_connections):
            # Get joint positions
            parent_pos = joint_positions[frame_idx, parent_idx]
            child_pos = joint_positions[frame_idx, child_idx]
            
            # Calculate direction vector
            direction = child_pos - parent_pos
            
            # Normalize to unit length
            norm = np.linalg.norm(direction)
            if norm > 1e-10:  # Avoid division by zero
                direction = direction / norm
            
            # Store direction vector
            dir_vectors[frame_idx, conn_idx] = direction
    
    return dir_vectors

def visualize_hierarchy(joint_data, root_joint):
    """
    Generate a text-based visualization of the hierarchy
    
    Args:
        joint_data (dict): Joint data
        root_joint (str): Name of the root joint
        
    Returns:
        str: A string representation of the hierarchy
    """
    lines = []
    
    def add_joint(joint, level=0):
        indent = "  " * level
        lines.append(f"{indent}- {joint}")
        for child in joint_data[joint]['children']:
            add_joint(child, level + 1)
    
    lines.append("Joint Hierarchy:")
    add_joint(root_joint)
    
    return "\n".join(lines)

def check_bone_consistency(joint_positions, connections, joint_order, threshold=0.01):
    """
    Check if bone lengths are consistent across frames
    
    Args:
        joint_positions (numpy.ndarray): Joint positions, shape (frames, joints, 3)
        connections (list): List of tuples (parent_idx, child_idx, length)
        joint_order (list): Order of joints (names)
        threshold (float): Maximum allowable relative deviation from mean bone length
        
    Returns:
        tuple: (is_consistent, stats_report)
            - is_consistent (bool): True if all bones are consistent within threshold
            - stats_report (dict): Statistics on bone length variations
    """
    skip_frames = 0
    num_frames = joint_positions.shape[0]-skip_frames
    num_connections = len(connections)
    
    # Initialize arrays to store bone lengths for each connection across frames
    bone_lengths = np.zeros((num_frames, num_connections))

    # Calculate bone length for each connection in each frame
    i =0
    for frame_idx in range(skip_frames,num_frames+skip_frames):
        for conn_idx, (parent_idx, child_idx, _) in enumerate(connections):
            # Get joint positions
            parent_pos = joint_positions[frame_idx, parent_idx]
            child_pos = joint_positions[frame_idx, child_idx]
            
            # Calculate bone length
            length = np.linalg.norm(child_pos - parent_pos)
            bone_lengths[i, conn_idx] = length
        i+=1

    # Calculate statistics for each bone
    stats = {
        'mean': np.mean(bone_lengths, axis=0),
        'std': np.std(bone_lengths, axis=0),
        'min': np.min(bone_lengths, axis=0),
        'max': np.max(bone_lengths, axis=0),
        'relative_std': np.zeros(num_connections)
    }
    
    # Calculate relative standard deviation (coefficient of variation)
    for i in range(num_connections):
        if stats['mean'][i] > 1e-10:  # Avoid division by zero
            stats['relative_std'][i] = stats['std'][i] / stats['mean'][i]
    
    # Check if any bone length deviates too much
    is_consistent = False
    inconsistent_bones = []
    
    for i, (parent_idx, child_idx, _) in enumerate(connections):
        parent_name = joint_order[parent_idx]
        child_name = joint_order[child_idx]
        
        if stats['relative_std'][i] > threshold:
            is_consistent = False
            inconsistent_bones.append({
                'connection': f"{parent_name} -> {child_name}",
                'mean_length': stats['mean'][i],
                'std_dev': stats['std'][i],
                'relative_std': stats['relative_std'][i],
                'min': stats['min'][i],
                'max': stats['max'][i]
            })
    
    # Create a report
    report = {
        'is_consistent': is_consistent,
        'threshold': threshold,
        'global_stats': {
            'mean_relative_std': np.mean(stats['relative_std']),
            'max_relative_std': np.max(stats['relative_std']),
            'min_relative_std': np.min(stats['relative_std'])
        },
        'inconsistent_bones': inconsistent_bones
    }
    
    return is_consistent, report

def convert_json_to_direction_vectors(json_file, output_file=None, exclude_lower_body=True, check_consistency=False):
    """
    Convert JSON file to direction vectors format
    
    Args:
        json_file (str): Path to the JSON file
        output_file (str): Path to save the converted data (optional)
        exclude_lower_body (bool): Whether to exclude lower body joints
        check_consistency (bool): Whether to check bone length consistency
        
    Returns:
        tuple: (joint_positions, direction_vectors, joint_names, connections, consistency_report)
    """
    # Parse JSON file
    print(f"Parsing JSON file: {json_file}")
    json_data = parse_json_motion(json_file)
    
    # Define joints to exclude
    exclude_joints = []
    
    if exclude_lower_body:
        # Specify the lower body joints to exclude
        exclude_joints = [
            'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
            'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
            'Hips',  # Including Hips as you mentioned you might delete the root
            'Spine', 'LeftToes', 'RightToes'
            # Add any other joints you want to exclude
        ]
    
    print(f"Excluding {len(exclude_joints)} joints: {exclude_joints}")
    
    # Filter out excluded joints FIRST
    filtered_json_data = filter_joints(json_data, exclude_joints)
    
    # Build joint hierarchy from filtered data
    joint_data, parent_dict, joint_order, root_joint = build_joint_hierarchy(filtered_json_data)
    
    # Print hierarchy for debugging
    hierarchy_text = visualize_hierarchy(joint_data, root_joint)
    print(hierarchy_text)
    
    # Extract joint positions
    joint_positions = extract_joint_positions(filtered_json_data, joint_order)
    
    # Build connections
    connections = build_connections(joint_order, parent_dict, joint_positions)
    
    # Check bone consistency if requested
    consistency_report = None
    if check_consistency:
        print("Checking bone length consistency...")
        is_consistent, consistency_report = check_bone_consistency(joint_positions, connections, joint_order)
        if not is_consistent:
            print("WARNING: Bone lengths are not consistent across frames!")
            print(f"Global mean relative std: {consistency_report['global_stats']['mean_relative_std']:.6f}")
            print(f"Max relative std: {consistency_report['global_stats']['max_relative_std']:.6f}")
            print(f"Number of inconsistent bones: {len(consistency_report['inconsistent_bones'])}")
            for bone in consistency_report['inconsistent_bones']:
                print(f"  {bone['connection']}: rel_std={bone['relative_std']:.6f}, "
                      f"mean={bone['mean_length']:.4f}, min={bone['min']:.4f}, max={bone['max']:.4f}")
    
    # Calculate direction vectors
    print("Calculating direction vectors...")
    dir_vectors = calculate_direction_vectors(joint_positions, connections)
    
    # Save to file if specified
    if output_file:
        print(f"Saving to {output_file}...")
        np.savez(
            output_file,
            joint_positions=joint_positions,
            direction_vectors=dir_vectors,
            joint_names=joint_order,
            connections=connections
        )
    
    return joint_positions, dir_vectors, joint_order, connections

def main():
    parser = argparse.ArgumentParser(description='Convert JSON motion data to direction vectors')
    parser.add_argument('--input', type=str,  help='Input JSON file or directory')
    parser.add_argument('--output', type=str, help='Output file or directory (default: same as input with _direction_vectors.npz)')
    parser.add_argument('--exclude_lower_body', default=True, help='Exclude lower body joints')
    parser.add_argument('--check_consistency', default=True, action='store_true', help='Check bone length consistency across frames')
    
    args = parser.parse_args()
    # args.input = "/home/bsd/cospeech/DiffGesture/data/quest/voice_20250427_143848_motion_0.json"
    
    if os.path.isdir(args.input):
        # Process directory
        if args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
            
        for filename in tqdm(os.listdir(args.input)):
            if filename.endswith('.json'):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output if args.output else args.input, 
                                          filename.replace('.json', '_direction_vectors.npz'))
                
                # Skip if already converted
                if os.path.exists(output_path):
                    continue
                
                try:
                    convert_json_to_direction_vectors(
                        input_path,
                        output_path,
                        exclude_lower_body=args.exclude_lower_body,
                        check_consistency=args.check_consistency
                    )
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    else:
        # Process single file
        if not args.output:
            args.output = args.input.replace('.json', '_direction_vectors.npz')
            
        convert_json_to_direction_vectors(
            args.input,
            args.output,
            exclude_lower_body=args.exclude_lower_body,
            check_consistency=args.check_consistency
        )

if __name__ == "__main__":
    main()
