import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from tqdm import tqdm
def parse_bvh(file_path):
    """
    Parse BVH file and extract hierarchy and motion data with more robust handling
    
    Args:
        file_path (str): Path to the BVH file
        
    Returns:
        dict: Contains joint hierarchy, offsets, channels, and motion data
    """
    # Read the BVH file
    try:
        with open(file_path, 'r') as f:
            content = f.readlines()
    except Exception as e:
        raise ValueError(f"Error reading BVH file: {e}")
    
    # Split into hierarchy and motion sections
    hierarchy_lines = []
    motion_lines = []
    in_motion = False
    
    for line in content:
        if line.strip().upper() == "MOTION":
            in_motion = True
            continue
        if not in_motion:
            hierarchy_lines.append(line)
        else:
            motion_lines.append(line)
    
    if not hierarchy_lines:
        raise ValueError("Error: HIERARCHY section seems empty or missing.")
    if not motion_lines:
        raise ValueError("Error: MOTION section not found or empty.")
    
    # Parse hierarchy with the more robust approach from bvh3d.py
    joints, root_name, joint_list = parse_bvh_hierarchy(hierarchy_lines)
    
    # Parse motion data
    try:
        frames_line = motion_lines[0].strip()
        frame_time_line = motion_lines[1].strip()
        
        if not frames_line.upper().startswith("FRAMES:") or not frame_time_line.upper().startswith("FRAME TIME:"):
            raise ValueError("Motion header format incorrect (Expected 'Frames:' and 'Frame Time:').")
        
        # Convert to float first, then to int to handle frames values like "74437.0"
        num_frames = int(float(frames_line.split(':')[1].strip()))
        frame_time = float(frame_time_line.split(':')[1].strip())
        
        # Parse all motion data
        motion_data = []
        for i in range(2, len(motion_lines)):
            line = motion_lines[i].strip()
            if line:
                try:
                    motion_values = [float(x) for x in line.split()]
                    if motion_values:
                        motion_data.append(motion_values)
                except ValueError:
                    print(f"Warning: Could not parse motion data line: {line}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing MOTION section: {e}")
    
    # Convert joint structure from bvh3d format to BVH_convert format
    joint_data = {}
    parent_dict = {}
    
    for joint_name, joint_info in joints.items():
        # Skip end sites for main joint data (they will be included as children)
        if joint_info.get('is_end_site', False):
            continue
            
        parent_dict[joint_name] = joint_info['parent']
        
        joint_data[joint_name] = {
            'offset': joint_info['offset'],
            'channels': joint_info['channels'],
            'children': []
        }
        
        # Add children to joint_data
        for child in joint_info['children']:
            if isinstance(child, dict):
                child_name = child['name']
                if not child.get('is_end_site', False):  # Only add non-end site joints to children list
                    joint_data[joint_name]['children'].append(child_name)
    
    # Clean up joint_list to exclude end sites
    clean_joint_list = [j for j in joint_list if j in joint_data]
    
    bvh_data = {
        'joint_data': joint_data,
        'joint_order': clean_joint_list,
        'parent_dict': parent_dict,
        'root_joint': root_name,
        'frames': num_frames,
        'frame_time': frame_time,
        'motion_data': np.array(motion_data) if motion_data else np.array([])
    }
    
    return bvh_data

def parse_bvh_hierarchy(lines):
    """ Parses the HIERARCHY section of a BVH file. """
    joints = {}
    stack = []
    current_joint = None
    joint_list = [] # To maintain order and hierarchy for FK mapping and processing

    for line_num, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue

        keyword = parts[0].upper() # Use uppercase for consistency

        if keyword == "ROOT" or keyword == "JOINT":
            if len(parts) < 2:
                 print(f"Warning: Invalid ROOT/JOINT format at line {line_num+1}: {line.strip()}")
                 continue
            joint_name = parts[1]
            # Simple check for duplicate joint names which can break FK logic
            if joint_name in joints:
                print(f"Warning: Duplicate joint name '{joint_name}' found at line {line_num+1}. Behavior might be unpredictable.")

            joint_info = {
                'name': joint_name,
                'offset': None,
                'channels': [],
                'children': [],
                'parent': stack[-1]['name'] if stack else None
            }
            if stack: # If not root
                # Ensure parent in stack is valid dictionary before appending
                if isinstance(stack[-1], dict):
                    stack[-1]['children'].append(joint_info)
                else:
                    print(f"Error: Parent structure invalid when processing {joint_name}. Stack top: {stack[-1]}")
            joints[joint_name] = joint_info
            stack.append(joint_info)
            current_joint = joint_info
            joint_list.append(joint_name) # Add to ordered list crucial for data mapping

        elif keyword == "END": # End Site
            if not current_joint:
                 print(f"Warning: 'End Site' found outside of a JOINT block at line {line_num+1}")
                 continue
            # Create a unique name for the end site
            # Check if parent already has an End Site defined to avoid potential name clashes if multiple exist
            base_name = f"{current_joint['name']}_EndSite"
            end_site_name = base_name
            count = 1
            while end_site_name in joints:
                end_site_name = f"{base_name}{count}"
                count += 1

            joint_info = {
                'name': end_site_name,
                'offset': None,
                'channels': [], # No channels for End Site
                'children': [],
                'parent': stack[-1]['name'] if stack else None,
                'is_end_site': True # Flag for easier identification
            }
            if isinstance(stack[-1], dict): # Ensure parent is valid
                 stack[-1]['children'].append(joint_info)
            joints[end_site_name] = joint_info
            stack.append(joint_info) # Push End Site onto stack
            current_joint = joint_info
            # Do not add End Site to the primary joint_list used for motion data mapping

        elif keyword == "OFFSET":
             if not current_joint or 'offset' not in current_joint:
                 print(f"Warning: OFFSET found outside of a JOINT/End Site block or invalid context at line {line_num+1}")
                 continue
             try:
                 current_joint['offset'] = np.array([float(p) for p in parts[1:]])
             except (ValueError, IndexError):
                 print(f"Warning: Invalid OFFSET format at line {line_num+1}: {line.strip()}")
                 current_joint['offset'] = np.zeros(3) # Default to zero offset on error

        elif keyword == "CHANNELS":
             if not current_joint or 'channels' not in current_joint:
                 print(f"Warning: CHANNELS found outside of a JOINT block at line {line_num+1}")
                 continue
             try:
                 num_channels = int(parts[1])
                 current_joint['channels'] = parts[2:]
                 if len(current_joint['channels']) != num_channels:
                      print(f"Warning: Mismatch in channel count for {current_joint['name']} at line {line_num+1}. Expected {num_channels}, got {len(current_joint['channels'])}.")
                      # Attempt to truncate or pad? Better to warn and proceed cautiously.
                      current_joint['channels'] = current_joint['channels'][:num_channels] # Truncate if too many listed
             except (ValueError, IndexError):
                 print(f"Warning: Invalid CHANNELS format at line {line_num+1}: {line.strip()}")
                 current_joint['channels'] = []

        elif keyword == "}":
            if stack:
                 stack.pop() # Pop current joint/end site
                 if stack:
                     # Check if the new stack top is a dictionary (valid joint)
                     if isinstance(stack[-1], dict):
                          current_joint = stack[-1] # Go back to parent
                     else:
                          print("Error: Invalid stack state after popping '}'.")
                          current_joint = None
                 else:
                     current_joint = None # Reached root's closing brace
            else:
                print(f"Warning: Extra closing brace '}}' found at line {line_num+1}")

    if not joint_list:
        print("Error: No ROOT joint found in hierarchy.")
        return {}, None, []

    root_name = joint_list[0]
    return joints, root_name, joint_list

def validate_bvh_structure(bvh_data):
    """
    Validate the BVH hierarchy structure for consistency
    
    Args:
        bvh_data (dict): The parsed BVH data
        
    Returns:
        bool: True if valid, False otherwise
    """
    joint_data = bvh_data['joint_data']
    parent_dict = bvh_data['parent_dict']
    root_joint = bvh_data['root_joint']
    
    # Check for orphaned joints (joints with no parent except the root)
    orphaned_joints = [j for j, p in parent_dict.items() if p is None and j != root_joint]
    if orphaned_joints:
        print(f"Warning: Found orphaned joints: {orphaned_joints}")
    
    # Check for consistency in parent-child relationships
    for joint_name, joint_info in joint_data.items():
        children = joint_info['children']
        for child in children:
            if child not in joint_data:
                print(f"Error: Child joint {child} not found in joint_data")
                return False
            if parent_dict[child] != joint_name:
                print(f"Error: Inconsistent parent-child relationship for {child}")
                return False
    
    # Create a visual representation of the hierarchy
    def print_hierarchy(joint, level=0):
        indent = "  " * level
        print(f"{indent}- {joint}")
        for child in joint_data[joint]['children']:
            print_hierarchy(child, level + 1)
    
    print("BVH Hierarchy:")
    print_hierarchy(root_joint)
    
    return True


def visualize_hierarchy(bvh_data):
    """
    Generate a text-based visualization of the BVH hierarchy
    
    Args:
        bvh_data (dict): The parsed BVH data
        
    Returns:
        str: A string representation of the hierarchy
    """
    joint_data = bvh_data['joint_data']
    root_joint = bvh_data['root_joint']
    
    lines = []
    
    def add_joint(joint, level=0):
        indent = "  " * level
        lines.append(f"{indent}- {joint}")
        for child in joint_data[joint]['children']:
            add_joint(child, level + 1)
    
    lines.append("BVH Hierarchy:")
    add_joint(root_joint)
    
    return "\n".join(lines)
def calculate_joint_positions(bvh_data, exclude_joints=None):
    """
    Calculate joint positions for each frame using forward kinematics
    
    Args:
        bvh_data (dict): BVH data from parse_bvh function
        exclude_joints (list): List of joints to exclude
        
    Returns:
        numpy.ndarray: Joint positions for each frame, shape (frames, joints, 3)
    """
    if exclude_joints is None:
        exclude_joints = []
    
    # Filter joints to include (remove excluded joints)
    include_joints = [j for j in bvh_data['joint_order'] if j not in exclude_joints]
    joint_data = bvh_data['joint_data']
    parent_dict = bvh_data['parent_dict']
    motion_data = bvh_data['motion_data']
    
    num_frames = len(motion_data)
    num_joints = len(include_joints)
    
    # Initialize joint positions array
    positions = np.zeros((num_frames, num_joints, 3))
    
    # Set up channel indices for each joint
    channel_indices = {}
    index = 0
    
    for joint_name in bvh_data['joint_order']:
        if joint_name not in joint_data:
            continue
        num_channels = len(joint_data[joint_name]['channels'])
        channel_indices[joint_name] = (index, index + num_channels)
        index += num_channels
    
    # Create a mapping from excluded parents to their nearest included ancestors
    parent_mapping = {}
    for joint_name in include_joints:
        current = parent_dict[joint_name]
        # If the parent is excluded, find the first non-excluded ancestor
        while current is not None and current in exclude_joints:
            current = parent_dict[current]
        parent_mapping[joint_name] = current
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame_data = motion_data[frame_idx]
        
        # Calculate global positions for all included joints
        joint_positions = {}
        orientations = {}  # Store world orientation matrices
        
        # Helper function to calculate global position and rotation for a joint
        def calculate_global_transform(joint_name):
            if joint_name in joint_positions:
                return joint_positions[joint_name], orientations[joint_name]
            
            if joint_name not in joint_data:
                # This shouldn't happen with proper filtering, but handle it just in case
                return np.zeros(3), np.identity(3)
            
            # Get joint properties
            parent_name = parent_mapping.get(joint_name, None)
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
            if parent_name is None:  # Root joint or joint with all parents excluded
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
        
        # Calculate position for each included joint
        for i, joint_name in enumerate(include_joints):
            global_pos, _ = calculate_global_transform(joint_name)
            positions[frame_idx, i] = global_pos
    
    return positions, include_joints

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


    
def convert_bvh_to_direction_vectors(bvh_file, output_file=None, exclude_lower_body=True, start_frame=0, end_frame=None):
    """
    Convert BVH file to direction vectors format
    
    Args:
        bvh_file (str): Path to the BVH file
        output_file (str): Path to save the converted data (optional)
        exclude_lower_body (bool): Whether to exclude lower body joints
        start_frame (int): First frame to process
        end_frame (int): Last frame to process (None means all frames)
        
    Returns:
        tuple: (joint_positions, direction_vectors, joint_names, connections)
    """
    # Parse BVH file
    print(f"Parsing BVH file: {bvh_file}")
    bvh_data = parse_bvh(bvh_file)
    # print(bvh_data['joint_order'])

    # print(validate_bvh_structure(bvh_data))
    # Define joints to exclude
    exclude_joints = []
    
    if exclude_lower_body:
        # Exclude lower body joints and lower spine
        exclude_joints = [
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightForeFoot', 
            'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase','LeftForeFoot',"LeftToeBaseEnd","RightToeBaseEnd",'LeftToeBase',"HeadEnd"
            'Hips', 'Spine',"Spine1","pCube4"  # Exclude all spines below Spine2
        ]

    
    print(f"Excluding {len(exclude_joints)} joints: {exclude_joints}")
    
    # Calculate joint positions
    print("Calculating joint positions...")
    joint_positions, included_joints = calculate_joint_positions(bvh_data, exclude_joints)
    
    # Apply frame range
    if end_frame is None:
        end_frame = joint_positions.shape[0]
    
    joint_positions = joint_positions[start_frame:end_frame]
    
    # Define joint connections based on parent-child relationships
    connections = []
    
    for i, joint_name in enumerate(included_joints):
        parent_name = bvh_data['parent_dict'][joint_name]
        if parent_name in included_joints:
            parent_idx = included_joints.index(parent_name)
            child_idx = i
            
            # Get joint offset as distance
            offset = np.array(bvh_data['joint_data'][joint_name]['offset'])
            length = np.linalg.norm(offset)
            
            connections.append((parent_idx, child_idx, length))
    
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
            joint_names=included_joints,
            connections=connections
        )
    
    return joint_positions, dir_vectors, included_joints, connections,bvh_data

def visualize_skeleton(joint_positions, joint_connections, frame_idx=0):
    """
    Visualize skeleton for a given frame
    
    Args:
        joint_positions (numpy.ndarray): Joint positions, shape (frames, joints, 3)
        joint_connections (list): List of tuples (parent_idx, child_idx, length)
        frame_idx (int): Frame index to visualize
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get joint positions for the frame
    positions = joint_positions[frame_idx]
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50)
    
    # Plot connections
    for parent_idx, child_idx, _ in joint_connections:
        parent_pos = positions[parent_idx]
        child_pos = positions[child_idx]
        # Use the original coordinate system - maintain consistency with visualize_original_bvh
        ax.plot([parent_pos[0], child_pos[0]], 
                [parent_pos[1], child_pos[1]], 
                [parent_pos[2], child_pos[2]], 'r-', linewidth=2)
    
    # Set axis properties
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
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    plt.title(f'Skeleton Visualization (Frame {frame_idx})')
    plt.show()

def visualize_direction_vectors(joint_positions, dir_vectors, connections, frame_idx=0):
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
    
    ax1.set_title('Joint Positions')
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Plot direction vectors
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot each direction vector - maintaining correct coordinate system
    for i, (parent_idx, child_idx, length) in enumerate(connections):
        # Get parent position and direction vector
        parent_pos = positions[parent_idx]
        direction = dir_vectors[frame_idx, i] * length  # Scale by original length for visualization
        
        # Calculate end position
        end_pos = parent_pos + direction
        
        # Plot vector using original coordinate system
        ax2.quiver(parent_pos[0], parent_pos[1], parent_pos[2],
                  direction[0], direction[1], direction[2],
                  color='green', alpha=0.8, linewidth=2)
    
    # Plot joints
    ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=30)
    
    ax2.set_title('Direction Vectors')
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
    plt.show()

def visualize_original_bvh(bvh_data, frame_idx=0):
    """
    Visualize the original BVH skeleton for a given frame before any conversion
    
    Args:
        bvh_data (dict): BVH data from parse_bvh function
        frame_idx (int): Frame index to visualize
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R
    
    # Extract required data
    joint_data = bvh_data['joint_data']
    parent_dict = bvh_data['parent_dict']
    joint_order = bvh_data['joint_order']
    motion_data = bvh_data['motion_data']
    
    # Ensure frame index is valid
    if frame_idx >= len(motion_data):
        print(f"Warning: Frame index {frame_idx} out of range. Using frame 0.")
        frame_idx = 0
    
    # Set up channel indices for each joint
    channel_indices = {}
    index = 0
    for joint_name in joint_order:
        if joint_name not in joint_data:
            continue
        num_channels = len(joint_data[joint_name]['channels'])
        channel_indices[joint_name] = (index, index + num_channels)
        index += num_channels
    
    # Calculate global positions for all joints
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
    
    positions = np.array(positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50)
    
    # Add joint labels if needed (can be commented out if too cluttered)
    for i, label in enumerate(labels):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, size=8, color='black')
    
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
                
                # Use the original coordinate system
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
    
    plt.title(f'Original BVH Skeleton (Frame {frame_idx})')
    plt.tight_layout()
    plt.show()
    
    return positions, labels
# Example usage:
# bvh_data = parse_bvh('your_file.bvh')
# visualize_original_bvh(bvh_data, frame_idx=0)    
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Trinity/Beat/all")
    args = parser.parse_args()
    if args.dataset == "Trinity" or args.dataset == "all":
        for bvh_file_path in tqdm(os.listdir("./data/trinity/allRec")):
            #check if the bvh file is already converted
            if not bvh_file_path.endswith('.bvh'):
                continue
            if os.path.exists(os.path.join("./data/trinity/allRec", bvh_file_path.replace('.bvh', '_direction_vectors.npz'))):
                continue
            bvh_file_path = os.path.join("./data/trinity/allRec", bvh_file_path)
            output_file_path = bvh_file_path.replace('.bvh', '_direction_vectors.npz')
            joint_positions, dir_vectors, joint_names, connections, bvh_data = convert_bvh_to_direction_vectors(
                bvh_file_path, 
                output_file_path,
                exclude_lower_body=True,
                start_frame=0,
                end_frame=None  # Process all frames
            )
        for bvh_file_path in tqdm(os.listdir("./data/trinity/allTestMotion")):
            #check if the bvh file is already converted
            if not bvh_file_path.endswith('.bvh'):
                continue
            if os.path.exists(os.path.join("./data/trinity/allTestMotion", bvh_file_path.replace('.bvh', '_direction_vectors.npz'))):
                continue
            bvh_file_path = os.path.join("./data/trinity/allTestMotion", bvh_file_path)
            output_file_path = bvh_file_path.replace('.bvh', '_direction_vectors.npz')
            joint_positions, dir_vectors, joint_names, connections, bvh_data = convert_bvh_to_direction_vectors(
                bvh_file_path, 
                output_file_path,
                exclude_lower_body=True,
                start_frame=0,
                end_frame=None  # Process all frames
            )
    if args.dataset == "Beat" or args.dataset == "all":
        for folder in tqdm(os.listdir("./data/beat_english_v0.2.1")):
            for bvh_file_path in os.listdir(os.path.join("./data/beat_english_v0.2.1", folder)):
                #check if the bvh file is already converted
                if not bvh_file_path.endswith('.bvh'):
                    continue
                if os.path.exists(os.path.join("./data/beat_english_v0.2.1", folder, bvh_file_path.replace('.bvh', '_direction_vectors.npz'))):
                    continue
                
                
                bvh_file_path = os.path.join("./data/beat_english_v0.2.1", folder, bvh_file_path)
                output_file_path = bvh_file_path.replace('.bvh', '_direction_vectors.npz')
                joint_positions, dir_vectors, joint_names, connections, bvh_data = convert_bvh_to_direction_vectors(
                    bvh_file_path, 
                    output_file_path,
                    exclude_lower_body=True,
                    start_frame=0,
                    end_frame=None  # Process all frames
                )