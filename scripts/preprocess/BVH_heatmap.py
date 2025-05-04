import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import re
import math
import os

def parse_bvh(file_path):
    """Parse a BVH file and extract hierarchy and motion data."""
    print(f"Starting to parse {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        hierarchy_section, motion_section = content.split('MOTION')
        
        # Parse joint hierarchy
        joints = {}
        joint_stack = []
        current_joint = None
        line_number = 0
        
        for line in hierarchy_section.split('\n'):
            line_number += 1
            line = line.strip()
            
            if 'ROOT' in line:
                print(f"  Line {line_number}: Parsing ROOT {line}")
                joint_name = line.split('ROOT')[1].strip()
                current_joint = {'name': joint_name, 'offset': None, 'channels': [], 'children': [], 'parent': None}
                joints[joint_name] = current_joint
                joint_stack.append(current_joint)
                
            elif 'JOINT' in line:
                print(f"  Line {line_number}: Parsing JOINT {line}")
                joint_name = line.split('JOINT')[1].strip()
                if len(joint_stack) == 0:
                    print(f"ERROR in {file_path} at line {line_number}: Joint stack is empty when processing joint '{joint_name}'")
                    raise ValueError(f"Invalid BVH structure: Joint stack is empty when processing joint '{joint_name}'")
                parent_joint = joint_stack[-1]
                current_joint = {'name': joint_name, 'offset': None, 'channels': [], 'children': [], 'parent': parent_joint['name']}
                parent_joint['children'].append(joint_name)
                joints[joint_name] = current_joint
                joint_stack.append(current_joint)
                
            elif 'End site' in line or 'End Site' in line:
                print(f"  Line {line_number}: Parsing End Site")
                if len(joint_stack) == 0:
                    print(f"ERROR in {file_path} at line {line_number}: Joint stack is empty when processing End Site")
                    raise ValueError(f"Invalid BVH structure: Joint stack is empty when processing End Site")
                joint_name = joint_stack[-1]['name'] + '_end'
                parent_joint = joint_stack[-1]
                current_joint = {'name': joint_name, 'offset': None, 'channels': [], 'children': [], 'parent': parent_joint['name']}
                parent_joint['children'].append(joint_name)
                joints[joint_name] = current_joint
                joint_stack.append(current_joint)
                
            elif 'OFFSET' in line:
                print(f"  Line {line_number}: Parsing OFFSET {line}")
                offset = [float(x) for x in line.split('OFFSET')[1].strip().split()]
                current_joint['offset'] = offset
                
            elif 'CHANNELS' in line:
                print(f"  Line {line_number}: Parsing CHANNELS {line}")
                channel_info = line.split('CHANNELS')[1].strip().split()
                num_channels = int(channel_info[0])
                channels = channel_info[1:1+num_channels]
                current_joint['channels'] = channels
                
            elif '}' in line:
                print(f"  Line {line_number}: Closing bracket, popping from stack (depth: {len(joint_stack)})")
                if joint_stack:
                    joint_stack.pop()
                    print(f"    Stack depth after pop: {len(joint_stack)}")
        
        # Parse motion data
        motion_lines = motion_section.strip().split('\n')
        frames_match = re.search(r'Frames:\s+(\d+)', motion_lines[0])
        frame_time_match = re.search(r'Frame Time:\s+([\d.]+)', motion_lines[1])
        
        if not frames_match or not frame_time_match:
            raise ValueError("Couldn't parse number of frames or frame time")
        
        num_frames = int(frames_match.group(1))
        frame_time = float(frame_time_match.group(1))
        
        # Get motion data
        motion_data = []
        for i in range(2, len(motion_lines)):
            if motion_lines[i].strip():  # Skip empty lines
                frame_values = [float(x) for x in motion_lines[i].strip().split()]
                motion_data.append(frame_values)
        
        print(f"Successfully parsed {file_path}: {len(joints)} joints, {len(motion_data)} frames")
        return joints, motion_data, num_frames, frame_time
    except Exception as e:
        print(f"ERROR parsing {file_path}: {str(e)}")
        raise

def create_skeleton_from_frame(joints, motion_data, frame_idx):
    """Create skeleton coordinate data from a specific frame of motion data."""
    skeleton = {}
    channel_index = 0  # Start at the beginning of the motion data for this frame
    
    # First pass: collect positions and rotations
    joint_transforms = {}
    
    # Second pass: calculate global positions using parent rotations
    def process_joint(joint_name, parent_global_position=None, parent_global_rotation_matrix=np.identity(3)):
        nonlocal channel_index
        joint = joints[joint_name]
        local_offset = np.array(joint['offset'])
        
        # For the root joint, get position directly from motion data
        if parent_global_position is None:  # Root joint
            position = np.zeros(3)
            rotation = np.zeros(3)
            
            # Extract values from motion data according to the channels
            for channel in joint['channels']:
                if 'position' in channel.lower():
                    axis = channel[0].lower()
                    axis_idx = 'xyz'.index(axis)
                    position[axis_idx] = motion_data[frame_idx][channel_index]
                elif 'rotation' in channel.lower():
                    axis = channel[0].lower()
                    axis_idx = 'xyz'.index(axis)
                    rotation[axis_idx] = motion_data[frame_idx][channel_index]
                channel_index += 1
            
            # Apply rotations to create rotation matrix (ZYX order is common in BVH)
            rot_matrix = create_rotation_matrix(rotation)
            
            # Save the global position and rotation
            global_position = position
            global_rotation_matrix = rot_matrix
            
        else:  # Non-root joint
            rotation = np.zeros(3)
            
            # Extract rotation values from motion data
            for channel in joint['channels']:
                if 'rotation' in channel.lower():
                    axis = channel[0].lower()
                    axis_idx = 'xyz'.index(axis)
                    rotation[axis_idx] = motion_data[frame_idx][channel_index]
                channel_index += 1
            
            # Create local rotation matrix
            local_rot_matrix = create_rotation_matrix(rotation)
            
            # Calculate global position and rotation
            global_rotation_matrix = np.dot(parent_global_rotation_matrix, local_rot_matrix)
            
            # Apply parent's rotation to the local offset
            rotated_offset = np.dot(parent_global_rotation_matrix, local_offset)
            
            # Calculate global position by adding the rotated offset to parent's position
            global_position = parent_global_position + rotated_offset
        
        # Store joint data
        skeleton[joint_name] = {
            'position': global_position.tolist(),
            'rotation': rotation.tolist() if 'rotation' in locals() else [0, 0, 0]
        }
        
        # Process children
        for child_name in joint['children']:
            process_joint(child_name, global_position, global_rotation_matrix)
    
    # Start with the root joint
    root_name = next(name for name, data in joints.items() if data['parent'] is None)
    process_joint(root_name)
    
    return skeleton

def create_rotation_matrix(rotation_angles):
    """Create a rotation matrix from Euler angles in degrees (ZYX order)."""
    # Convert to radians
    rx, ry, rz = np.radians(rotation_angles)
    
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations (order: Z, Y, X)
    return np.dot(Rx, np.dot(Ry, Rz))

def is_leg_joint(joint_name):
    """Check if a joint is part of the legs."""
    leg_keywords = ['upleg', 'leg', 'foot', 'toe', 'thigh', 'knee', 'ankle', 'heel']
    return 'right' in joint_name.lower() and any(keyword in joint_name.lower() for keyword in leg_keywords) or \
           'left' in joint_name.lower() and any(keyword in joint_name.lower() for keyword in leg_keywords)

def is_spine_joint(joint_name):
    """Check if a joint is part of the spine."""
    return 'spine' in joint_name.lower() or 'neck' in joint_name.lower() or 'head' in joint_name.lower()

def is_shoulder_joint(joint_name):
    """Check if joint is a shoulder or clavicle."""
    return ('shoulder' in joint_name.lower() or 'clavicle' in joint_name.lower()) and \
           ('left' in joint_name.lower() or 'right' in joint_name.lower())

def get_direct_parent_child_relationships(joints):
    """Get direct parent-child relationships for all joints."""
    relationships = []
    for joint_name, joint_data in joints.items():
        if joint_data['parent'] is not None:
            relationships.append((joint_data['parent'], joint_name))
    return relationships

def fix_skeleton_orientation(skeletons, joints, plane='XY'):
    """
    Fix the skeleton orientation to face directly toward the viewer:
    1. Position all spine joints at x=0 (horizontally centered)
    2. Stack spine joints vertically, maintaining their original distances
    3. Position shoulders horizontally level with equal height, facing forward
    4. Adjust other joints accordingly to maintain relative positions
    """
    # Find spine and neck related joints in the first skeleton
    spine_related_joints = []
    for joint_name in skeletons[0].keys():
        if is_spine_joint(joint_name):
            spine_related_joints.append(joint_name)
    
    # Find shoulder joints
    shoulder_joints = []
    for joint_name in skeletons[0].keys():
        if is_shoulder_joint(joint_name):
            shoulder_joints.append(joint_name)
    
    # Get direct parent-child relationships
    parent_child_pairs = get_direct_parent_child_relationships(joints)
    
    # Find direct connections between spine and shoulders
    spine_shoulder_connections = []
    for parent, child in parent_child_pairs:
        if parent in spine_related_joints and child in shoulder_joints:
            spine_shoulder_connections.append((parent, child))
        elif parent in shoulder_joints and child in spine_related_joints:
            spine_shoulder_connections.append((child, parent))
    
    # Sort spine joints based on hierarchy (from root upward)
    def get_joint_depth(joint_name):
        """Calculate the depth of a joint in the hierarchy"""
        depth = 0
        current = joint_name
        while current in joints and joints[current]['parent'] is not None:
            depth += 1
            current = joints[current]['parent']
        return depth
    
    # Sort spine joints by depth in hierarchy
    spine_related_joints.sort(key=get_joint_depth)
    
    if not spine_related_joints:
        print("Warning: No spine or neck joints found!")
        return skeletons
    
    # Set which axis should be vertical based on viewing plane
    if plane == 'XY':
        vertical_axis = 1  # Y-axis is vertical
        horizontal_axis = 0  # X-axis is horizontal
        depth_axis = 2      # Z-axis is depth
    elif plane == 'XZ':
        vertical_axis = 2  # Z-axis is vertical
        horizontal_axis = 0  # X-axis is horizontal
        depth_axis = 1      # Y-axis is depth
    else:  # YZ
        vertical_axis = 2  # Z-axis is vertical
        horizontal_axis = 1  # Y-axis is horizontal
        depth_axis = 0      # X-axis is depth
    
    # Calculate average positions for spine joints
    avg_positions = {}
    for joint_name in spine_related_joints + shoulder_joints:
        positions = np.array([skeleton[joint_name]['position'] for skeleton in skeletons if joint_name in skeleton])
        if len(positions) > 0:
            avg_positions[joint_name] = np.mean(positions, axis=0)
    
    # Find the base spine joint (closest to root)
    base_spine_joint = spine_related_joints[0]
    base_position = avg_positions[base_spine_joint].copy()
    
    # Create new centered and vertical positions for spine joints
    vertical_positions = {}
    
    # Set the base spine joint at position x=0, keeping its original height
    base_position[horizontal_axis] = 0.0  # Center horizontally
    base_position[depth_axis] = 0.0      # Set depth to zero (facing forward)
    vertical_positions[base_spine_joint] = base_position.copy()
    
    # For each subsequent spine joint, calculate distance to previous joint
    # and place it directly above while maintaining the original distances
    for i in range(1, len(spine_related_joints)):
        prev_joint = spine_related_joints[i-1]
        curr_joint = spine_related_joints[i]
        
        if prev_joint in avg_positions and curr_joint in avg_positions:
            # Original positions
            prev_pos = avg_positions[prev_joint]
            curr_pos = avg_positions[curr_joint]
            
            # Calculate original vertical distance between joints
            distance = np.abs(curr_pos[vertical_axis] - prev_pos[vertical_axis])
            
            # Create new position by moving up from previous joint
            new_pos = vertical_positions[prev_joint].copy()
            new_pos[horizontal_axis] = 0.0  # Keep centered at x=0
            new_pos[depth_axis] = 0.0       # Set depth to zero (facing forward)
            new_pos[vertical_axis] += distance  # Move up by the original distance
            
            vertical_positions[curr_joint] = new_pos
    
    # Determine which spine joint connects to each shoulder
    shoulder_parents = {}
    for joint_name in shoulder_joints:
        if joint_name in joints and joints[joint_name]['parent'] in spine_related_joints:
            shoulder_parents[joint_name] = joints[joint_name]['parent']
    
    # Find left and right shoulders
    left_shoulder = None
    right_shoulder = None
    for joint in shoulder_joints:
        if 'left' in joint.lower():
            left_shoulder = joint
        elif 'right' in joint.lower():
            right_shoulder = joint
    
    # If we found both shoulders, position them horizontally at the same height
    if left_shoulder and right_shoulder:
        # Get the parent spine joints for each shoulder
        left_parent = shoulder_parents.get(left_shoulder, spine_related_joints[-1])  # Default to top spine joint
        right_parent = shoulder_parents.get(right_shoulder, spine_related_joints[-1])
        
        # Using the actual parent-child relationships from the BVH
        if left_parent in vertical_positions and right_parent in vertical_positions:
            # Calculate original offsets from parent to shoulders
            if left_shoulder in avg_positions and right_shoulder in avg_positions:
                left_offset = np.array(avg_positions[left_shoulder]) - np.array(avg_positions[left_parent])
                right_offset = np.array(avg_positions[right_shoulder]) - np.array(avg_positions[right_parent])
                
                # Calculate shoulder horizontal distance (use maximum of left and right shoulder distances)
                left_distance = np.abs(left_offset[horizontal_axis])
                right_distance = np.abs(right_offset[horizontal_axis])
                shoulder_distance = max(left_distance, right_distance) * 2  # Ensure symmetry
                
                # Get average shoulder height relative to their parent
                left_vertical_offset = left_offset[vertical_axis]
                right_vertical_offset = right_offset[vertical_axis]
                avg_vertical_offset = (left_vertical_offset + right_vertical_offset) / 2
                
                # Set new shoulder positions based on parent spine positions
                left_pos = vertical_positions[left_parent].copy()
                right_pos = vertical_positions[right_parent].copy()
                
                # Both shoulders should be at the same height
                left_pos[vertical_axis] = vertical_positions[left_parent][vertical_axis] + avg_vertical_offset
                right_pos[vertical_axis] = vertical_positions[right_parent][vertical_axis] + avg_vertical_offset
                
                # Set horizontal positions to be symmetric from center
                left_pos[horizontal_axis] = -shoulder_distance/2
                right_pos[horizontal_axis] = shoulder_distance/2
                
                # Set depth to zero (facing forward)
                left_pos[depth_axis] = 0.0
                right_pos[depth_axis] = 0.0
                
                # Save the positioned shoulders
                vertical_positions[left_shoulder] = left_pos
                vertical_positions[right_shoulder] = right_pos
    
    # Adjust all skeletons to use the new fixed positions
    fixed_skeletons = []
    for skeleton in skeletons:
        fixed_skeleton = {}
        
        # Find a reference spine joint to use for calculating the offset
        ref_joint = next((j for j in spine_related_joints if j in skeleton), None)
        if ref_joint is None:
            # If no spine joints in this skeleton, just copy it
            fixed_skeleton = skeleton.copy()
        else:
            # Calculate offset needed to move reference joint to its vertical position
            current_pos = np.array(skeleton[ref_joint]['position'])
            target_pos = vertical_positions[ref_joint]
            offset = target_pos - current_pos
            
            # Apply the offset to all joints (except spine joints which get exact positions)
            for joint_name, joint_data in skeleton.items():
                # Skip leg joints
                if is_leg_joint(joint_name):
                    continue
                    
                fixed_joint = joint_data.copy()
                
                # For spine and shoulder joints, use their centered positions
                if joint_name in vertical_positions:
                    fixed_joint['position'] = vertical_positions[joint_name].tolist()
                # For all other joints, apply offsets to maintain relative positioning
                else:
                    current_pos = np.array(joint_data['position'])
                    fixed_pos = current_pos + offset
                    
                    # Set depth to zero for ALL joints to ensure a completely flat skeleton
                    fixed_pos[depth_axis] = 0.0
                    
                    fixed_joint['position'] = fixed_pos.tolist()
                
                fixed_skeleton[joint_name] = fixed_joint
        
        fixed_skeletons.append(fixed_skeleton)
    
    return fixed_skeletons

def plot_skeleton_2d(skeleton, joints, ax, frame_num, plane='XY'):
    """Plot the skeleton in 2D on a specified plane."""
    # Set title and labels
    ax.clear()
    ax.set_title(f'Frame {frame_num}')
    
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
    
    # Draw connections between joints
    for joint_name, joint_data in joints.items():
        # Skip leg joints
        if is_leg_joint(joint_name):
            continue
            
        if joint_data['parent'] is not None:
            # Skip connections to leg joints
            if is_leg_joint(joint_data['parent']):
                continue
                
            if joint_name in skeleton and joint_data['parent'] in skeleton:
                child_pos = skeleton[joint_name]['position']
                parent_pos = skeleton[joint_data['parent']]['position']
                
                # Determine color for hands
                color = 'black'  # Default color
                
                # Color for spine and neck joints
                if is_spine_joint(joint_name):
                    color = 'green'
                # Color for shoulder connections
                elif is_shoulder_joint(joint_name) or (is_shoulder_joint(joint_data['parent'])):
                    color = 'purple'
                # Check if this joint is part of the right hand
                elif 'Right' in joint_name and any(hand_part in joint_name for hand_part in ['Hand', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
                    color = 'red'
                # Check if this joint is part of the left hand
                elif 'Left' in joint_name and any(hand_part in joint_name for hand_part in ['Hand', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
                    color = 'blue'
                
                ax.plot([child_pos[x_idx], parent_pos[x_idx]], 
                        [child_pos[y_idx], parent_pos[y_idx]], 
                        color=color, linewidth=2)
    
    # Draw joints
    for joint_name, joint_data in skeleton.items():
        # Skip leg joints
        if is_leg_joint(joint_name):
            continue
            
        pos = joint_data['position']
        
        # Determine color for hands
        color = 'black'  # Default color
        
        # Color for spine and neck joints
        if is_spine_joint(joint_name):
            color = 'green'
        # Color for shoulder joints
        elif is_shoulder_joint(joint_name):
            color = 'purple'
        # Check if this joint is part of the right hand
        elif 'Right' in joint_name and any(hand_part in joint_name for hand_part in ['Hand', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
            color = 'red'
        # Check if this joint is part of the left hand
        elif 'Left' in joint_name and any(hand_part in joint_name for hand_part in ['Hand', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
            color = 'blue'
        
        ax.scatter(pos[x_idx], pos[y_idx], color=color, s=20)
    
    # Draw vertical center line to show spine alignment
    y_min, y_max = ax.get_ylim()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Print joint names for debugging
    if frame_num == 0:  # Only print for the first frame
        print("Joint hierarchy:")
        for joint_name, joint_data in joints.items():
            if not is_leg_joint(joint_name):  # Skip leg joints
                parent = joint_data['parent'] if joint_data['parent'] else "ROOT"
                print(f"{joint_name} → parent: {parent}")

def visualize_bvh(file_path, num_frames=5, start_frame=0, step=1, plane='XY'):
    """Visualize frames of a BVH file."""
    joints, motion_data, total_frames, frame_time = parse_bvh(file_path)
    
    # Create a figure with subplots for each frame
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    if num_frames == 1:
        axes = [axes]
    
    # Select frames to visualize
    frame_indices = []
    for i in range(num_frames):
        frame_idx = min(start_frame + i * step, total_frames - 1)
        frame_indices.append(frame_idx)
    
    # Process all selected frames
    skeletons = []
    for frame_idx in frame_indices:
        skeleton = create_skeleton_from_frame(joints, motion_data, frame_idx)
        skeletons.append(skeleton)
    
    # Fix skeleton orientation to face directly toward the viewer
    fixed_skeletons = fix_skeleton_orientation(skeletons, joints, plane)
    
    # Visualize each frame with fixed spine and shoulders
    for i, skeleton in enumerate(fixed_skeletons):
        # Plot the skeleton
        plot_skeleton_2d(skeleton, joints, axes[i], frame_indices[i], plane)
    
    plt.tight_layout()
    plt.savefig(f'bvh_skeleton_forward_facing.png', dpi=300)
    plt.show()

def visualize_stacked_frames(file_path, plane='XY', downsample_factor=1, trail_alpha=0.05):
    """Visualize all frames of a BVH file stacked together to show motion trajectory."""
    joints, motion_data, total_frames, frame_time = parse_bvh(file_path)
    
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
    
    # Create and fix all skeletons
    skeletons = []
    for frame_idx in range(0, total_frames, downsample_factor):
        skeleton = create_skeleton_from_frame(joints, motion_data, frame_idx)
        skeletons.append(skeleton)
    
    # Fix orientation for all skeletons
    fixed_skeletons = fix_skeleton_orientation(skeletons, joints, plane)
    
    # Helper function to check if a joint is part of a hand
    def is_hand_joint(joint_name):
        hand_keywords = ['hand', 'thumb', 'index', 'middle', 'ring', 'pinky', 'finger']
        return any(keyword in joint_name.lower() for keyword in hand_keywords)
    
    # Helper function to check if a connection is between hand joints
    def is_hand_connection(joint1, joint2):
        return is_hand_joint(joint1) and is_hand_joint(joint2)
    
    # Store hand joint positions for point cloud visualization
    right_hand_points_x = []
    right_hand_points_y = []
    right_hand_alphas = []
    right_hand_colors = []
    
    left_hand_points_x = []
    left_hand_points_y = []
    left_hand_alphas = []
    left_hand_colors = []
    
    # Plot all frames with decreasing alpha to show progression
    for frame_idx, skeleton in enumerate(fixed_skeletons):
        # Calculate alpha based on frame index - more recent frames are more opaque
        alpha = 0.3
        
        # Draw connections between joints
        for joint_name, joint_data in joints.items():
            # Skip leg joints
            if is_leg_joint(joint_name):
                continue
                
            # For hand joints, collect the point position for the point cloud
            if is_hand_joint(joint_name) and joint_name in skeleton:
                pos = skeleton[joint_name]['position']
                
                # Determine if it's right or left hand
                if 'right' in joint_name.lower():
                    right_hand_points_x.append(pos[x_idx])
                    right_hand_points_y.append(pos[y_idx])
                    right_hand_alphas.append(alpha)
                    right_hand_colors.append('red')
                elif 'left' in joint_name.lower():
                    left_hand_points_x.append(pos[x_idx])
                    left_hand_points_y.append(pos[y_idx])
                    left_hand_alphas.append(alpha)
                    left_hand_colors.append('blue')
                
            # Draw connections between joints (except hand-to-hand connections)
            if joint_data['parent'] is not None:
                # Skip connections to leg joints
                if is_leg_joint(joint_data['parent']):
                    continue
                    
                if joint_name in skeleton and joint_data['parent'] in skeleton:
                    child_pos = skeleton[joint_name]['position']
                    parent_pos = skeleton[joint_data['parent']]['position']
                    
                    # Skip hand-to-hand connections as we'll visualize them as point clouds
                    if is_hand_connection(joint_name, joint_data['parent']):
                        continue
                    
                    # Determine color for other connections
                    color = 'black'  # Default color
                    
                    # Color for spine and neck joints
                    if is_spine_joint(joint_name):
                        color = 'green'
                    # Color for shoulder connections
                    elif is_shoulder_joint(joint_name) or (is_shoulder_joint(joint_data['parent'])):
                        color = 'purple'
                    # Check if this is a connection to a hand joint (but not between hand joints)
                    elif is_hand_joint(joint_name) and not is_hand_joint(joint_data['parent']):
                        color = 'red' if 'right' in joint_name.lower() else 'blue'
                    elif is_hand_joint(joint_data['parent']) and not is_hand_joint(joint_name):
                        color = 'red' if 'right' in joint_data['parent'].lower() else 'blue'
                    
                    ax.plot([child_pos[x_idx], parent_pos[x_idx]], 
                            [child_pos[y_idx], parent_pos[y_idx]], 
                            color=color, alpha=0.1, linewidth=1)
    
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
    ax.set_title(f'All Frames Stacked - {os.path.basename(file_path)}')
    ax.set_xlim(-100, 100)
    ax.set_ylim(50, 190)
    plt.tight_layout()
    output_file = file_path.replace('.bvh', f'_stacked_{plane}_hand_pointcloud.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved stacked visualization to {output_file}")
    # plt.show()

if __name__ == "__main__":
    # Replace with the path to your BVH file
    # bvh_dir_path = "/home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion/"
    # bvh_file_paths = [os.path.join(bvh_dir_path, f) for f in os.listdir(bvh_dir_path) if f.endswith('.bvh')]
    # for bvh_file_path in bvh_file_paths:
    #     visualize_stacked_frames(bvh_file_path, plane='XY', downsample_factor=2,trail_alpha=0.05)
    
    # Uncomment the visualization method you want to use:
    bvh_file_path= "/home/bsd/cospeech/DiffGesture/output/train_diffusion_expressive/custom_output/TestSeq001.bvh"
    # Option 1: Original visualization with separate frames
    # visualize_bvh(bvh_file_path, num_frames=5, start_frame=0, step=200, plane='XY')
    
    # Option 2: Stacked visualization of all frames
    visualize_stacked_frames(bvh_file_path, plane='XY', downsample_factor=2,trail_alpha=0.05)