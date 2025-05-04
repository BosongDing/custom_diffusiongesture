import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import lmdb
import pyarrow
import json
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import wavfile

# Reuse the custom_deserialize function from your original code
def custom_deserialize(value):
    """Custom deserialization function to handle specific PyArrow errors"""
    try:
        # First try standard pyarrow deserialization
        return pyarrow.deserialize(value)
    except pyarrow.lib.ArrowSerializationError as e:
        error_msg = str(e)
        print(f"Handling error: {error_msg}")
        
        # Case 1: Empty string to int8_t conversion error
        if "Cannot convert string" in error_msg and "to int8_t" in error_msg:
            try:
                # Try to decode as JSON
                decoded = value.decode('utf-8')
                return json.loads(decoded)
            except:
                pass
        
        # Case 2: Sparse tensor vs tensor error
        if "Expected IPC message of type sparse tensor but got tensor" in error_msg:
            try:
                # Try using pickle instead of pyarrow
                return pickle.loads(value)
            except:
                pass
        
        # Generic fallback for any PyArrow error
        try:
            # Try pickle as a fallback
            return pickle.loads(value)
        except:
            pass
            
        try:
            # Try to extract raw numpy data
            return {
                'vid': 'unknown',
                'clips': [{
                    'skeletons': np.frombuffer(value, dtype=np.float32).reshape(-1, 24)
                }]
            }
        except:
            pass
            
        # If all else fails, return a minimal valid structure
        print(f"Could not deserialize data, returning empty structure")
        return {'vid': 'unknown', 'clips': []}

def load_lmdb_data(lmdb_path, max_samples=5):
    """Load data from an LMDB database with robust error handling"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    dataset = []
    
    with env.begin() as txn:
        # Get the number of entries in the database
        num_entries = txn.stat()['entries']
        print(f"Found {num_entries} entries in the LMDB database")
        
        # Get all keys first
        keys = []
        cursor = txn.cursor()
        for key, _ in cursor:
            keys.append(key)
        
        print(f"Found {len(keys)} keys in the database")
        
        # Try different deserialization methods
        samples_loaded = 0
        for i, key in enumerate(keys):
            if samples_loaded >= max_samples:
                break
                
            value = txn.get(key)
            
            if value is None:
                print(f"No data found for key: {key}")
                continue
                
            try:
                # Try our custom deserialize function
                data = custom_deserialize(value)
                
                # Verify the data has the expected structure
                if 'vid' in data and 'clips' in data and len(data['clips']) > 0:
                    dataset.append(data)
                    print(f"Successfully deserialized entry {i+1}/{len(keys)}")
                    samples_loaded += 1
                else:
                    print(f"Entry {i+1}/{len(keys)} has invalid structure")
                    
            except Exception as e:
                print(f"Error processing entry {i+1}/{len(keys)}: {e}")
    
    env.close()
    print(f"Successfully loaded {len(dataset)} entries")
    return dataset

def create_colored_visualization(motion_data, output_path, title, split_name):
    """
    Create a visualization similar to the provided reference image
    with colored points representing left (blue) and right (red) sides.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a figure with multiple subplots (3x3 grid)
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"{title} - {split_name}", fontsize=16)
    
    # Flatten the axes for easy iteration
    axs = axs.flatten()
    
    # Define a custom colormap for the visualization
    # Blue to Red colormap similar to the reference image
    colors = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
    cmap_br = LinearSegmentedColormap.from_list("BlueRed", colors)
    
    # Process up to 9 samples (or fewer if less available)
    for i in range(min(9, len(motion_data))):
        data = motion_data[i]
        
        # Get the first clip for this sample
        if len(data['clips']) == 0:
            continue
            
        clip = data['clips'][0]
        
        # Check if skeleton data is available
        if 'skeletons' not in clip or clip['skeletons'] is None or len(clip['skeletons']) == 0:
            continue
            
        # Get the skeleton data
        skeletons = np.array(clip['skeletons'])
        
        # If 3D data is available, use it
        if 'skeletons_3d' in clip and clip['skeletons_3d'] is not None and len(clip['skeletons_3d']) > 0:
            skeletons_3d = np.array(clip['skeletons_3d'])
        else:
            # If only 2D data is available, we'll just use that
            skeletons_3d = None
        
        # Create a density plot similar to the reference image
        ax = axs[i]
        
        # If we have any skeleton data
        if skeletons is not None and len(skeletons) > 0:
            # Determine how many frames to use for visualization (use all frames)
            num_frames = len(skeletons)
            
            # Get the number of keypoints
            num_keypoints = skeletons[0].shape[0] if skeletons[0].ndim > 0 else 0
            if num_keypoints == 0:
                continue
                
            # Check the shape and format of the skeleton data
            # Flatten to 2D if needed
            if skeletons.ndim > 3:
                skeletons = skeletons.reshape(num_frames, -1, 2)
            
            # Plot the base skeleton outline (simplified - adjust based on your skeleton format)
            # This is a generic approach, you'll need to adjust based on your specific skeleton format
            
            # Assuming keypoints are ordered starting from center and branching out to limbs
            # Plot a simplified skeleton from one frame (e.g., first frame)
            frame_to_plot = skeletons[0]
            
            # Draw a simple stick figure based on your skeleton format
            # This is a very simplified version and needs to be adapted to your skeleton format
            center_idx = 0  # Assuming center of body is first point
            
            # Create connections for a basic humanoid skeleton
            # These connections need to be adjusted for your specific skeleton structure
            connections = []
            
            # Try to infer some basic connections based on number of keypoints
            if num_keypoints >= 15:  # COCO-like format possibly
                # A very simplified skeleton
                connections = [
                    (0, 1), (1, 2),  # Center to head
                    (0, 3), (3, 4), (4, 5),  # Right arm
                    (0, 6), (6, 7), (7, 8),  # Left arm
                    (0, 9), (9, 10), (10, 11),  # Right leg
                    (0, 12), (12, 13), (13, 14)  # Left leg
                ]
            elif num_keypoints >= 8:  # Very simplified
                connections = [
                    (0, 1),  # Center to head
                    (0, 2), (2, 3),  # Right arm
                    (0, 4), (4, 5),  # Left arm
                    (0, 6), (0, 7)   # Legs
                ]
            
            # Create a scatterplot for all frames to show motion over time
            # Left side (blue) and right side (red)
            
            # Split keypoints into left and right sides
            # This is an approximation and needs to be adjusted for your data
            mid_idx = num_keypoints // 2
            
            # Collect all x, y coordinates across frames for left and right sides
            left_x, left_y = [], []
            right_x, right_y = [], []
            
            for frame in skeletons:
                # Assuming first half of keypoints are left side, second half are right side
                # This is a simplification and might need adjustment
                left_indices = list(range(mid_idx))
                right_indices = list(range(mid_idx, num_keypoints))
                
                # Add all left side keypoints from this frame
                left_x.extend(frame[left_indices, 0].flatten())
                left_y.extend(frame[left_indices, 1].flatten())
                
                # Add all right side keypoints from this frame
                right_x.extend(frame[right_indices, 0].flatten())
                right_y.extend(frame[right_indices, 1].flatten())
            
            # Plot density of movements for left and right sides
            ax.scatter(left_x, left_y, c='blue', alpha=0.2, s=5)
            ax.scatter(right_x, right_y, c='red', alpha=0.2, s=5)
            
            # Plot the base skeleton
            for conn in connections:
                if conn[0] < len(frame_to_plot) and conn[1] < len(frame_to_plot):
                    ax.plot(
                        [frame_to_plot[conn[0], 0], frame_to_plot[conn[1], 0]],
                        [frame_to_plot[conn[0], 1], frame_to_plot[conn[1], 1]],
                        'k-', linewidth=2
                    )
            
            # Plot the head (if we have enough keypoints)
            if num_keypoints >= 3:
                head_idx = 1  # Assuming second point is head
                ax.scatter(frame_to_plot[head_idx, 0], frame_to_plot[head_idx, 1], 
                           c='black', s=50, zorder=3)
        
        # Set equal aspect ratio and remove axis ticks for cleaner visualization
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sample {i+1}")
    
    # Remove any unused subplots
    for i in range(len(motion_data), 9):
        fig.delaxes(axs[i])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

def visualize_datasets(base_path, output_dir):
    """
    Visualize samples from train, validation, and test datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset splits to process
    splits = ['train', 'val', 'test']
    splits = ['test']
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split} split not found at {split_path}")
            continue
            
        print(f"Processing {split} dataset from {split_path}")
        
        # Load samples from this split
        dataset = load_lmdb_data(split_path, max_samples=9)  # Load up to 9 samples
        
        if not dataset:
            print(f"No valid data found in {split} dataset")
            continue
            
        # Create visualization
        output_path = os.path.join(output_dir, f"{split}_visualization.png")
        create_colored_visualization(
            dataset, 
            output_path, 
            title="Motion Data Visualization", 
            split_name=split.capitalize()
        )
        
        print(f"Completed visualization for {split} dataset")
    
    print("All visualizations complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motion data from LMDB datasets")
    parser.add_argument("--data_path", type=str, default="/home/bsd/cospeech/DiffGesture/data/ted_expressive_dataset",
                        help="Base path containing train, val, test LMDB directories")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    print(f"Starting visualization process for datasets in {args.data_path}")
    visualize_datasets(args.data_path, args.output_dir)