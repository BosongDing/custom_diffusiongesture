import json
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

class BodyTrackingToMP4:
    def __init__(self, json_path, audio_path=None):
        # Load the JSON data
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract frames and bone references
        self.frames = self.data.get('frames', [])
        self.bone_references = self.data.get('boneReferences', [])
        
        # Number of frames
        self.num_frames = len(self.frames)
        print(f"Loaded {self.num_frames} frames from {json_path}")
        
        # Create index for frame times
        self.frame_times = [frame['time'] for frame in self.frames]
        
        # Audio path
        self.audio_path = audio_path
        
        # Cache for joint parent mappings (for efficiency)
        self.parent_mappings = {}
        self._build_parent_mappings()
    
    def _build_parent_mappings(self):
        """Build a cache of parent mappings from the first frame"""
        if not self.frames:
            return
        
        first_frame = self.frames[0]
        for joint in first_frame['joints']:
            joint_name = joint['name']
            parent_name = joint['parentName']
            self.parent_mappings[joint_name] = parent_name
    
    def get_joint_positions(self, frame_idx):
        """Extract joint positions for a specific frame"""
        if frame_idx >= len(self.frames):
            return {}, {}
        
        frame = self.frames[frame_idx]
        positions = {}
        rotations = {}
        
        for joint in frame['joints']:
            name = joint['name']
            pos = joint['position']
            rot = joint['rotation']
            
            # Store positions as numpy arrays for easier manipulation
            positions[name] = np.array([pos['x'], pos['y'], pos['z']])
            rotations[name] = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
        
        return positions, rotations
    
    def update_frame(self, frame_idx):
        """Update function for animation"""
        positions, _ = self.get_joint_positions(frame_idx)
        
        if not positions:
            return self.ax
        
        self.ax.clear()
        
        # Add joints as scatter points
        joint_x = []
        joint_y = []
        joint_z = []
        
        # First, collect all joint positions
        for joint_name, position in positions.items():
            joint_x.append(position[0])
            joint_y.append(position[1])
            joint_z.append(position[2])
        
        # Calculate the center and size of the skeleton for proper viewing
        all_points = np.array([joint_x, joint_y, joint_z]).T
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.ptp(all_points, axis=0)) * 0.6
        
        # Draw bones first (so they appear behind the joints)
        for joint_name, position in positions.items():
            parent_name = self.parent_mappings.get(joint_name)
            if parent_name != "none" and parent_name in positions:
                parent_pos = positions[parent_name]
                self.ax.plot([position[0], parent_pos[0]], 
                        [position[1], parent_pos[1]], 
                        [position[2], parent_pos[2]], 
                        'r-', linewidth=2)
        
        # Now draw the joints
        self.ax.scatter(joint_x, joint_y, joint_z, c='blue', s=20)
        
        # Set axis limits to keep the view consistent across frames
        self.ax.set_xlim([center[0]-max_range, center[0]+max_range])
        self.ax.set_ylim([center[1]-max_range, center[1]+max_range])
        self.ax.set_zlim([center[2]-max_range, center[2]+max_range])
        
        # Set title and labels
        self.ax.set_title(f'Frame {frame_idx} (Time: {self.frames[frame_idx]["time"]:.2f}s)')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set consistent view angle
        self.ax.view_init(elev=20, azim=45)
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        return self.ax
    
    def create_mp4(self, output_path, fps=30, dpi=100):
        """Create an MP4 video of the animation with audio using FFmpeg"""
        print(f"Creating animation with {self.num_frames} frames at {fps} FPS...")
        
        # Create a figure for the animation
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create animation
        anim = FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=tqdm(range(self.num_frames)), 
            interval=1000/fps, 
            blit=False
        )
        
        # Save animation as video without audio
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        
        print(f"Saving animation to temporary file: {temp_video_path}")
        writer = FFMpegWriter(fps=fps, metadata=dict(title='Body Tracking Animation'), bitrate=5000)
        anim.save(temp_video_path, writer=writer, dpi=dpi)
        
        # If audio is provided, combine video with audio using FFmpeg
        if self.audio_path and os.path.exists(self.audio_path):
            print(f"Combining video with audio from {self.audio_path}...")
            
            # Use FFmpeg to combine video and audio
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', self.audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',  # End when the shortest input stream ends
                output_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"Video with audio saved to {output_path}")
                # Remove temporary video
                os.remove(temp_video_path)
            except subprocess.CalledProcessError as e:
                print(f"Error combining video and audio: {e}")
                print(f"Using video without audio: {temp_video_path}")
                os.rename(temp_video_path, output_path)
        else:
            # If no audio, just rename the temp video
            os.rename(temp_video_path, output_path)
            print(f"Video saved to {output_path}")
        
        plt.close(self.fig)

# Example usage
if __name__ == '__main__':
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Create MP4 video from body tracking JSON with audio')
    parser.add_argument('--json', help='Path to the body tracking JSON file')
    parser.add_argument('--audio', help='Path to the audio file to combine with the video')
    parser.add_argument('--output', default='body_tracking_animation.mp4', help='Path to save the output video')
    parser.add_argument('--fps', type=int, default=70, help='Frames per second for the animation')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for the output video')
    
    # Parse arguments
    args = parser.parse_args()
    args.json = "/home/bsd/cospeech/DiffGesture/data/quest/voice_20250427_143848_motion_0.json"
    args.audio = "/home/bsd/cospeech/DiffGesture/data/quest/voice_20250427_143848_speech_0.wav"
    # Check if JSON file exists
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found at {args.json}")
        exit(1)
    
    # Check if audio file exists
    audio_path = args.audio
    if audio_path and not os.path.exists(audio_path):
        print(f"Warning: Audio file not found at {audio_path}")
        print("Proceeding without audio...")
        audio_path = None
    
    # Create the MP4
    visualizer = BodyTrackingToMP4(args.json, audio_path)
    visualizer.create_mp4(args.output, fps=args.fps, dpi=args.dpi)
    
    print("Done!")
        
        
