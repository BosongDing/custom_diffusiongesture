import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

class BodyTrackingVisualizer:
    def __init__(self, json_path):
        # Load the JSON data
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
    
    def generate_skeleton_figure(self, frame_idx):
        """Generate a 3D figure of the skeleton for a specific frame"""
        positions, _ = self.get_joint_positions(frame_idx)
        
        if not positions:
            return go.Figure()
        
        # Create lists to hold the lines between joints
        lines_x, lines_y, lines_z = [], [], []
        
        # Add joints as scatter points
        joint_x = []
        joint_y = []
        joint_z = []
        joint_names = []
        
        for joint_name, position in positions.items():
            joint_x.append(position[0])
            joint_y.append(position[1])
            joint_z.append(position[2])
            joint_names.append(joint_name)
            
            # If joint has a parent, draw a line to it
            parent_name = self.parent_mappings.get(joint_name)
            if parent_name != "none" and parent_name in positions:
                parent_pos = positions[parent_name]
                
                # Add the line segments (joint to parent)
                lines_x.extend([position[0], parent_pos[0], None])
                lines_y.extend([position[1], parent_pos[1], None])
                lines_z.extend([position[2], parent_pos[2], None])
        
        # Create the figure
        fig = go.Figure()
        
        # Add the joints as scatter points
        fig.add_trace(go.Scatter3d(
            x=joint_x, y=joint_y, z=joint_z,
            mode='markers',
            marker=dict(size=5, color='blue'),
            text=joint_names,
            hoverinfo='text',
            name='Joints'
        ))
        
        # Add the bones as lines
        fig.add_trace(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='red', width=3),
            hoverinfo='none',
            name='Bones'
        ))
        
        # Calculate the center and size of the skeleton for proper viewing
        all_points = np.array([joint_x, joint_y, joint_z]).T
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.ptp(all_points, axis=0))
        
        # Set layout properties
        fig.update_layout(
            title=f'Skeleton Visualization - Frame {frame_idx} (Time: {self.frames[frame_idx]["time"]:.2f}s)',
            scene=dict(
                xaxis=dict(range=[center[0]-max_range/2, center[0]+max_range/2]),
                yaxis=dict(range=[center[1]-max_range/2, center[1]+max_range/2]),
                zaxis=dict(range=[center[2]-max_range/2, center[2]+max_range/2]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        return fig

    def create_dashboard(self):
        """Create a Dash app for interactive visualization"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Body Tracking Visualization"),
            
            html.Div([
                html.Label("Frame:"),
                dcc.Slider(
                    id='frame-slider',
                    min=0,
                    max=len(self.frames)-1,
                    value=0,
                    marks={i: str(i) for i in range(0, len(self.frames), max(1, len(self.frames)//10))},
                    step=1
                )
            ]),
            
            html.Div([
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Stop', id='stop-button', n_clicks=0)
            ]),
            
            dcc.Graph(id='skeleton-graph', style={'height': '800px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=14.28,  # in milliseconds (default 10 FPS)
                n_intervals=0,
                disabled=True
            )
        ])
        
        @app.callback(
            Output('skeleton-graph', 'figure'),
            [Input('frame-slider', 'value')]
        )
        def update_graph(frame_idx):
            return self.generate_skeleton_figure(frame_idx)
        
        @app.callback(
            [Output('interval-component', 'disabled'),
             Output('frame-slider', 'value')],
            [Input('play-button', 'n_clicks'),
             Input('stop-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [Input('frame-slider', 'value')]
        )
        def control_playback(play_clicks, stop_clicks, n_intervals, current_frame):
            ctx = dash.callback_context
            if not ctx.triggered:
                return True, current_frame
            
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if triggered_id == 'play-button':
                return False, current_frame
            elif triggered_id == 'stop-button':
                return True, current_frame
            elif triggered_id == 'interval-component':
                next_frame = (current_frame + 1) % len(self.frames)
                return False, next_frame
            
            return True, current_frame
        
        return app

# Example usage
if __name__ == '__main__':
    # Path to your JSON file - change this to your actual path
    json_file_path = "/home/bsd/cospeech/DiffGesture/data/quest/voice_20250427_143848_motion_0.json"
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
    else:
        visualizer = BodyTrackingVisualizer(json_file_path)
        app = visualizer.create_dashboard()
        print("Starting visualization server. Open a web browser and navigate to http://127.0.0.1:8050/")
        app.run(debug=True)