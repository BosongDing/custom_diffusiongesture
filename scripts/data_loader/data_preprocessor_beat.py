""" create data samples for BEAT dataset """
import logging
import os
import math
import numpy as np
import lmdb
import pyarrow
import librosa
from scipy.interpolate import interp1d
from tqdm import tqdm

def custom_serialize(data):
    """Convert numpy arrays to lists before serialization"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [custom_serialize(item) for item in data]
    elif isinstance(data, dict):
        return {k: custom_serialize(v) for k, v in data.items()}
    else:
        return data

def resample_pose_seq(poses, original_fps, target_fps):
    """Resample pose sequence to target frame rate"""
    n_frames = len(poses)
    duration_sec = n_frames / original_fps
    target_frames = int(duration_sec * target_fps)
     
    # Create time points for interpolation
    src_times = np.arange(n_frames) / original_fps
    target_times = np.arange(target_frames) / target_fps
    
    # Create interpolation function
    interp_func = interp1d(
        src_times, 
        poses.reshape(n_frames, -1), 
        axis=0, 
        kind='linear', 
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    # Interpolate
    resampled = interp_func(target_times)
    return resampled.reshape(target_frames, poses.shape[1], poses.shape[2])

def extract_melspectrogram(y, sr=16000):
    """Extract mel spectrogram from audio"""
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec

def make_audio_fixed_length(audio, expected_length):
    """Ensure audio has the expected length"""
    n_padding = expected_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_length]
    return audio

def calc_spectrogram_length_from_motion_length(n_frames, fps):
    """Calculate expected spectrogram length from motion length"""
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))

class BeatDataPreprocessor:
    def __init__(self, data_root_dir, out_lmdb_dir, n_poses, subdivision_stride,
                 original_fps, target_fps, mean_dir_vec=None, user_ids=None):
        self.data_root_dir = data_root_dir
        self.out_lmdb_dir = out_lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.original_fps = original_fps
        self.target_fps = target_fps
        self.mean_dir_vec = mean_dir_vec
        self.user_ids = user_ids  # List of user IDs to process
        
        # Get all NPZ files from the specified users
        self.npz_files = []
        if user_ids is None:
            # If no specific users, use all directories
            user_ids = [d for d in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, d))]
        
        for user_id in user_ids:
            user_dir = os.path.join(data_root_dir, str(user_id))
            if not os.path.isdir(user_dir):
                print(f"Warning: User directory {user_dir} not found")
                continue
                
            user_npz_files = [(user_id, f) for f in os.listdir(user_dir) 
                             if f.endswith('_direction_vectors.npz')]
            self.npz_files.extend(user_npz_files)
        
        print(f"Found {len(self.npz_files)} sequences from {len(user_ids)} users")
        
        self.expected_audio_length = int(n_poses / target_fps * 16000)
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(n_poses, target_fps)
        
        # Create lmdb environment
        map_size = 1024 * 200  # in MB
        map_size <<= 20  # in B
        self.lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def run(self):
        """Process all files and create LMDB database"""
        for user_id, npz_file in tqdm(self.npz_files, desc="Processing files"):
            self._process_file(user_id, npz_file)
        
        # Print stats
        with self.lmdb_env.begin() as txn:
            print('Total samples in LMDB: ', txn.stat()['entries'])
        
        # Close db
        self.lmdb_env.sync()
        self.lmdb_env.close()
        
    def _process_file(self, user_id, npz_file):
        """Process a single NPZ file and add samples to LMDB"""
        # In BEAT dataset, NPZ and WAV files are in the same folder with the same name
        base_name = npz_file.replace('_direction_vectors.npz', '')
        
        # Construct full paths
        user_dir = os.path.join(self.data_root_dir, str(user_id))
        npz_path = os.path.join(user_dir, npz_file)
        audio_path = os.path.join(user_dir, f"{base_name}.wav")
        
        # Check if files exist
        if not os.path.exists(npz_path):
            print(f"NPZ file {npz_path} not found")
            return
            
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found")
            return
        
        # Load direction vectors
        try:
            data = np.load(npz_path)
            dir_vectors = data['direction_vectors']  # Shape: (frames, connections, 3)
        except Exception as e:
            print(f"Error loading NPZ file {npz_path}: {e}")
            return
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return
        
        # Process direction vectors
        # 1. Resample to target fps
        resampled_vectors = resample_pose_seq(dir_vectors, self.original_fps, self.target_fps)
        resampled_length = len(resampled_vectors)
        
        # Calculate subdivisions
        if resampled_length <= self.n_poses:
            # Sequence is shorter than n_poses, process the entire sequence
            segments = [(0, min(resampled_length, self.n_poses))]
        else:
            # Create overlapping subdivisions
            segments = []
            num_subdivisions = max(1, (resampled_length - self.n_poses) // self.subdivision_stride + 1)
            for i in range(num_subdivisions):
                start_idx = i * self.subdivision_stride
                end_idx = start_idx + self.n_poses
                if end_idx <= resampled_length:
                    segments.append((start_idx, end_idx))
        
        # Process each segment
        for start_idx, end_idx in segments:
            segment_vectors = resampled_vectors[start_idx:end_idx]
            
            # Pad if needed (should rarely happen with proper subdivision)
            if len(segment_vectors) < self.n_poses:
                pad_length = self.n_poses - len(segment_vectors)
                segment_vectors = np.pad(segment_vectors, ((0, pad_length), (0, 0), (0, 0)), mode='edge')
            
            # Normalize if mean_dir_vec is provided
            if self.mean_dir_vec is not None:
                if self.mean_dir_vec.shape[-1] != 3:
                    # Reshape if the mean is flat
                    mean_dir_vec_reshaped = self.mean_dir_vec.reshape(-1, 3)
                    segment_vectors = segment_vectors - mean_dir_vec_reshaped
                else:
                    segment_vectors = segment_vectors - self.mean_dir_vec
            
            # Flatten the direction vectors for model consumption
            flat_vectors = segment_vectors.reshape(segment_vectors.shape[0], -1)
            
            # Process audio for this segment
            # Calculate audio start and end times
            duration_sec = len(dir_vectors) / self.original_fps
            audio_start_time = start_idx / self.target_fps
            audio_end_time = end_idx / self.target_fps
            
            # Extract the audio segment
            audio_start_idx = int(audio_start_time * sr)
            audio_end_idx = int(audio_end_time * sr)
            
            if audio_end_idx > len(audio):
                # Skip if audio segment is out of bounds
                continue
                
            audio_segment = audio[audio_start_idx:audio_end_idx]
            
            # Make sure audio is the right length
            audio_segment = make_audio_fixed_length(audio_segment, self.expected_audio_length)
            
            # Calculate spectrogram
            spectrogram = extract_melspectrogram(audio_segment)
            
            # Ensure spectrogram has correct length
            if spectrogram.shape[1] > self.expected_spectrogram_length:
                spectrogram = spectrogram[:, :self.expected_spectrogram_length]
            elif spectrogram.shape[1] < self.expected_spectrogram_length:
                pad_width = self.expected_spectrogram_length - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            
            # Create auxiliary info
            aux_info = {
                'vid': f"{user_id}_{base_name}",
                'start_frame_no': start_idx,
                'end_frame_no': end_idx,
                'start_time': audio_start_time,
                'end_time': audio_end_time,
                'is_correct_motion': True,
                'filtering_message': 'PASS'
            }
            
            # Save to LMDB
            with self.lmdb_env.begin(write=True) as txn:
                # Create dummy word sequence since we don't have text
                dummy_word_seq = []  # Just a placeholder

                # Ensure all numpy arrays have correct dtypes
                segment_vectors = segment_vectors.astype(np.float32)
                flat_vectors = flat_vectors.astype(np.float32)
                audio_segment = audio_segment.astype(np.float32)
                spectrogram = spectrogram.astype(np.float32)
                
                # Save
                k = '{:010}'.format(self.n_out_samples).encode('ascii')
                v = [dummy_word_seq, segment_vectors, flat_vectors, audio_segment, spectrogram, aux_info]
                
                try:
                    # Convert complex numpy structures to lists before serialization
                    serializable_v = custom_serialize(v)
                    v = pyarrow.serialize(serializable_v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1
                except Exception as e:
                    print(f"Serialization error: {e}")
                    print(f"Types: {[type(x) for x in v]}")
                    print(f"Shapes: motion {segment_vectors.shape}, dir_vec {flat_vectors.shape}, audio {audio_segment.shape}, spec {spectrogram.shape}")

def create_lmdb_for_user_groups(data_root_dir, out_lmdb_dir, n_poses=34, subdivision_stride=10, 
                               original_fps=60, target_fps=15, mean_dir_vec=None):
    """Create LMDB databases for individual users 1-5 and groups of 10, 20, 30 users"""
    
    # Create directories if they don't exist
    os.makedirs(out_lmdb_dir, exist_ok=True)
    
    # Individual users 1-5
    for user_id in range(1, 6):
        user_lmdb_dir = os.path.join(out_lmdb_dir, f"user_{user_id}_cache")
        if os.path.exists(user_lmdb_dir):
            print(f"Cache for user {user_id} already exists at {user_lmdb_dir}")
            continue
            
        print(f"Creating cache for user {user_id}")
        preprocessor = BeatDataPreprocessor(
            data_root_dir=data_root_dir,
            out_lmdb_dir=user_lmdb_dir,
            n_poses=n_poses,
            subdivision_stride=subdivision_stride,
            original_fps=original_fps,
            target_fps=target_fps,
            mean_dir_vec=mean_dir_vec,
            user_ids=[str(user_id)]
        )
        preprocessor.run()
    
    # Groups of users
    for n_users in [10, 20, 30]:
        group_lmdb_dir = os.path.join(out_lmdb_dir, f"users_{n_users}_cache")
        if os.path.exists(group_lmdb_dir):
            print(f"Cache for {n_users} users already exists at {group_lmdb_dir}")
            continue
            
        user_ids = [str(i) for i in range(1, n_users+1)]
        print(f"Creating cache for users 1-{n_users}")
        preprocessor = BeatDataPreprocessor(
            data_root_dir=data_root_dir,
            out_lmdb_dir=group_lmdb_dir,
            n_poses=n_poses,
            subdivision_stride=subdivision_stride,
            original_fps=original_fps,
            target_fps=target_fps,
            mean_dir_vec=mean_dir_vec,
            user_ids=user_ids
        )
        preprocessor.run()

if __name__ == "__main__":
    # Set paths
    beat_data_root = '/home/bsd/cospeech/DiffGesture/data/beat_english_v0.2.1'
    beat_out_dir = '/home/bsd/cospeech/DiffGesture/data/beat_english_v0.2.1/beat_cache'
    
    # Calculate mean direction vector if not available (you may need to adapt this)
    mean_dir_vec_path = '/home/bsd/cospeech/DiffGesture/data/beat_english_v0.2.1/mean_dir_vec.npy'
    
    # Check if mean_dir_vec exists, otherwise use Trinity's mean_dir_vec as a placeholder
    # NOTE: You should compute the proper mean for BEAT dataset
    if os.path.exists(mean_dir_vec_path):
        mean_dir_vec = np.load(mean_dir_vec_path)
    else:
        # Use Trinity's mean_dir_vec as a fallback
        mean_dir_vec = np.load('/home/bsd/cospeech/DiffGesture/data/trinity/mean_dir_vec.npy')
        print("WARNING: Using Trinity's mean_dir_vec as a placeholder. Consider computing BEAT's mean_dir_vec.")
    
    # Create LMDB databases for different user groups
    create_lmdb_for_user_groups(
        data_root_dir=beat_data_root,
        out_lmdb_dir=beat_out_dir,
        n_poses=34,
        subdivision_stride=10,
        original_fps=60, 
        target_fps=15,
        mean_dir_vec=mean_dir_vec
    )