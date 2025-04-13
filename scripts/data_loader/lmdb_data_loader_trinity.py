import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import logging
import os
import pickle

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import utils.data_utils_expressive
from model.vocab import Vocab
from data_loader.data_preprocessor_expressive import DataPreprocessor
import pyarrow
import copy
def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info

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

class TrinityDataset(Dataset):
    def __init__(self, data_dir, audio_dir, n_poses=34, n_pre_poses=4, 
                 original_fps=60, target_fps=15, mean_dir_vec=None,
                 subdivision_stride=10):
        """
        Custom dataset for the Trinity data
        
        Args:
            data_dir: Directory containing NPZ files
            audio_dir: Directory containing WAV files
            n_poses: Number of poses per sequence
            n_pre_poses: Number of initial poses to use as context
            original_fps: Original frame rate of the motion data
            target_fps: Target frame rate for resampling
            mean_dir_vec: Mean direction vector for normalization
            subdivision_stride: Stride for subdividing long sequences
        """
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.n_poses = n_poses
        self.n_pre_poses = n_pre_poses
        self.original_fps = original_fps
        self.target_fps = target_fps
        self.mean_dir_vec = mean_dir_vec
        self.subdivision_stride = subdivision_stride
        
        # Get all NPZ files
        self.npz_files = [f for f in os.listdir(data_dir) if f.endswith('_direction_vectors.npz')]
        print(f"Found {len(self.npz_files)} sequences")
        
        # Create a list of all samples (sequence, start frame)
        self.samples = []
        
        for npz_file in self.npz_files:
            # Load data to get sequence length
            npz_path = os.path.join(data_dir, npz_file)
            data = np.load(npz_path)
            dir_vectors = data['direction_vectors']
            
            # Calculate resampled length
            resampled_length = int(dir_vectors.shape[0] * target_fps / original_fps)
            
            # Calculate number of subdivisions
            if resampled_length <= n_poses:
                # Sequence is shorter than n_poses, so use the entire sequence
                self.samples.append((npz_file, 0))
            else:
                # Create overlapping subdivisions
                num_subdivisions = max(1, (resampled_length - n_poses) // subdivision_stride + 1)
                for i in range(num_subdivisions):
                    start_idx = i * subdivision_stride
                    if start_idx + n_poses <= resampled_length:
                        self.samples.append((npz_file, start_idx))
        
        print(f"Created {len(self.samples)} samples from {len(self.npz_files)} sequences")
        
        # Expected audio and spectrogram lengths
        self.expected_audio_length = int(n_poses / target_fps * 16000)
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(n_poses, target_fps)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        npz_file, start_idx = self.samples[idx]
        base_name = npz_file.replace('_direction_vectors.npz', '')
        
        # Load direction vectors
        npz_path = os.path.join(self.data_dir, npz_file)
        data = np.load(npz_path)
        dir_vectors = data['direction_vectors']  # Shape: (frames, connections, 3)
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, f"{base_name}.wav")
        audio, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
        
        # Process direction vectors
        # 1. Resample to target fps
        resampled_vectors = resample_pose_seq(dir_vectors, self.original_fps, self.target_fps)
        
        # 2. Extract the segment we want
        end_idx = start_idx + self.n_poses
        segment_vectors = resampled_vectors[start_idx:end_idx]
        
        # Pad if needed (should rarely happen with proper subdivision)
        if len(segment_vectors) < self.n_poses:
            pad_length = self.n_poses - len(segment_vectors)
            segment_vectors = np.pad(segment_vectors, ((0, pad_length), (0, 0), (0, 0)), mode='edge')
        
        # 3. Normalize if mean_dir_vec is provided
        if self.mean_dir_vec is not None:
            if self.mean_dir_vec.shape[-1] != 3:
                # Reshape if the mean is flat
                mean_dir_vec_reshaped = self.mean_dir_vec.reshape(-1, 3)
                segment_vectors = segment_vectors - mean_dir_vec_reshaped
            else:
                segment_vectors = segment_vectors - self.mean_dir_vec
            
        # 4. Flatten the direction vectors
        flat_vectors = segment_vectors.reshape(segment_vectors.shape[0], -1)
        
        # Process audio
        # 1. Calculate audio start and end times
        duration_sec = len(dir_vectors) / self.original_fps
        audio_start_time = start_idx / self.target_fps
        audio_end_time = end_idx / self.target_fps
        
        # 2. Extract the audio segment
        audio_start_idx = int(audio_start_time * sr)
        audio_end_idx = int(audio_end_time * sr)
        audio_segment = audio[audio_start_idx:audio_end_idx]
        
        # 3. Make sure audio is the right length
        audio_segment = make_audio_fixed_length(audio_segment, self.expected_audio_length)
        
        # 4. Calculate spectrogram
        spectrogram = extract_melspectrogram(audio_segment)
        
        # Ensure spectrogram has correct length
        if spectrogram.shape[1] > self.expected_spectrogram_length:
            spectrogram = spectrogram[:, :self.expected_spectrogram_length]
        elif spectrogram.shape[1] < self.expected_spectrogram_length:
            pad_width = self.expected_spectrogram_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        
        # Create dummy word sequence since we don't have text
        dummy_word_seq = torch.tensor([0])  # Just a placeholder
        extended_word_seq = torch.zeros(self.n_poses, dtype=torch.long)  # Placeholder
        
        # Create auxiliary info
        aux_info = {
            'vid': base_name,
            'start_frame_no': start_idx,
            'end_frame_no': end_idx,
            'start_time': audio_start_time,
            'end_time': audio_end_time,
            'is_correct_motion': True,
            'filtering_message': 'PASS'
        }
        
        # Convert to torch tensors
        pose_seq = torch.from_numpy(flat_vectors).float()
        vec_seq = pose_seq.clone()  # In this case they're the same
        audio_tensor = torch.from_numpy(audio_segment).float()
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        
        return dummy_word_seq, extended_word_seq, pose_seq, vec_seq, audio_tensor, spectrogram_tensor, aux_info
    
# if __name__ == "__main__":
#     import numpy as np
#     sample_file = "/home/bsd/cospeech/DiffGesture/data/trinity/allRec/Recording_001_direction_vectors.npz"
#     data = np.load(sample_file)
#     print("Keys in the NPZ file:", list(data.keys()))
#     print("Shape of direction_vectors:", data['direction_vectors'].shape)
#     if 'joint_positions' in data:
#         print("Shape of joint_positions:", data['joint_positions'].shape)
#     print("Number of connections:", len(data['connections']))


def test_trinity_dataset():
    # Get mean direction vector
    # Replace with your computed mean dir vec or None if you haven't computed it yet
    mean_dir_vec = None
    
    # Create dataset
    dataset = TrinityDataset(
        data_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRec',
        audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRecAudio',
        n_poses=34,
        n_pre_poses=4,
        original_fps=60,
        target_fps=15,
        mean_dir_vec=mean_dir_vec,
        subdivision_stride=10
    )
    
    # Get a sample
    sample = dataset[0]
    word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
    
    # Print shapes
    print("Word sequence shape:", word_seq.shape)
    print("Extended word sequence shape:", extended_word_seq.shape)
    print("Pose sequence shape:", pose_seq.shape)
    print("Direction vector shape:", vec_seq.shape)
    print("Audio shape:", audio.shape)
    print("Spectrogram shape:", spectrogram.shape)
    print("Aux info:", aux_info)

if __name__ == "__main__":
    test_trinity_dataset()