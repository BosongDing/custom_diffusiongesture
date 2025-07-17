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

import scripts.utils.data_utils_expressive as utils
from scripts.model.vocab import Vocab
from scripts.data_loader.data_preprocessor_new import DataPreprocessor
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

class LMDBDataset(Dataset):
    def __init__(self, dataset_name=None,dataset_path=None, data_path_list=None, audio_path_list=None, n_poses=34, subdivision_stride=10, 
                 original_fps=60, target_fps=15, mean_dir_vec=None):
        """
        LMDB-based dataset for Trinity data
        
        Args:
            data_dir: Directory containing NPZ files
            audio_dir: Directory containing WAV files
            n_poses: Number of poses per sequence
            subdivision_stride: Stride for subdividing long sequences
            original_fps: Original frame rate of the motion data
            target_fps: Target frame rate for resampling
            mean_dir_vec: Mean direction vector for normalization
        """
        if dataset_name is None and dataset_path is None:
            raise ValueError("dataset_name or dataset_path must be provided")
        self.dataset_name = dataset_name
        self.data_path_list = data_path_list
        self.audio_path_list = audio_path_list
        self.n_poses = n_poses
        self.target_fps = target_fps
        self.mean_dir_vec = mean_dir_vec
        
        # Expected audio and spectrogram lengths
        self.expected_audio_length = int(round(n_poses / target_fps * 16000))
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(n_poses, target_fps)
        
        # Check for existing LMDB cache
        if dataset_path is None:
            preloaded_dir = "./data/"+dataset_name + '_cache'
            if "beat" in data_path_list[0]:
                preloaded_dir = "./data/beat_english_v0.2.1/"+dataset_name + '_cache'
        else:
            preloaded_dir = dataset_path
        if not os.path.exists(preloaded_dir):
            if data_path_list is None or audio_path_list is None:
                raise ValueError("data_path_list and audio_path_list must be provided if cache does not exist")
            logging.info('Creating the dataset cache...')
            preprocessor = DataPreprocessor(
                data_path_list=data_path_list,
                audio_path_list=audio_path_list,
                out_lmdb_dir=preloaded_dir,
                n_poses=n_poses,
                subdivision_stride=subdivision_stride,
                original_fps=original_fps,
                target_fps=target_fps,
                mean_dir_vec=mean_dir_vec
            )
            preprocessor.run()
        else:
            logging.info(f'Found the cache {preloaded_dir}')

        # Initialize LMDB environment
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
            
        logging.info(f"Loaded {self.n_samples} samples from {preloaded_dir}")
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)
            
            if sample is None:
                raise IndexError(f"Sample with index {idx} not found in the LMDB database")

            sample = pyarrow.deserialize(sample)
            word_seq, motion_seq, vec_seq, audio, spectrogram, aux_info = sample
        
        # To tensors
        dummy_word_seq = torch.tensor([0])  # Just a placeholder
        extended_word_seq = torch.zeros(self.n_poses, dtype=torch.long)  # Placeholder
        
        # Make sure we're working with numpy arrays
        vec_seq = np.array(vec_seq, dtype=np.float32)
        motion_seq = np.array(motion_seq, dtype=np.float32)
        audio = np.array(audio, dtype=np.float32)
        spectrogram = np.array(spectrogram, dtype=np.float32)
        
        # Convert to tensors
        vec_seq = torch.from_numpy(vec_seq).float()
        pose_seq = torch.from_numpy(motion_seq).float()
        audio = torch.from_numpy(audio).float()
        spectrogram = torch.from_numpy(spectrogram).float()
        
        # Ensure correct length
        do_clipping = False
        if do_clipping:
            audio = utils.data_utils_expressive.make_audio_fixed_length(audio, self.expected_audio_length)
            
            # Ensure spectrogram has correct length
            if len(spectrogram.shape) > 1 and spectrogram.shape[1] > self.expected_spectrogram_length:
                spectrogram = spectrogram[:, :self.expected_spectrogram_length]
            
            # Ensure pose sequences have the right length
            if vec_seq.shape[0] > self.n_poses:
                vec_seq = vec_seq[:self.n_poses]
            if pose_seq.shape[0] > self.n_poses:
                pose_seq = pose_seq[:self.n_poses]
        
        # Add validation checks
        # Check for NaN values
        if torch.isnan(pose_seq).any() or torch.isnan(vec_seq).any():
            print(f"Warning: NaN values detected in sample {idx}")
            # Replace NaNs with zeros
            pose_seq = torch.nan_to_num(pose_seq)
            vec_seq = torch.nan_to_num(vec_seq)
        
        # Check for extreme values that might indicate normalization issues
        max_val = 10.0  # Set a reasonable threshold
        if torch.abs(pose_seq).max() > max_val or torch.abs(vec_seq).max() > max_val:
            print(f"Warning: Extreme values detected in sample {idx}, max: {torch.abs(pose_seq).max()}")
            # Clip extreme values
            pose_seq = torch.clamp(pose_seq, -max_val, max_val)
            vec_seq = torch.clamp(vec_seq, -max_val, max_val)
        
        return dummy_word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info

# def test_trinity_lmdb_dataset():
#     """Test the LMDB-based Trinity dataset"""
#     # Get mean direction vector
#     mean_dir_vec = np.load('/home/bsd/cospeech/DiffGesture/data/trinity/mean_dir_vec.npy')
    
#     # Create dataset
#     dataset = TrinityLMDBDataset(
#         data_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRec',
#         audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRecAudio',
#         n_poses=34,
#         subdivision_stride=10,
#         original_fps=60,
#         target_fps=15,
#         mean_dir_vec=mean_dir_vec
#     )
    
#     # Get a sample
#     sample = dataset[0]
#     word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
    
#     # Print shapes
#     print("Word sequence shape:", word_seq.shape)
#     print("Extended word sequence shape:", extended_word_seq.shape)
#     print("Pose sequence shape:", pose_seq.shape)
#     print("Direction vector shape:", vec_seq.shape)
#     print("Audio shape:", audio.shape)
#     print("Spectrogram shape:", spectrogram.shape)
#     print("Aux info:", aux_info)

# if __name__ == "__main__":
#     test_trinity_lmdb_dataset()