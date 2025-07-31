import os
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import lmdb
import pyarrow

# Note: TED Expressive datasets don't actually use language models
# (word sequences are replaced with dummy tensors in collate_fn)

class DummyLanguageModel:
    """Minimal dummy language model for TED Expressive datasets that don't use text"""
    def __init__(self):
        self.SOS_token = 1
        self.EOS_token = 2
        self.n_words = 3
    
    def get_word_index(self, word):
        return 0  # Return dummy index

class UnifiedMotionDataset(Dataset):
    """
    Unified dataset that can load Trinity, BEAT, and TED Expressive datasets together
    Keeps original dimensions and provides dataset labels for routing to appropriate encoders
    """
    
    def __init__(self, dataset_configs, target_fps=15, n_poses=34):
        """
        Args:
            dataset_configs: List of dictionaries with dataset configurations
                Example: [
                    {
                        'name': 'trinity',
                        'path': './data/trinity_all_cache',
                        'type': 'lmdb_new',
                        'pose_dim': 129,
                        'weight': 1.0  # Optional: sampling weight for this dataset
                    },
                    {
                        'name': 'beat', 
                        'path': './data/beat_english_v0.2.1/beat_all_cache',
                        'type': 'lmdb_new',
                        'pose_dim': 177,
                        'weight': 1.0
                    },
                    {
                        'name': 'ted_expressive',
                        'path': './data/ted_expressive_dataset/train',
                        'type': 'lmdb_expressive', 
                        'pose_dim': 126,
                        'weight': 1.0
                    }
                ]
        """
        self.dataset_configs = dataset_configs
        self.target_fps = target_fps
        self.n_poses = n_poses
        
        # Load all datasets and track their samples
        self.datasets = []
        self.sample_indices = []  # (dataset_idx, sample_idx, dataset_name, pose_dim)
        
        successful_datasets = []
        for dataset_idx, config in enumerate(dataset_configs):
            print(f"Loading dataset: {config['name']} from {config['path']}")
            try:
                dataset = self._load_single_dataset(config)
                
                # Test loading a sample to make sure the dataset works
                if len(dataset) > 0:
                    test_sample = dataset[0]
                    # Check if all required elements are present
                    if len(test_sample) >= 7:
                        self.datasets.append(dataset)
                        successful_datasets.append(dataset_idx)
                        
                        # Track which dataset each sample belongs to
                        for sample_idx in range(len(dataset)):
                            self.sample_indices.append((
                                len(self.datasets) - 1,  # Use index in successful datasets
                                sample_idx, 
                                config['name'],
                                config['pose_dim']
                            ))
                        
                        print(f"  Successfully loaded {len(dataset)} samples")
                    else:
                        print(f"  Dataset {config['name']} sample format invalid (got {len(test_sample)} elements, expected 7)")
                else:
                    print(f"  Dataset {config['name']} is empty")
                    
            except Exception as e:
                print(f"  Failed to load dataset {config['name']}: {e}")
                import traceback
                print(f"  Error details: {traceback.format_exc()}")
                continue
        
        # Keep only successful dataset configs
        self.dataset_configs = [self.dataset_configs[i] for i in successful_datasets]
        
        if len(self.datasets) == 0:
            raise ValueError("No datasets were successfully loaded!")
        
        # TED Expressive datasets don't actually need language models (word sequences are ignored)
        # Just ensure they have a minimal setup to avoid crashes
        
        print(f"Unified dataset loaded with {len(self.sample_indices)} total samples from {len(self.datasets)} datasets")
        self._print_dataset_stats()
    
    def _load_single_dataset(self, config):
        """Load a single dataset based on its type"""
        if config['type'] == 'lmdb_new':
            # For Trinity and BEAT datasets
            from scripts.data_loader.lmdb_data_loader_new import LMDBDataset
            return LMDBDataset(dataset_path=config['path'])
            
        elif config['type'] == 'lmdb_expressive':
            # For TED Expressive dataset
            from scripts.data_loader.lmdb_data_loader_expressive import SpeechMotionDataset
            
            # Load mean vectors for expressive dataset
            mean_pose_path = os.path.join(os.path.dirname(config['path']), 'mean_pose.npy')
            mean_dir_vec_path = os.path.join(os.path.dirname(config['path']), 'mean_dir_vec.npy')
            
            # Use default values if mean files don't exist
            mean_pose = np.zeros(config['pose_dim'] // 3 * 3) if not os.path.exists(mean_pose_path) else np.load(mean_pose_path)
            mean_dir_vec = np.zeros(config['pose_dim']) if not os.path.exists(mean_dir_vec_path) else np.load(mean_dir_vec_path)
            
            try:
                dataset = SpeechMotionDataset(
                    lmdb_dir=config['path'],
                    n_poses=self.n_poses,
                    subdivision_stride=10,
                    pose_resampling_fps=self.target_fps,
                    mean_pose=mean_pose,
                    mean_dir_vec=mean_dir_vec
                )
                
                # Set dummy language model (word sequences are ignored anyway)
                dataset.set_lang_model(DummyLanguageModel())
                return dataset
            except Exception as e:
                print(f"  Error loading TED Expressive dataset: {e}")
                # Try without language model if that's the issue
                try:
                    dataset = SpeechMotionDataset(
                        lmdb_dir=config['path'],
                        n_poses=self.n_poses,
                        subdivision_stride=10,
                        pose_resampling_fps=self.target_fps,
                        mean_pose=mean_pose,
                        mean_dir_vec=mean_dir_vec,
                        speaker_model=0  # Disable speaker model
                    )
                    # Set dummy language model (word sequences are ignored anyway)
                    dataset.set_lang_model(DummyLanguageModel())
                    return dataset
                except Exception as e2:
                    print(f"  Failed even with speaker_model=0: {e2}")
                    raise e2
        else:
            raise ValueError(f"Unknown dataset type: {config['type']}")
    

    
    def _print_dataset_stats(self):
        """Print statistics about the unified dataset"""
        dataset_counts = {}
        dataset_dims = {}
        
        for _, _, dataset_name, pose_dim in self.sample_indices:
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            dataset_dims[dataset_name] = pose_dim
        
        print("\nDataset composition:")
        for name, count in dataset_counts.items():
            percentage = (count / len(self.sample_indices)) * 100
            print(f"  {name}: {count} samples ({percentage:.1f}%) - dim: {dataset_dims[name]}")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        # Get which dataset and sample index
        dataset_idx, sample_idx, dataset_name, pose_dim = self.sample_indices[idx]
        dataset = self.datasets[dataset_idx]
        
        # Get the sample from the appropriate dataset
        sample = dataset[sample_idx]
        
        # Unpack the sample (format may vary slightly between datasets)
        word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
        
        # Ensure all data is in tensor format (handle TED Expressive list format)
        word_seq = self._ensure_tensor(word_seq)
        extended_word_seq = self._ensure_tensor(extended_word_seq)
        pose_seq = self._ensure_tensor(pose_seq)
        vec_seq = self._ensure_tensor(vec_seq)
        audio = self._ensure_tensor(audio)
        spectrogram = self._ensure_tensor(spectrogram)
        
        # Create dataset identifier
        dataset_info = {
            'name': dataset_name,
            'idx': dataset_idx,
            'pose_dim': pose_dim,
            'sample_idx': sample_idx
        }
        
        # Add dataset information to aux_info
        if isinstance(aux_info, dict):
            aux_info = aux_info.copy()
        else:
            aux_info = {}
        
        aux_info.update({
            'dataset_info': dataset_info
        })
        
        return (word_seq, extended_word_seq, pose_seq, vec_seq, 
                audio, spectrogram, aux_info, dataset_info)
    
    def _ensure_tensor(self, data):
        """Ensure data is a PyTorch tensor, converting from list/numpy if needed"""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            # For scalar values or other types, try converting to tensor
            try:
                return torch.tensor(data, dtype=torch.float32)
            except:
                # If conversion fails, return as-is (might be non-numeric data like aux_info)
                return data
    
    def get_dataset_info(self):
        """Return information about the datasets"""
        dataset_stats = {}
        for config in self.dataset_configs:
            name = config['name']
            count = sum(1 for _, _, dataset_name, _ in self.sample_indices if dataset_name == name)
            dataset_stats[name] = {
                'count': count,
                'pose_dim': config['pose_dim'],
                'type': config['type'],
                'path': config['path']
            }
        
        return {
            'datasets': dataset_stats,
            'total_samples': len(self.sample_indices),
            'num_datasets': len(self.dataset_configs)
        }
    
    def get_samples_by_dataset(self, dataset_name):
        """Get all sample indices for a specific dataset"""
        return [i for i, (_, _, name, _) in enumerate(self.sample_indices) if name == dataset_name]
    
    def filter_by_dataset(self, dataset_names):
        """Create a filtered view of the dataset with only specified datasets"""
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        filtered_indices = [i for i, (_, _, name, _) in enumerate(self.sample_indices) if name in dataset_names]
        
        # Create a new dataset view
        class FilteredDatasetView:
            def __init__(self, parent_dataset, filtered_indices):
                self.parent = parent_dataset
                self.indices = filtered_indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]
        
        return FilteredDatasetView(self, filtered_indices)

def unified_collate_fn(batch):
    """Custom collate function for unified dataset"""
    # Separate samples by dataset to handle different dimensions
    dataset_batches = {}
    
    for sample in batch:
        word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info, dataset_info = sample
        dataset_name = dataset_info['name']
        
        if dataset_name not in dataset_batches:
            dataset_batches[dataset_name] = []
        
        dataset_batches[dataset_name].append(
            (word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info)
        )
    
    # Collate each dataset separately
    collated_batches = {}
    for dataset_name, samples in dataset_batches.items():
        # Unpack samples
        word_seqs, extended_word_seqs, pose_seqs, vec_seqs, audios, spectrograms, aux_infos = zip(*samples)
        
        # Collate using default collate
        try:
            collated_batch = {
                'word_seqs': default_collate(word_seqs),
                'extended_word_seqs': default_collate(extended_word_seqs),
                'pose_seqs': default_collate(pose_seqs),
                'vec_seqs': default_collate(vec_seqs),
                'audios': default_collate(audios),
                'spectrograms': default_collate(spectrograms),
                'aux_infos': _collate_aux_info(aux_infos),
                'batch_size': len(samples)
            }
            collated_batches[dataset_name] = collated_batch
        except Exception as e:
            print(f"Error collating batch for dataset {dataset_name}: {e}")
            continue
    
    return collated_batches

def _collate_aux_info(aux_infos):
    """Helper function to collate aux_info dictionaries"""
    if not aux_infos or not aux_infos[0]:
        return {}
    
    aux_info_batch = {}
    keys = aux_infos[0].keys()
    
    for key in keys:
        try:
            aux_info_batch[key] = default_collate([aux[key] for aux in aux_infos])
        except:
            # If collation fails (e.g., mixed types), keep as list
            aux_info_batch[key] = [aux[key] for aux in aux_infos]
    
    return aux_info_batch

def create_unified_dataset(trinity_path=None, beat_path=None, ted_expressive_path=None, **kwargs):
    """
    Convenience function to create a unified dataset with common paths
    """
    dataset_configs = []
    
    if trinity_path and os.path.exists(trinity_path):
        dataset_configs.append({
            'name': 'trinity',
            'path': trinity_path,
            'type': 'lmdb_new',
            'pose_dim': 129,  # Update based on your actual Trinity dimension
            'weight': kwargs.get('trinity_weight', 1.0)
        })
    
    if beat_path and os.path.exists(beat_path):
        dataset_configs.append({
            'name': 'beat',
            'path': beat_path, 
            'type': 'lmdb_new',
            'pose_dim': 177,  # Update based on your actual BEAT dimension
            'weight': kwargs.get('beat_weight', 1.0)
        })
    
    if ted_expressive_path and os.path.exists(ted_expressive_path):
        dataset_configs.append({
            'name': 'ted_expressive',
            'path': ted_expressive_path,
            'type': 'lmdb_expressive',
            'pose_dim': 126,  # Update based on your actual TED Expressive dimension
            'weight': kwargs.get('ted_weight', 1.0)
        })
    
    if not dataset_configs:
        raise ValueError("At least one valid dataset path must be provided")
    
    return UnifiedMotionDataset(dataset_configs, **kwargs)

class DatasetComparator:
    """Utility class to compare individual datasets with unified dataset"""
    
    def __init__(self, unified_dataset):
        self.unified_dataset = unified_dataset
        self.individual_datasets = {}
        
        # Load individual datasets for comparison
        for config in unified_dataset.dataset_configs:
            try:
                dataset = unified_dataset._load_single_dataset(config)
                self.individual_datasets[config['name']] = dataset
            except Exception as e:
                print(f"Could not load individual dataset {config['name']}: {e}")
    
    def compare_samples(self, num_samples=5):
        """Compare samples between individual and unified datasets"""
        results = {}
        
        for dataset_name in self.individual_datasets.keys():
            print(f"\nComparing {dataset_name} dataset:")
            
            # Get unified dataset samples for this dataset
            unified_indices = self.unified_dataset.get_samples_by_dataset(dataset_name)
            individual_dataset = self.individual_datasets[dataset_name]
            
            matches = 0
            mismatches = 0
            
            for i in range(min(num_samples, len(unified_indices), len(individual_dataset))):
                # Get sample from individual dataset
                individual_sample = individual_dataset[i]
                
                # Get corresponding sample from unified dataset
                unified_idx = unified_indices[i]
                unified_sample = self.unified_dataset[unified_idx]
                
                # Compare (excluding dataset_info which is added by unified dataset)
                individual_pose = individual_sample[2]  # pose_seq
                unified_pose = unified_sample[2]  # pose_seq
                
                if torch.allclose(individual_pose, unified_pose, atol=1e-6):
                    matches += 1
                else:
                    mismatches += 1
                    print(f"  Sample {i}: MISMATCH")
                    print(f"    Individual shape: {individual_pose.shape}")
                    print(f"    Unified shape: {unified_pose.shape}")
                    print(f"    Max diff: {torch.max(torch.abs(individual_pose - unified_pose))}")
            
            results[dataset_name] = {
                'matches': matches,
                'mismatches': mismatches,
                'total_compared': matches + mismatches
            }
            
            print(f"  Results: {matches} matches, {mismatches} mismatches")
        
        return results 