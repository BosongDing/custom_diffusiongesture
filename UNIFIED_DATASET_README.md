# Unified Dataset Loader

This module provides a unified interface to load and combine Trinity, BEAT, and TED Expressive datasets for co-speech gesture generation training.

## Features

- **Multi-dataset support**: Load Trinity, BEAT, and TED Expressive datasets simultaneously
- **Dimension preservation**: Each dataset keeps its original dimensions (no forced unification)
- **Automatic routing**: Batches are separated by dataset type for appropriate encoder routing
- **Filtering**: Train on specific subsets of datasets
- **Consistency validation**: Compare unified dataset with individual datasets
- **Performance optimized**: Efficient LMDB-based loading

## Quick Start

### Basic Usage

```python
from scripts.data_loader.unified_data_loader import create_unified_dataset, unified_collate_fn
from torch.utils.data import DataLoader

# Create unified dataset
dataset = create_unified_dataset(
    trinity_path="./data/trinity_all_cache",
    beat_path="./data/beat_english_v0.2.1/beat_all_cache",
    ted_expressive_path="./data/ted_expressive_dataset/train"
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=unified_collate_fn,
    num_workers=4
)

# Iterate through batches
for batch_dict in dataloader:
    for dataset_name, dataset_batch in batch_dict.items():
        pose_seqs = dataset_batch['pose_seqs']
        audios = dataset_batch['audios']
        batch_size = dataset_batch['batch_size']
        
        # Route to appropriate encoder based on dataset_name
        if dataset_name == 'trinity':
            output = trinity_encoder(pose_seqs, audios)
        elif dataset_name == 'beat':
            output = beat_encoder(pose_seqs, audios)
        # ... etc
```

### Dataset Filtering

```python
# Train on only Trinity data
trinity_only = dataset.filter_by_dataset('trinity')

# Train on Trinity and BEAT only
multi_dataset = dataset.filter_by_dataset(['trinity', 'beat'])

# Create filtered dataloader
filtered_loader = DataLoader(
    trinity_only,
    batch_size=16,
    collate_fn=unified_collate_fn
)
```

## File Structure

```
DiffGesture/
├── scripts/data_loader/
│   └── unified_data_loader.py          # Main unified dataset implementation
├── test_unified_dataset.py             # Comprehensive test suite
├── example_unified_usage.py            # Usage examples
└── UNIFIED_DATASET_README.md          # This file
```

## Dataset Configuration

Each dataset is configured with:

```python
dataset_config = {
    'name': 'trinity',                   # Dataset identifier
    'path': './data/trinity_all_cache',  # Path to LMDB cache
    'type': 'lmdb_new',                 # Dataset type (lmdb_new or lmdb_expressive)
    'pose_dim': 129,                    # Pose dimension for this dataset
    'weight': 1.0                       # Optional: sampling weight
}
```

### Supported Dataset Types

- **lmdb_new**: For Trinity and BEAT datasets
- **lmdb_expressive**: For TED Expressive datasets

## Batch Structure

The unified collate function returns a dictionary where keys are dataset names:

```python
batch_dict = {
    'trinity': {
        'word_seqs': tensor,        # Word sequences
        'extended_word_seqs': tensor,   # Extended word sequences  
        'pose_seqs': tensor,        # Pose sequences [batch_size, seq_len, pose_dim]
        'vec_seqs': tensor,         # Direction vectors [batch_size, seq_len, pose_dim]
        'audios': tensor,           # Audio data [batch_size, audio_len]
        'spectrograms': tensor,     # Mel spectrograms [batch_size, mel_bins, time]
        'aux_infos': dict,          # Auxiliary information
        'batch_size': int           # Number of samples in this dataset batch
    },
    'beat': {
        # Same structure but potentially different dimensions
    }
}
```

## Testing

Run the comprehensive test suite:

```bash
cd DiffGesture
python test_unified_dataset.py --trinity_path ./data/trinity_all_cache --beat_path ./data/beat_english_v0.2.1/beat_all_cache
```

Test options:
- `--skip_individual`: Skip testing individual datasets
- `--skip_consistency`: Skip consistency validation
- `--skip_performance`: Skip performance benchmarks

## Examples

Run the usage examples:

```bash
cd DiffGesture
python example_unified_usage.py
```

This will show:
- Basic unified dataset creation and usage
- Dataset filtering examples
- Training loop simulation
- Dataset statistics analysis

## Key Classes

### UnifiedMotionDataset

Main dataset class that combines multiple motion datasets.

**Methods:**
- `get_dataset_info()`: Get statistics about loaded datasets
- `get_samples_by_dataset(name)`: Get sample indices for a specific dataset
- `filter_by_dataset(names)`: Create filtered view of specific datasets

### DatasetComparator

Utility class to validate that unified dataset returns identical data to individual datasets.

**Methods:**
- `compare_samples(num_samples)`: Compare samples between individual and unified datasets

## Performance Considerations

1. **Memory Usage**: Each dataset is loaded separately, so memory usage scales with the number of datasets
2. **LMDB Caching**: Relies on pre-computed LMDB caches for fast loading
3. **Batch Size**: Effective batch size is split across datasets in each batch
4. **Data Locality**: Samples from the same dataset are grouped together in batches

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure `scripts` directory is in your Python path
2. **Missing Datasets**: Check that LMDB cache directories exist and are readable
3. **Dimension Mismatches**: Verify pose dimensions in dataset configs match actual data
4. **Memory Issues**: Reduce batch size or use fewer datasets simultaneously

### Validation

Always run the test suite when setting up new datasets:

```bash
python test_unified_dataset.py
```

Look for:
- ✓ All datasets load successfully
- ✓ Consistency tests pass (unified == individual)
- ✓ DataLoader works without errors
- ✓ Filtering produces correct sample counts

## Integration with Training

For training with multiple datasets:

1. **Separate Encoders**: Use different encoders for different datasets based on their pose dimensions
2. **Loss Weighting**: Weight losses from different datasets based on dataset size or importance
3. **Gradient Accumulation**: Accumulate gradients across different dataset batches
4. **Learning Rate Scheduling**: Consider different learning rates for different datasets

Example training integration:

```python
def train_step(batch_dict, models, optimizers):
    total_loss = 0
    
    for dataset_name, dataset_batch in batch_dict.items():
        # Get appropriate model and optimizer
        model = models[dataset_name]
        optimizer = optimizers[dataset_name]
        
        # Forward pass
        loss = model.get_loss(
            dataset_batch['vec_seqs'],
            dataset_batch['pose_seqs'][:, :4],  # pre_poses
            dataset_batch['audios']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss
```

## Data Format Compatibility

The unified dataset ensures compatibility across different dataset formats:

- **Trinity**: NPZ files converted to LMDB via `lmdb_data_loader_new`
- **BEAT**: NPZ files converted to LMDB via `lmdb_data_loader_new`  
- **TED Expressive**: Original LMDB format via `lmdb_data_loader_expressive`

All datasets provide the same 7-tuple output format:
`(word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info)` 